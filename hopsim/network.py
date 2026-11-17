"""
Network model: m parallel paths, each a FIFO bottleneck queue whose service
rate (capacity) is piecewise constant over slots of `slot` seconds, followed
by a fixed one-way propagation delay tau/2 (and tau/2 for the ACK back).

This mirrors the paper's ns-3 setup (Sec. VI): PointToPointChannel paths
whose capacity follows a trace that changes every 25 ms, large buffers (no
drops), MTU-sized packets, min RTT 25 ms.

The queue is simulated exactly with the cumulative service curve
B(t) = integral of C(u) du: a packet whose last bit needs cumulative service
b departs at B^{-1}(b). This makes a whole 30 s run cost a few vector ops
per block.
"""

import numpy as np

MBPS = 1e6


class Path:
    def __init__(self, cap_bps, slot, tau):
        """cap_bps: capacity per slot (bits/s); tau: min RTT (s)."""
        self.C = np.maximum(np.asarray(cap_bps, dtype=float), 1e3)  # avoid div by zero
        self.slot = slot
        self.tau = tau
        # cumulative bits served at each slot boundary
        self.cum = np.concatenate([[0.0], np.cumsum(self.C * slot)])
        self.last_bits = 0.0          # cumulative service level of the last queued bit

    def B(self, t):
        s = min(int(t // self.slot), len(self.C) - 1)
        return self.cum[s] + self.C[s] * (t - s * self.slot)

    def Binv(self, b):
        b = np.atleast_1d(b)
        s = np.searchsorted(self.cum, b, side="right") - 1
        s = np.clip(s, 0, len(self.C) - 1)
        return s * self.slot + (b - self.cum[s]) / self.C[s]

    def backlog_bits(self, t):
        """True number of bits still queued at time t."""
        return max(0.0, self.last_bits - self.B(t))

    def send_burst(self, t, n, pkt_bits):
        """Enqueue n packets at time t; return their bottleneck departure times."""
        if n <= 0:
            return np.empty(0)
        start = max(self.last_bits, self.B(t))
        b = start + pkt_bits * np.arange(1, n + 1)
        self.last_bits = b[-1]
        return self.Binv(b)


# --------------------------------------------------------------------------- capacity traces
def gaussian_trace(rng, n_slots, mean_mbps, std_mbps, floor_mbps=0.5):
    c = rng.normal(mean_mbps, std_mbps, n_slots)
    return np.maximum(c, floor_mbps) * MBPS


def correlated_traces(rng, n_slots, means, stds, rho, floor_mbps=0.5):
    """Gaussian capacities with a common factor: corr(C_i, C_j) = rho.

    Models paths that share a bottleneck (same backhaul, same AP, same
    blockage event) - the case HOP's independence assumption ignores.
    """
    z = rng.normal(size=n_slots)
    out = []
    for mu, sd in zip(means, stds):
        e = rng.normal(size=n_slots)
        c = mu + sd * (np.sqrt(rho) * z + np.sqrt(1 - rho) * e)
        out.append(np.maximum(c, floor_mbps) * MBPS)
    return out


def blockage_trace(rng, n_slots, mean_mbps, std_mbps, p_block=0.01, p_unblock=0.25,
                   blocked_mbps=1.0, ar=0.7):
    """mmWave-like path: AR(1) Gaussian capacity plus Markov blockage outages.

    p_block / p_unblock are per-slot transition probabilities (mean outage
    1/p_unblock slots). Captures the 'sudden drop' behaviour of the paper's
    WiGig trace (Fig. 10) that motivates SKIP's step/outage modes.
    """
    c = np.empty(n_slots)
    x = 0.0
    blocked = False
    for i in range(n_slots):
        x = ar * x + np.sqrt(1 - ar * ar) * rng.normal()
        if blocked:
            blocked = rng.random() > p_unblock
        else:
            blocked = rng.random() < p_block
        c[i] = blocked_mbps if blocked else max(mean_mbps + std_mbps * x, 0.5)
    return c * MBPS
