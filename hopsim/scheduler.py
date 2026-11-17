"""
Scheduling: lateness probabilities, block failure probability P_e, JUMP,
and the team's extensions.

Notation (paper Sec. IV): a block of k packets; schedule s = (s_1..s_m)
packets on m paths; lambda_i(j) = P(j-th packet on path i is late);
d_i = packets on time on path i; the block succeeds iff sum_i d_i >= k.

Three ways to get P_e(s, k, T)
------------------------------
1. pe_enumerate   - the paper's literal method: enumerate every delivery
                    vector d in D_k(s) (eq. 3) and sum products of psi (eq. 4).
                    O(prod (s_i+1)) = O(k^m) per schedule.
2. pe_conv(product=True)  - SAME formula as the paper (psi of eq. 5), but the
                    sum over D_k is computed as a convolution, because the
                    paths are independent: sum_{d in D_k} prod_i psi_i(d_i)
                    is the upper tail of psi_1 * psi_2 * ... * psi_m.
                    O(m k^2) - polynomial in the number of paths.
3. pe_conv(product=False) - the team's EXACT per-path distribution.
                    Under the paper's own model the on-time events are nested
                    (packet j+1 on time => packet j on time, FIFO + one
                    capacity variable), so
                        P(d_i = d) = lambda(d+1) - lambda(d)   (exact)
                    whereas eq. (5) uses (1 - lambda(d)) lambda(d+1), which is
                    >= the exact value by lambda(d)(1 - lambda(d+1)) and is not
                    a probability distribution (it sums to more than 1).
                    Eq. (5) therefore over-estimates on-time delivery, i.e. it
                    UNDER-estimates P_e, making JUMP add too little FEC.
"""

import itertools
import math

import numpy as np
from scipy.special import ndtr

PKT_BITS = 1500 * 8


# =========================================================================== lateness
def lateness(n, backlog_bits, window, mean, std, p_out=0.0):
    """lambda(j) for j = 1..n on one path (eq. 34 with the condition of eq. 27).

    Packet j is on time iff the path's capacity over the usable window
    W = T - tau/2 is at least (backlog + j L) / W.
    """
    j = np.arange(1, n + 1)
    cmin = (backlog_bits + j * PKT_BITS) / max(window, 1e-6)
    return p_out + (1.0 - p_out) * ndtr((cmin - mean) / max(std, 1e-3))


def expected_spill(load_bits, mean_served, std_served):
    """E[(load - X)^+] for X ~ N(mean_served, std_served^2): the backlog a path is
    expected to carry into the next block (team's queue-aware extension).

    With Y = load - X ~ N(mu, sd^2):  E[Y^+] = mu Phi(mu/sd) + sd phi(mu/sd).
    """
    mu = load_bits - mean_served
    sd = max(std_served, 1e-9)
    z = mu / sd
    return float(mu * ndtr(z) + sd * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi))


# =========================================================================== per-path distributions
def psi_exact(lam):
    """P(d = 0..s) for one path, exact under nested on-time events."""
    s = len(lam)
    out = np.empty(s + 1)
    if s == 0:
        out[0] = 1.0
        return out
    out[0] = lam[0]
    out[1:s] = np.diff(lam)
    out[s] = 1.0 - lam[-1]
    return np.clip(out, 0.0, 1.0)


def psi_product(lam):
    """The paper's eq. (5): psi(d) = (1-lambda(d)) lambda(d+1), psi(s) = 1-lambda(s)."""
    s = len(lam)
    out = np.empty(s + 1)
    if s == 0:
        out[0] = 1.0
        return out
    lam0 = np.concatenate([[0.0], lam])                # lambda(0) = 0
    out[:s] = (1.0 - lam0[:s]) * lam0[1:s + 1]
    out[s] = 1.0 - lam[-1]
    return out


# =========================================================================== P_e
def pe_conv(lams, k, product=False):
    """P_e via convolution of per-path distributions: O(m k^2) instead of O(k^m)."""
    if product:
        # the paper's formula: P_e = 1 - sum_{d in D_k} prod psi(d_i) = 1 - tail of
        # the convolution (psi does not sum to 1, so the tail must be summed directly)
        full = np.array([1.0])
        for lam in lams:
            full = np.convolve(full, psi_product(lam))
        return float(np.clip(1.0 - full[k:].sum(), 0.0, 1.0))
    acc = np.array([1.0])
    for lam in lams:
        acc = np.convolve(acc, psi_exact(lam))[:k]   # keep only totals < k
    return float(np.clip(acc.sum(), 0.0, 1.0))


def pe_enumerate(lams, k, product=True):
    """The paper's literal enumeration over D_k(s) (eqs. 3-4). Exponential in m."""
    psi = psi_product if product else psi_exact
    tables = [psi(lam) for lam in lams]
    sizes = [len(lam) for lam in lams]
    total = 0.0
    for d in itertools.product(*(range(s + 1) for s in sizes)):
        if sum(d) >= k:
            p = 1.0
            for t, di in zip(tables, d):
                p *= t[di]
            total += p
    return float(np.clip(1.0 - total, 0.0, 1.0))


def pe_monte_carlo(rng, n_samples, schedule, backlogs, windows, means, stds, k,
                   p_outs=None, rho=0.0):
    """Ground truth under the model: sample capacities, count on-time packets."""
    m = len(schedule)
    z = rng.normal(size=n_samples)
    ok = np.zeros(n_samples)
    for i in range(m):
        e = rng.normal(size=n_samples)
        c = means[i] + stds[i] * (np.sqrt(rho) * z + np.sqrt(1 - rho) * e)
        if p_outs is not None and p_outs[i] > 0:
            c = np.where(rng.random(n_samples) < p_outs[i], 0.0, c)
        c = np.maximum(c, 0.0)
        d = np.floor((c * windows[i] - backlogs[i]) / PKT_BITS)
        ok += np.clip(d, 0, schedule[i])
    return float(np.mean(ok < k))


def pe_factor(path_params, schedule, k, rho, nodes=24):
    """Correlation-aware P_e: one-factor Gaussian model integrated by Gauss-Hermite.

    C_i = m_i + s_i (sqrt(rho) Z + sqrt(1-rho) e_i). Conditional on Z the paths
    are independent, so P_e = E_Z[ P_e(s | Z) ], each term via convolution.
    """
    if rho <= 1e-6:
        lams = [lateness(n, *pp) for n, pp in zip(schedule, path_params)]
        return pe_conv(lams, k)
    x, w = np.polynomial.hermite_e.hermegauss(nodes)       # probabilists' Hermite
    w = w / w.sum()
    total = 0.0
    for zi, wi in zip(x, w):
        lams = []
        for n, (backlog, window, mean, std, p_out) in zip(schedule, path_params):
            lams.append(lateness(n, backlog, window, mean + std * math.sqrt(rho) * zi,
                                 std * math.sqrt(1 - rho), p_out))
        total += wi * pe_conv(lams, k)
    return float(np.clip(total, 0.0, 1.0))


# =========================================================================== JUMP
class Decision:
    def __init__(self, schedule, pe, feasible, dropped=False):
        self.schedule = schedule
        self.pe = pe
        self.feasible = feasible
        self.dropped = dropped


def jump(path_params, k, p_thr, caps, product=True, min_per_path=10, rho=0.0,
         infeasible="max", drop_below=0.5):
    """JUMP (Sec. IV): greedy minimum-size schedule with P_e <= 1 - p_thr.

    path_params: per path (backlog_bits, window, mean, std, p_out).
    caps: max packets per path (S_i).
    The greedy picks, at each step, the path whose next packet has the lowest
    lateness probability (eq. 2). Since every lambda_i is non-decreasing, the
    whole greedy order is the sorted merge of all lambda_i(j) - so instead of
    re-evaluating P_e after every packet we binary-search the prefix length
    (P_e is non-increasing in the prefix). Same result, ~log N evaluations.

    infeasible = "max"  : paper behaviour - send the largest allowed schedule.
                 "guard": team's anti-collapse rule - if even the largest
                          schedule succeeds with prob < drop_below, drop the
                          block (send nothing) so queues can drain.
    """
    m = len(path_params)
    caps = [int(max(0, min(c, k))) for c in caps]
    lam_all = [lateness(caps[i], *path_params[i]) for i in range(m)]
    # greedy order = merge of per-path lambda sequences
    order = []
    for i in range(m):
        order += [(lam_all[i][j], i) for j in range(caps[i])]
    order.sort(key=lambda t: (t[0], t[1]))
    path_seq = np.array([i for _, i in order], dtype=int)
    target = 1.0 - p_thr

    def sched_of(n):
        return np.bincount(path_seq[:n], minlength=m) if n else np.zeros(m, dtype=int)

    def pe_of(s):
        if rho > 1e-6:
            return pe_factor(path_params, s, k, rho)
        lams = [lam_all[i][:s[i]] for i in range(m)]
        return pe_conv(lams, k, product=product)

    N = len(path_seq)
    if N < k:
        s = sched_of(N)
        pe = 1.0
        return _infeasible(s, pe, infeasible, drop_below, k)
    lo, hi = k, N
    pe_hi = pe_of(sched_of(hi))
    if pe_hi > target:
        return _infeasible(sched_of(hi), pe_hi, infeasible, drop_below, k)
    while lo < hi:
        mid = (lo + hi) // 2
        if pe_of(sched_of(mid)) <= target:
            hi = mid
        else:
            lo = mid + 1
    s = sched_of(lo)
    # practical rule (Sec. IV): at least min_per_path packets per path for probing
    s = np.array([max(s[i], min(min_per_path, caps[i])) for i in range(m)])
    return Decision(s, pe_of(s), True)


def _infeasible(s, pe, mode, drop_below, k):
    if mode == "guard" and (1.0 - pe) < drop_below:
        return Decision(np.zeros_like(s), 1.0, False, dropped=True)
    return Decision(s, pe, False)


# =========================================================================== baselines
def earliest_completion(path_params, n_packets):
    """Assign packets one by one to the path that would deliver them first
    (ECF/STTF-style), using each path's estimated mean capacity, backlog and
    usable window W_i = T - tau_i/2 (a longer window = an earlier arrival)."""
    m = len(path_params)
    s = np.zeros(m, dtype=int)
    rate = np.array([max(pp[2], 1e3) for pp in path_params])
    finish = np.array([pp[0] for pp in path_params], dtype=float) / rate
    window = np.array([pp[1] for pp in path_params])
    for _ in range(n_packets):
        t_next = finish + PKT_BITS / rate
        i = int(np.argmin(t_next - window))
        s[i] += 1
        finish[i] = t_next[i]
    return s
