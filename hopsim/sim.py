"""
Block-level simulation of one run (paper Sec. VI setup).

Every Tb seconds the application hands the sender a block of B bytes
(k = ceil(B / 1500) packets) with deadline T. The scheduler decides how many
coded packets go on each path; packets are sent as a burst, queue at the
path bottleneck, and arrive tau/2 later. The block is on time iff at least
k packets (any k: systematic MDS code) arrive within T.

Default parameters match the paper: T = 35 ms, P_thr = 0.98, min RTT 25 ms,
MTU 1500 B, capacity changing every 25 ms, block period 25 ms (so a 145 kB
block = 46.4 Mb/s, the paper's reported operating point).
"""

import math
from dataclasses import dataclass, field

import numpy as np

from . import scheduler as S
from .network import Path
from .skip import SkipEstimator

PKT_BITS = S.PKT_BITS


@dataclass
class Scheme:
    name: str
    kind: str = "jump"            # jump | split | static_fec
    estimator: str = "skip"       # skip | ideal
    product: bool = False         # True = paper's printed eq. (5)
    guard: bool = False           # team: backlog-aware caps + drop rule
    guard_z: float = 2.0          # cap at capacity quantile mean + z*std (minus backlog)
    relax: tuple = ()             # team: fall-back reliability targets when infeasible
    drop_below: float = 0.0       # team: drop block if best success prob < this
    exp_backlog: bool = False     # team: expected (not mean-based) backlog prediction
    t_thr: int = 3                # SKIP outage/step threshold (paper: 3)
    outage_wait: float = 2.0      # block periods of ACK silence (beyond tau) before a zero sample
    rho: str = "none"             # none | est (team: correlation-aware)
    fec_ratio: float = 0.2        # for static_fec
    p_thr: float = 0.98


@dataclass
class RunResult:
    late: int = 0
    blocks: int = 0
    dropped: int = 0
    sent_pkts: int = 0
    data_pkts: int = 0
    goodput_bits: float = 0.0
    duration: float = 0.0
    backlog_trace: list = field(default_factory=list)
    fail_trace: list = field(default_factory=list)
    rho_est: list = field(default_factory=list)
    pred_pe_sum: float = 0.0          # sum of the scheduler's own P_e predictions
    pred_n: int = 0
    infeasible: int = 0
    est_trace: list = field(default_factory=list)

    @property
    def predicted(self):
        return self.pred_pe_sum / max(self.pred_n, 1)

    @property
    def failure(self):
        return self.late / max(self.blocks, 1)

    @property
    def overhead(self):
        return self.sent_pkts / max(self.data_pkts, 1) - 1.0

    @property
    def goodput_mbps(self):
        return self.goodput_bits / max(self.duration, 1e-9) / 1e6


def run(scheme, traces, taus, true_means, true_stds, block_bytes, duration=20.0,
        T=0.035, Tb=0.025, slot=0.025, warmup=1.0, true_rho=0.0, record=False):
    m = len(traces)
    paths = [Path(traces[i], slot, taus[i]) for i in range(m)]
    est = [SkipEstimator(t_thr=scheme.t_thr) for _ in range(m)]
    pending = [[] for _ in range(m)]         # (send_t, acks, zero_fed)
    pred_finish = np.zeros(m)
    exp_bl = np.zeros(m)                      # team: expected backlog (bits) at next block
    windows = [T - taus[i] / 2 for i in range(m)]
    k = math.ceil(block_bytes * 8 / PKT_BITS)
    res = RunResult()
    n_blocks = int(duration / Tb)
    pair_hist = []                           # paired samples for correlation estimate
    for b in range(n_blocks):
        t = b * Tb
        # ------------------------------------------------ 1. feedback -> SKIP
        sample_now = [None] * m
        for i in range(m):
            keep = []
            for (st, acks, fed) in pending[i]:
                if acks[-1] <= t:
                    if len(acks) >= 2 and acks[-1] > acks[0]:
                        c = PKT_BITS * (len(acks) - 1) / (acks[-1] - acks[0])
                        est[i].update(c)
                        sample_now[i] = c
                else:
                    keep.append((st, acks, fed))
            pending[i] = keep
            if pending[i]:
                st, acks, fed = pending[i][0]
                # nothing back long after it should have been: outage evidence
                if acks[0] > t and t - st > taus[i] + scheme.outage_wait * Tb and fed < b:
                    est[i].update(0.0)
                    pending[i][0] = (st, acks, b)
        if all(v is not None for v in sample_now):
            pair_hist.append(sample_now)
            if len(pair_hist) > 200:
                pair_hist.pop(0)
        # ------------------------------------------------ 2. path state for the scheduler
        params = []
        ready = all(e.ready for e in est)
        for i in range(m):
            if scheme.estimator == "ideal":
                params.append((paths[i].backlog_bits(t), windows[i], true_means[i],
                               true_stds[i], 0.0))
            elif ready:
                mu, sd, pout = est[i].predictive()
                if scheme.exp_backlog:
                    backlog = exp_bl[i]
                else:
                    backlog = max(0.0, pred_finish[i] - t) * mu
                params.append((backlog, windows[i], mu, sd, pout))
        # ------------------------------------------------ 3. schedule
        dropped = False
        dec = None
        if not params or (scheme.estimator == "skip" and not ready):
            # warm-up (the paper uses a slow-start phase): plain split + 10% FEC
            share = math.ceil(1.1 * k / m)
            sched = np.full(m, share)
        elif scheme.kind == "split":
            sched = S.earliest_completion(params, k)
        elif scheme.kind == "static_fec":
            sched = S.earliest_completion(params, math.ceil((1 + scheme.fec_ratio) * k))
        else:
            caps = []
            for i, (bl, w, mu, sd, po) in enumerate(params):
                cap = math.floor(mu * w / PKT_BITS)                    # S_i = mu_i (T - tau_i/2)
                if scheme.guard:
                    # team: packets that would need more than mean + 2 sigma
                    # capacity are (almost) never on time - don't send them
                    cap = min(cap, math.floor(max(0.0, (mu + scheme.guard_z * sd) * w - bl)
                                              / PKT_BITS))
                caps.append(cap)
            rho = 0.0
            if scheme.rho == "est" and len(pair_hist) >= 30:
                arr = np.array(pair_hist)
                cc = np.corrcoef(arr.T)
                rho = float(np.clip(np.mean(cc[np.triu_indices(m, 1)]), 0.0, 0.95))
            elif scheme.rho == "true":
                rho = true_rho
            if record:
                res.rho_est.append(rho)
            dec = S.jump(params, k, scheme.p_thr, caps, product=scheme.product, rho=rho,
                         infeasible="guard" if scheme.drop_below > 0 else "max",
                         drop_below=scheme.drop_below)
            if not dec.feasible and not dec.dropped and scheme.relax:
                # team: graceful degradation - meet the best still-reachable target
                # with the minimum schedule instead of flooding the paths with FEC
                for p_alt in scheme.relax:
                    d2 = S.jump(params, k, p_alt, caps, product=scheme.product, rho=rho)
                    if d2.feasible:
                        dec = d2
                        break
            sched = dec.schedule
            dropped = dec.dropped
        # queue left over from earlier blocks, measured before this block is sent
        carry_kB = sum(p.backlog_bits(t) for p in paths) / 8 / 1000
        # ------------------------------------------------ 4. transmit
        on_time = 0
        for i in range(m):
            n = int(sched[i])
            if n <= 0:
                continue
            dep = paths[i].send_burst(t, n, PKT_BITS)
            arr = dep + taus[i] / 2
            on_time += int(np.sum(arr <= t + T))
            pending[i].append((t, dep + taus[i], -1))
            mu_e = est[i].x if est[i].ready else true_means[i]
            pred_finish[i] = max(pred_finish[i], t) + n * PKT_BITS / max(mu_e, 1e3)
        if scheme.exp_backlog:
            for i in range(m):
                if est[i].ready:
                    mu, sd, _ = est[i].predictive()
                    exp_bl[i] = S.expected_spill(exp_bl[i] + int(sched[i]) * PKT_BITS,
                                                 mu * Tb, sd * Tb)
        ok = on_time >= k and not dropped
        # ------------------------------------------------ 5. metrics (after warm-up)
        if t >= warmup:
            if dec is not None:
                res.pred_pe_sum += dec.pe
                res.pred_n += 1
                res.infeasible += int(not dec.feasible)
            if record:
                res.est_trace.append([t] + [paths[i].C[min(int(round(t / slot)), len(paths[i].C) - 1)] / 1e6
                                            for i in range(m)]
                                     + [(est[i].x / 1e6 if est[i].ready else np.nan) for i in range(m)])
            res.blocks += 1
            res.late += 0 if ok else 1
            res.dropped += int(dropped)
            res.sent_pkts += int(np.sum(sched))
            res.data_pkts += k
            if ok:
                res.goodput_bits += block_bytes * 8
            if record:
                res.backlog_trace.append(carry_kB)
                res.fail_trace.append(0 if ok else 1)
    res.duration = max(duration - warmup, Tb)
    return res
