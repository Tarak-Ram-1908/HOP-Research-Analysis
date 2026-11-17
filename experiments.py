"""
Run every experiment of the report and store results in results/*.json.

    python experiments.py            # everything (~25 min on 2 cores)
    python experiments.py e1 e3      # selected experiments

E1  reproduction of the paper's synthetic scenarios (Figs. 5-8)
E2  eq. (5) as printed vs the exact per-path distribution
E3  complexity: enumeration (paper) vs convolution (ours), m = 2..6 paths
E4  anti-collapse: ablation of graceful-degradation variants + backlog trace
E5  correlated paths with SKIP: promised vs delivered reliability
E5a model-level calibration of the promise under correlation (exact, Monte Carlo)
E5b correlated paths with ideal marginals: isolates the independence assumption
E6  mmWave-blockage + LTE scenario (SKIP step/outage modes)
"""

import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from hopsim import scheduler as S
from hopsim.network import blockage_trace, correlated_traces, gaussian_trace
from hopsim.sim import Scheme, run

OUT = "results"
SEEDS = 6
DUR = 20.0
RELAX = (0.95, 0.9, 0.8, 0.6)

SCENARIOS = {   # name: (means Mb/s, stds Mb/s, taus s)  - paper Sec. VI
    "balanced": ([40, 40], [8, 8], [0.025, 0.025]),
    "unbalanced": ([32, 48], [8, 8], [0.025, 0.025]),
    "highly_unbalanced": ([64, 16], [8, 8], [0.025, 0.025]),
    "rtt_unbalanced": ([40, 40], [8, 8], [0.020, 0.030]),
}

SCHEMES = {
    "split": dict(kind="split"),
    "static20": dict(kind="static_fec", fec_ratio=0.2),
    "hop": dict(),
    "ideal": dict(estimator="ideal"),
    "hop_relax": dict(relax=RELAX),
    "hop_printed": dict(product=True),
    "hop_fast": dict(t_thr=1, outage_wait=1.0),               # SKIP sensitivity (E6)
    "hop_fast_relax": dict(t_thr=1, outage_wait=1.0, relax=RELAX),
}


def traces_for(scen, seed, n=1000):
    means, stds, _ = SCENARIOS[scen]
    rng = np.random.default_rng(seed)
    return [gaussian_trace(rng, n, mu, sd) for mu, sd in zip(means, stds)]


def _job_e1(a):
    scen, sname, B, seed = a
    means, stds, taus = SCENARIOS[scen]
    r = run(Scheme(sname, **SCHEMES[sname]), traces_for(scen, seed), taus,
            [m * 1e6 for m in means], [s * 1e6 for s in stds], B, duration=DUR)
    return dict(scen=scen, scheme=sname, B=B, seed=seed, fail=r.failure,
                goodput=r.goodput_mbps, overhead=r.overhead, pred=r.predicted)


def bound(scen, B, n=400000, seed=0):
    """Theoretical bound (paper Sec. VI): perfect scheduling, no queuing, full
    knowledge of the capacity distribution: fail iff total deliverable < k."""
    means, stds, taus = SCENARIOS[scen]
    rng = np.random.default_rng(seed)
    k = math.ceil(B * 8 / S.PKT_BITS)
    tot = 0
    for mu, sd, tau in zip(means, stds, taus):
        c = np.maximum(rng.normal(mu, sd, n), 0) * 1e6
        tot = tot + np.floor(c * (0.035 - tau / 2) / S.PKT_BITS)
    return float(np.mean(tot < k))


def e1(pool):
    sizes = list(range(80_000, 201_000, 10_000))
    jobs = []
    for scen in SCENARIOS:
        names = ["split", "static20", "hop", "ideal", "hop_relax"]
        if scen == "balanced":
            names.append("hop_printed")
        jobs += [(scen, s, B, seed) for s in names for B in sizes for seed in range(SEEDS)]
    rows = pool.map(_job_e1, jobs)
    bounds = {scen: {B: bound(scen, B) for B in sizes} for scen in SCENARIOS}
    return dict(rows=rows, bounds=bounds, sizes=sizes)


def e2(_pool):
    """eq. (5) as printed vs exact, against Monte Carlo ground truth."""
    rng = np.random.default_rng(7)
    W = 0.0225
    out = []
    for (m1, s1, m2, s2) in [(40, 8, 40, 8), (32, 8, 48, 8), (64, 8, 16, 8)]:
        pp = [(0.0, W, m1 * 1e6, s1 * 1e6, 0.0), (0.0, W, m2 * 1e6, s2 * 1e6, 0.0)]
        for k in (67, 87, 97):
            for extra in (0.0, 0.1, 0.2, 0.3):
                n = math.ceil(k * (1 + extra))
                s = S.earliest_completion(pp, n)
                lams = [S.lateness(int(c), *p) for c, p in zip(s, pp)]
                out.append(dict(paths=f"{m1}/{m2}", k=k, sched=[int(x) for x in s],
                                printed=S.pe_conv(lams, k, product=True),
                                exact=S.pe_conv(lams, k),
                                mc=S.pe_monte_carlo(rng, 300000, s, [0, 0], [W, W],
                                                    [m1 * 1e6, m2 * 1e6], [s1 * 1e6, s2 * 1e6], k),
                                psi_sum=[float(S.psi_product(l).sum()) for l in lams]))
    return dict(rows=out)


def e3(_pool):
    """Time to evaluate P_e once: paper enumeration vs our convolution."""
    rows = []
    rng = np.random.default_rng(3)
    for m in (2, 3, 4, 5, 6):
        for k in (20, 40, 80, 120):
            per = math.ceil(1.2 * k / m) + 1
            # Capacities sized so that ~20% redundancy sits near JUMP's operating point
            # (mean ~1.3 k/m packets per window, 20% std): P_e is then between ~0.3% and
            # ~15%, so the conv == enum comparison is not a trivial 0 == 0.
            pp = []
            for _ in range(m):
                mu = rng.uniform(0.9, 1.1) * 1.3 * (k / m) * S.PKT_BITS / 0.0225
                pp.append((0.0, 0.0225, mu, 0.2 * mu, 0.0))
            lams = [S.lateness(per, *p) for p in pp]
            # best of 5 batches (min is the standard noise-robust timing estimate)
            t_conv = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                reps = 0
                while time.perf_counter() - t0 < 0.05:
                    v_conv = S.pe_conv(lams, k, product=False)   # exact psi (I1)
                    reps += 1
                t_conv = min(t_conv, (time.perf_counter() - t0) / reps)
            n_vec = (per + 1) ** m
            t_enum, v_enum = None, None
            if n_vec <= 3_000_000:
                t_enum = float("inf")
                for _ in range(3):                                # best of 3 runs
                    t0 = time.perf_counter()
                    v_enum = S.pe_enumerate(lams, k, product=False)
                    t_enum = min(t_enum, time.perf_counter() - t0)
            rows.append(dict(m=m, k=k, per=per, vectors=n_vec, t_conv=t_conv, t_enum=t_enum,
                             v_conv=v_conv, v_enum=v_enum))
            print(f"  e3 m={m} k={k} conv={t_conv*1e6:.0f}us enum="
                  f"{'-' if t_enum is None else f'{t_enum:.3f}s'}", flush=True)
    return dict(rows=rows)


E4_VARIANTS = {
    "HOP": dict(),
    "+relax": dict(relax=RELAX),
    "+drop<0.3": dict(drop_below=0.3),
    "+expected backlog": dict(exp_backlog=True),
    "+exp. backlog & cap": dict(exp_backlog=True, guard=True, guard_z=0.0),
    "+cap & relax (full)": dict(exp_backlog=True, guard=True, guard_z=0.0, relax=RELAX),
    "Ideal JUMP": dict(estimator="ideal"),
    "Ideal + cap": dict(estimator="ideal", guard=True, guard_z=0.0),
    "Ideal + cap & relax": dict(estimator="ideal", guard=True, guard_z=0.0, relax=RELAX),
}


def _job_e4(a):
    name, B, seed = a
    r = run(Scheme(name, **E4_VARIANTS[name]), traces_for("balanced", 500 + seed),
            [0.025] * 2, [40e6] * 2, [8e6] * 2, B, duration=DUR)
    return dict(variant=name, B=B, seed=seed, fail=r.failure, goodput=r.goodput_mbps,
                overhead=r.overhead, dropped=r.dropped / max(r.blocks, 1))


def e4(pool):
    sizes = [140_000, 155_000, 170_000, 185_000]
    rows = pool.map(_job_e4, [(v, B, s) for v in E4_VARIANTS for B in sizes
                              for s in range(8)])
    traces = {}
    for name, kw in (("HOP", dict()), ("HOP + relax", dict(relax=RELAX))):
        r = run(Scheme(name, **kw), traces_for("balanced", 999), [0.025] * 2, [40e6] * 2,
                [8e6] * 2, 170_000, duration=8.0, record=True)
        traces[name] = dict(backlog=r.backlog_trace, fail=r.fail_trace)  # carried-over queue
    return dict(rows=rows, sizes=sizes, traces=traces)


def _job_e5(a):
    name, rho, p_thr, seed = a
    kw = dict(hop=dict(), hop_corr=dict(rho="est"))[name]
    rng = np.random.default_rng(700 + seed)
    tr = correlated_traces(rng, 1000, [40, 40], [8, 8], rho)
    r = run(Scheme(name, p_thr=p_thr, **kw), tr, [0.025] * 2, [40e6] * 2, [8e6] * 2,
            100_000, duration=DUR, true_rho=rho, record=True)
    return dict(scheme=name, rho=rho, p_thr=p_thr, seed=seed, fail=r.failure,
                pred=r.predicted, infeasible=r.infeasible / max(r.pred_n, 1),
                overhead=r.overhead, rho_est=float(np.mean(r.rho_est[-300:])) if r.rho_est else 0.0)


def e5(pool):
    jobs = [(n, rho, p, s) for n in ("hop", "hop_corr") for rho in (0.0, 0.2, 0.4, 0.6, 0.8)
            for p in (0.98, 0.995) for s in range(4)]
    return dict(rows=pool.map(_job_e5, jobs))


def _job_e5b(a):
    name, rho, seed = a
    kw = dict(indep=dict(estimator="ideal", relax=RELAX),
              corr=dict(estimator="ideal", relax=RELAX, rho="true"))[name]
    rng = np.random.default_rng(800 + seed)
    tr = correlated_traces(rng, 2000, [40, 40], [8, 8], rho)
    r = run(Scheme(name, p_thr=0.98, **kw), tr, [0.025] * 2, [40e6] * 2, [8e6] * 2,
            130_000, duration=30.0, true_rho=rho)
    return dict(scheme=name, rho=rho, seed=seed, fail=r.failure, pred=r.predicted,
                infeasible=r.infeasible / max(r.pred_n, 1), overhead=r.overhead,
                blocks=r.blocks, late=r.late)


def e5a(_pool):
    """Model-level calibration: the schedule JUMP picks under each joint model, and the
    TRUE failure probability of that schedule (Monte Carlo) when capacities correlate."""
    rng = np.random.default_rng(11)
    W = 0.035 - 0.0125
    pp = [(0.0, W, 40e6, 8e6, 0.0), (0.0, W, 40e6, 8e6, 0.0)]
    caps = [int(40e6 * W / S.PKT_BITS)] * 2
    rows = []
    for k in (80, 87, 94):                      # 120, 130, 140 kB blocks
        for rho in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9):
            for model in ("indep", "corr"):
                d = S.jump(pp, k, 0.98, caps, product=False, min_per_path=0,
                           rho=rho if model == "corr" else 0.0)
                true = S.pe_monte_carlo(rng, 400000, d.schedule, [0, 0], [W, W],
                                        [40e6] * 2, [8e6] * 2, k, rho=rho)
                rows.append(dict(k=k, rho=rho, model=model, sched=[int(x) for x in d.schedule],
                                 promised=d.pe, true=true, feasible=bool(d.feasible)))
    return dict(rows=rows)


def e5b(pool):
    """Isolate the independence assumption: ideal knowledge of each path's marginal
    distribution and queue; only the joint model differs. 130 kB blocks (JUMP needs
    redundancy there), target 98%, with relaxation so infeasibility is reported."""
    jobs = [(n, rho, s) for n in ("indep", "corr") for rho in (0.0, 0.45, 0.9)
            for s in range(4)]
    return dict(rows=pool.map(_job_e5b, jobs))


def blockage_traces(seed, n=1000):
    rng = np.random.default_rng(seed)
    a = blockage_trace(rng, n, 70, 12, p_block=0.012, p_unblock=0.2, blocked_mbps=1.0)
    b = blockage_trace(rng, n, 35, 10, p_block=0.0, ar=0.85)        # LTE-like, no blockage
    return [a, b]


def _job_e6(a):
    sname, B, seed = a
    r = run(Scheme(sname, **SCHEMES[sname]), blockage_traces(900 + seed), [0.025, 0.035],
            [70e6, 35e6], [12e6, 10e6], B, duration=DUR)
    return dict(scheme=sname, B=B, seed=seed, fail=r.failure, goodput=r.goodput_mbps,
                overhead=r.overhead)


def e6(pool):
    sizes = [30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 100_000, 120_000, 140_000,
             160_000]
    rows = pool.map(_job_e6, [(s, B, seed) for s in ("split", "static20", "hop", "hop_relax",
                                                      "hop_fast", "hop_fast_relax")
                              for B in sizes for seed in range(SEEDS)])
    r = run(Scheme("hop"), blockage_traces(901), [0.025, 0.035], [70e6, 35e6], [12e6, 10e6],
            50_000, duration=12.0, record=True)
    r2 = run(Scheme("hop_fast", **SCHEMES["hop_fast"]), blockage_traces(901), [0.025, 0.035],
             [70e6, 35e6], [12e6, 10e6], 50_000, duration=12.0, record=True)
    return dict(rows=rows, sizes=sizes, est_trace=r.est_trace, est_trace_fast=r2.est_trace)


EXPERIMENTS = dict(e1=e1, e2=e2, e3=e3, e4=e4, e5=e5, e5a=e5a, e5b=e5b, e6=e6)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    chosen = sys.argv[1:] or list(EXPERIMENTS)
    with Pool(os.cpu_count() or 2) as pool:
        for name in chosen:
            t0 = time.time()
            print(f"running {name} ...", flush=True)
            res = EXPERIMENTS[name](pool)
            with open(f"{OUT}/{name}.json", "w") as f:
                json.dump(res, f, default=float)
            print(f"  {name} done in {time.time() - t0:.0f} s", flush=True)
