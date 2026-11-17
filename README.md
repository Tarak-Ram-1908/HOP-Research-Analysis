# HOP protocol: reproduction and extensions (source code)

Implementation for the CS3001 research-paper analysis of

> F. Chiariotti, A. Zanella, S. Kucera, K. Fahmi, H. Claussen, *The HOP Protocol: Reliable
> Latency-Bounded End-to-End Multipath Communication*, IEEE/ACM ToN 29(5), 2021.

`hopsim` is a block-level multipath simulator. It reproduces HOP's **JUMP** scheduler and
**SKIP** capacity estimator on the paper's synthetic scenarios, compares them with
baselines, and evaluates the team's three improvements:

| ID | Improvement | Code |
|----|-------------|------|
| I1 | Exact per-path delivery distribution, `λ(d+1) − λ(d)`, and P_e by **convolution**: O(m k²) instead of the paper's O(k^m) enumeration | `scheduler.psi_exact`, `scheduler.pe_conv` |
| I2 | **Graceful degradation** against reliability collapse: fall back to the best reachable target with a minimum schedule. Also tested: expected-backlog caps and block dropping | `sim.Scheme(relax=...)`, `scheduler.expected_spill` |
| I3 | **Correlation-aware** P_e: a one-factor Gaussian model integrated by Gauss–Hermite quadrature, with ρ estimated online | `scheduler.pe_factor`, `sim.Scheme(rho="est")` |

## Requirements

Python 3.10+ with `numpy`, `scipy` and `matplotlib` (`pip install numpy scipy matplotlib`).
No other dependencies are needed.

## Run

```bash
python experiments.py            # all experiments, ~35 min on 2 cores -> results/*.json
python summarize.py              # headline numbers -> results/summary.json
python experiments.py e2 e3      # only selected experiments
python plots.py                  # figures -> figures/*.png and *.pdf
python -m pytest tests -q        # unit tests of the maths
```

| Experiment | What it shows | Figure |
|------------|---------------|--------|
| e1 | Reproduction of the paper's four synthetic scenarios (Figs. 5–8) | `e1_reproduction`, `e1_goodput`, `e2_printed_sim` |
| e2 | Eq. (5) as printed vs exact vs Monte Carlo | `e2_eq5` |
| e3 | Cost of one P_e evaluation: enumeration vs convolution, 2–6 paths | `e3_complexity` |
| e4 | Reliability collapse and the graceful-degradation ablation | `e4_collapse` |
| e5 | Correlated paths with SKIP estimation (limitation: the effect is masked by estimation error) | numbers in report |
| e5a | Calibration: promised vs true P_e of the schedule each joint model picks (exact, Monte Carlo) | `e5_correlation` |
| e5b | Correlated paths in the queueing simulation with exact per-path knowledge (130 kB) | numbers in report |
| e6 | mmWave blockage + LTE; SKIP outage-detection sensitivity (t_thr = 3 vs 1) | `e6_blockage` |

## Model (what is and is not simulated)

* Each path is a FIFO bottleneck whose capacity is constant over 25 ms slots (Gaussian i.i.d.
  as in the paper, correlated, or AR(1) with Markov blockages), followed by a fixed
  propagation delay. Queues are simulated exactly through the cumulative service curve.
* A block of B bytes is sent every 25 ms with deadline T = 35 ms and target P_thr = 98%.
  It is on time if at least k = ⌈B/1500⌉ coded packets arrive within T (an MDS code is
  assumed).
* SKIP samples capacity from the ACK spacing of each burst (the packet-train equivalent of
  the paper's RTT-gradient estimator), feeds an adaptive Kalman filter, and implements the
  outage and step modes with t_thr = 3 and a 50 kB/s reset.
* Baselines are **scheduler-level abstractions**, not full MPTCP stacks:
  *Split* (earliest-completion assignment, no FEC) and *Static FEC +20%*. They have no
  congestion control and no retransmission, which makes them stronger than the paper's
  MPTCP baselines. Absolute numbers are not comparable with the paper's ns-3 results;
  trends and relative effects are.

## Layout

```
hopsim/network.py    paths, exact FIFO queue, capacity trace generators
hopsim/skip.py       SKIP estimator (adaptive Kalman filter + outage/step modes)
hopsim/scheduler.py  lateness probability, P_e (enumeration / convolution / factor model),
                     JUMP, baselines
hopsim/sim.py        one simulation run; Scheme = the scheduler configuration
experiments.py       all experiments -> results/
plots.py             all figures -> figures/
tests/               checks of the maths against brute force and Monte Carlo
```
