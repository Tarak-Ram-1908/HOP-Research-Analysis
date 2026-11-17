"""Checks of the scheduling maths against brute force and Monte Carlo."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hopsim import scheduler as S          # noqa: E402
from hopsim.network import Path            # noqa: E402

W = 0.0225


def params(mu, sd, backlog=0.0, pout=0.0):
    return (backlog, W, mu * 1e6, sd * 1e6, pout)


def test_exact_psi_is_a_distribution():
    lam = S.lateness(80, *params(40, 8))
    assert abs(S.psi_exact(lam).sum() - 1.0) < 1e-12
    assert (S.psi_exact(lam) >= 0).all()


def test_printed_psi_overcounts():
    lam = S.lateness(80, *params(40, 8))
    assert S.psi_product(lam).sum() > 1.5        # not a distribution (finding F1)


def test_convolution_equals_enumeration_paper_and_exact():
    rng = np.random.default_rng(0)
    for m in (2, 3):
        pp = [params(rng.uniform(5, 15), rng.uniform(1, 4)) for _ in range(m)]
        lams = [S.lateness(12, *p) for p in pp]
        for k in (8, 15, 22):
            for product in (True, False):
                a = S.pe_enumerate(lams, k, product=product)
                b = S.pe_conv(lams, k, product=product)
                assert abs(a - b) < 1e-9


def test_exact_matches_monte_carlo():
    rng = np.random.default_rng(1)
    pp = [params(40, 8), params(32, 8)]
    s = (55, 50)
    lams = [S.lateness(n, *p) for n, p in zip(s, pp)]
    mc = S.pe_monte_carlo(rng, 300000, s, [0, 0], [W, W], [40e6, 32e6], [8e6, 8e6], 87)
    assert abs(S.pe_conv(lams, 87) - mc) < 0.002


def test_factor_model_matches_monte_carlo_with_correlation():
    rng = np.random.default_rng(2)
    pp = [params(40, 8), params(40, 8)]
    for rho in (0.4, 0.8):
        s = (60, 60)
        mc = S.pe_monte_carlo(rng, 300000, s, [0, 0], [W, W], [40e6] * 2, [8e6] * 2, 87,
                              rho=rho)
        assert abs(S.pe_factor(pp, s, 87, rho) - mc) < 0.002


def test_jump_meets_target_and_is_minimal():
    pp = [params(40, 8), params(40, 8)]
    d = S.jump(pp, 87, 0.98, [75, 75], product=False, min_per_path=0)
    assert d.feasible and d.pe <= 0.02
    # removing any packet breaks the target (minimality along the greedy order)
    s = d.schedule.copy()
    i = int(np.argmax(s))
    s[i] -= 1
    lams = [S.lateness(int(n), *p) for n, p in zip(s, pp)]
    assert S.pe_conv(lams, 87) > 0.02


def test_expected_spill_closed_form():
    rng = np.random.default_rng(3)
    x = rng.normal(1e6, 2e5, 400000)
    mc = np.mean(np.maximum(1.1e6 - x, 0))
    assert abs(S.expected_spill(1.1e6, 1e6, 2e5) - mc) / mc < 0.01


def test_fifo_queue_departures():
    p = Path([12e6] * 10, 0.025, 0.025)              # 12 Mb/s -> 1 ms per 1500 B packet
    dep = p.send_burst(0.0, 5, S.PKT_BITS)
    assert np.allclose(dep, [0.001, 0.002, 0.003, 0.004, 0.005])
    dep2 = p.send_burst(0.0025, 2, S.PKT_BITS)       # queues behind the first burst
    assert np.allclose(dep2, [0.006, 0.007])
