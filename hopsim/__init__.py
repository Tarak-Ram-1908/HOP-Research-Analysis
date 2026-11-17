"""hopsim - a block-level multipath simulator for analysing the HOP protocol.

Reproduces the core of Chiariotti et al., "The HOP Protocol: Reliable
Latency-Bounded End-to-End Multipath Communication", IEEE/ACM ToN 29(5), 2021:
the JUMP scheduler and the SKIP capacity estimator, plus baselines and the
team's proposed improvements (exact lateness model + polynomial-time P_e,
backlog-aware anti-collapse guard, correlation-aware scheduling).
"""
