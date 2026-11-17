"""
SKIP - Sender-side Kalman Inference Procedure (paper Sec. V), as implemented here.

Capacity samples
----------------
The paper fits the RTT slope dtau/dt = G/C - 1 (eqs. 15-16). HOP sends each
block as a burst, so all packets of a burst leave at (almost) the same time
and the bottleneck serves them back-to-back; the ACKs then come back spaced
by L/C. We therefore use the equivalent packet-train estimate
    C_sample = (n - 1) L / (a_n - a_1)
over the ACK times a_1..a_n of one burst - the same quantity the RTT
gradient measures when G >> C (see report, Sec. 3.4).

Filter
------
Random-walk state model (eqs. 17-18): x_{t+1} = x_t + w, C_t = x_t + v.
Adaptive Kalman filter: R and Q are tracked online from the innovations
(Sage-Husa style with forgetting factor), standing in for the recursive AKF
of the paper's ref. [54].

Special modes (eqs. 19-21, Fig. 2):
* outage: t_thr consecutive zero samples -> mean reset to 50 kB/s;
* step mode: |C_t - x| > 3 sqrt(Sigma) -> the filter runs open-loop; after
  t_thr consecutive step samples the mean is reset to their average.
p_out (eq. 34) = running fraction of recent samples that were outages or
down-steps.
"""

import numpy as np

MIN_RESET_BPS = 50e3 * 8          # 50 kB/s, the paper's outage reset value


class SkipEstimator:
    def __init__(self, t_thr=3, forget=0.05, pout_window=100):
        self.x = None
        self.P = None
        self.Q = None
        self.R = None
        self.t_thr = t_thr
        self.d = forget
        self.zero_run = 0
        self.step_samples = []
        self.flags = []
        self.pout_window = pout_window
        self.n = 0

    def update(self, z):
        """Feed one capacity sample (bits/s); 0 means 'nothing got through'."""
        self.n += 1
        if self.x is None:
            if z <= 0:
                return
            self.x, self.P = z, (0.3 * z) ** 2
            self.R, self.Q = (0.2 * z) ** 2, (0.02 * z) ** 2
            self._flag(False)
            return
        Pm = self.P + self.Q
        # ---- outage mode (eq. 19)
        if z <= 0:
            self.zero_run += 1
            self._flag(True)
            if self.zero_run >= self.t_thr and self.x > MIN_RESET_BPS:
                self.x = MIN_RESET_BPS
                self.P = Pm
            else:
                self.P = Pm                          # prediction step only
            return
        self.zero_run = 0
        nu = z - self.x
        S = Pm + self.R
        # ---- step mode (eqs. 20-21)
        if abs(nu) > 3.0 * np.sqrt(S):
            self.step_samples.append(z)
            self._flag(nu < 0)                      # only down-steps count towards p_out
            if len(self.step_samples) >= self.t_thr:
                self.x = float(np.mean(self.step_samples))
                self.P = self.R
                self.step_samples = []
            else:
                self.P = Pm                          # open-loop prediction
            return
        self.step_samples = []
        self._flag(False)
        K = Pm / S
        self.x = self.x + K * nu
        self.P = (1 - K) * Pm
        # ---- adaptive noise estimates
        self.R = (1 - self.d) * self.R + self.d * max(nu * nu - Pm, 1e-6 * self.x ** 2)
        self.Q = (1 - self.d) * self.Q + self.d * max((K * nu) ** 2, 1e-8 * self.x ** 2)

    def _flag(self, bad):
        self.flags.append(bad)
        if len(self.flags) > self.pout_window:
            self.flags.pop(0)

    @property
    def ready(self):
        return self.x is not None

    def predictive(self):
        """(mean, std, p_out) of next-slot capacity."""
        if self.x is None:
            return None
        std = np.sqrt(self.P + self.Q + self.R)
        p_out = float(np.mean(self.flags)) if self.flags else 0.0
        return self.x, std, min(p_out, 0.5)
