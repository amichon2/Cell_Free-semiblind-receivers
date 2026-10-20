"""
Local-MMSE detector with LSFD (Level 3 of Bjornson and Sanguinetti, semi-blind).

Estimates the channel from pilots (pilot_mmse), forms the local L-MMSE combiner
that accounts for interference and the residual estimation error, and fuses the
per-AP estimates with LSFD.  Requires pilots (tau_p > 0); it cannot run blind.

Local L-MMSE combiner at AP l for user k:
    v_{lk} = ( sum_i p_i ( hhat_{li} hhat_{li}^H + C_{li} ) + sigma^2 I )^{-1} sqrt(p_k) hhat_{lk}
With C_{li} = c_{li} I, the inverse term is  p Hhat_l Hhat_l^H + (p sum_i c_{li} + sigma^2) I.
"""
from __future__ import annotations

import torch

from detectors.base import Detector, Observation, register
from detectors.lsfd import local_estimates, lsfd_weights, fuse
from detectors.pilot_estimation import pilot_mmse
from utils.signal_toolbox import hard_demap


@register("lmmse")
class LMMSEDetector(Detector):
    """Pilot-based local L-MMSE combining with LSFD fusion (Level 3)."""

    def __init__(self, config):
        self.cfg = config

    def detect(self, obs: Observation):
        if obs.y_pilot is None or obs.pilots is None:
            raise ValueError("L-MMSE needs pilots (tau_p > 0), it cannot run blind")

        p, N0 = obs.p_tx, obs.N0
        h_hat, c = pilot_mmse(obs.y_pilot, obs.pilots, obs.beta, p, N0)  # (B,L,Nr,K), (L,K)
        B, L, Nr, K = h_hat.shape

        # R_l = p Hhat_l Hhat_l^H + (p sum_i c_{li} + sigma^2) I            (B,L,Nr,Nr)
        R = p * torch.einsum("blrk,blsk->blrs", h_hat, h_hat.conj())
        diag = (p * c.sum(dim=1) + N0)                                     # (L,)
        eye = torch.eye(Nr, dtype=R.dtype, device=R.device)
        R = R + diag.view(1, L, 1, 1) * eye

        # Local L-MMSE combiner V_l = R_l^{-1} sqrt(p) Hhat_l               (B,L,Nr,K)
        V = (p ** 0.5) * torch.einsum("blrs,blsk->blrk", torch.linalg.inv(R), h_hat)

        # LSFD statistics from the channel estimates (the receiver never sees the true channel).
        # Bjornson and Sanguinetti's eqs. (20)-(21) use the statistics of the true channel: the
        # means agree, the second moments lose the estimation-error term, and the SER differs
        # by about 3% at the reference point.
        xcheck = local_estimates(V, obs.y_data)
        a = lsfd_weights(V, h_hat, p, N0)
        xhat = fuse(a, xcheck)

        idx = hard_demap(xhat, obs.constellation)
        return obs.constellation[idx], h_hat