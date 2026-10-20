r"""
Pilot-based MMSE channel estimation with orthogonal pilots, the estimator of Bjornson and
Sanguinetti [7] eqs. (3)-(5) (numbers in brackets refer to README.md) for uncorrelated
fading.

Correlating the received pilot block at AP l with user k's pilot gives the
sufficient statistic of [7] eq. (3) (here without its normalisation by the pilot norm),
    yp_{lk} = sqrt(p) E_k h_{lk} + w,     w ~ CN(0, N0 E_k),
with the pilot energy E_k = ||phi_k||^2 (tau_p for the unit-modulus design).  The
estimate and its per-antenna error variance depend on the pilot design only through
the product p E_k:
    hhat_{lk} = ( sqrt(p) beta_{lk} / (p E_k beta_{lk} + N0) ) yp_{lk}
    c_{lk}    = beta_{lk} N0 / (p E_k beta_{lk} + N0)  =  beta_{lk} - gamma_{lk}
where gamma_{lk} = p E_k beta_{lk}^2 / (p E_k beta_{lk} + N0) is the variance of the
estimate.
"""
from __future__ import annotations

import torch


def pilot_mmse(y_pilot: torch.Tensor, pilots: torch.Tensor,
               beta: torch.Tensor, p: float, N0: float):
    """MMSE channel estimate from the pilot block.

    y_pilot (B, L, Nr, tau_p), pilots (K, tau_p), beta (L, K).
    Returns h_hat (B, L, Nr, K) and c (L, K), the per-antenna error variance.
    """
    yp = torch.einsum("blrt,kt->blrk", y_pilot, pilots.conj())     # (B, L, Nr, K)
    E = (pilots.abs() ** 2).sum(dim=1).to(beta.dtype)              # (K,) pilot energy
    pE = p * E[None, :]                                            # (1, K), broadcasts over L

    denom = pE * beta + N0                             # (L, K)
    scale = (p ** 0.5) * beta / denom                  # (L, K)
    gamma = pE * beta ** 2 / denom                     # (L, K)
    c = beta - gamma                                   # (L, K) error variance

    h_hat = scale[None, :, None, :] * yp               # (B, L, Nr, K)
    return h_hat, c
