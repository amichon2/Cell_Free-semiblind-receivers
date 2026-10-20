"""
Local combining and large-scale fading decoding (LSFD).

This is Level 3 in the taxonomy of Bjornson and Sanguinetti ("Making cell-free massive
MIMO competitive...", TWC 2020): each AP forms a local soft estimate, then the CPU fuses
the L estimates with weights a_k computed from channel statistics only.  These helpers
are combiner-agnostic: a detector supplies its local combining vectors V (and an
effective channel H for the statistics) and gets back the fused estimate.

The LSFD weights depend only on large-scale statistics; here they are estimated
by averaging over the batch of small-scale realisations sharing one geometry:
    a_k = ( sum_i p_i E{psi_ki psi_ki^H} + sigma^2 Lambda_k )^{-1} sqrt(p_k) E{psi_kk}
with psi_ki[l] = v_{lk}^H h_{li} and Lambda_k = diag(E{||v_{lk}||^2}).
"""
from __future__ import annotations

import torch


def local_estimates(V: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-AP soft estimates xcheck_{lk}[t] = v_{lk}^H y_l[t].

    V (B, L, Nr, K), y (B, L, Nr, T) -> xcheck (B, L, K, T).
    """
    return torch.einsum("blrk,blrt->blkt", V.conj(), y)


def lsfd_weights(V: torch.Tensor, H: torch.Tensor, p: float, N0: float,
                 jitter: float = 1e-9) -> torch.Tensor:
    """SINR-optimal LSFD weights a_k, estimated over the batch. Returns (K, L)."""
    # psi[b,l,k,i] = v_{lk}^H h_{li}                              (B, L, K, K)
    psi = torch.einsum("blrk,blri->blki", V.conj(), H)
    B, L, K, _ = psi.shape

    # E{psi_kk}                                                   (K, L)
    Epsi_kk = psi.diagonal(dim1=2, dim2=3).mean(0).transpose(0, 1)

    # sum_i p_i E{psi_ki psi_ki^H}                                (K, L, L)
    psi_p = psi * p                                               # equal power p_i = p
    M = torch.einsum("blki,bmki->klm", psi_p, psi.conj()) / B

    # Lambda_k = diag(E{||v_{lk}||^2})                            (K, L)
    vnorm = (V.abs() ** 2).sum(dim=2).mean(0).transpose(0, 1)     # (K, L)

    eye = torch.eye(L, dtype=M.dtype, device=M.device)
    Mreg = M + N0 * torch.diag_embed(vnorm.to(M.dtype)) + jitter * eye
    rhs = (p ** 0.5) * Epsi_kk                                    # (K, L)
    a = torch.linalg.solve(Mreg, rhs.unsqueeze(-1)).squeeze(-1)   # (K, L)
    return a


def fuse(a: torch.Tensor, xcheck: torch.Tensor) -> torch.Tensor:
    """CPU fusion xhat_k = sum_l a_{lk}^* xcheck_{lk}. a (K, L), xcheck (B,L,K,T)."""
    return torch.einsum("kl,blkt->bkt", a.conj(), xcheck)