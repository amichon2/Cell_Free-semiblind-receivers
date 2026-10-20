r"""
Cell-free channel.

Large-scale fading follows the 3GPP urban-microcell law of Bjornson and Sanguinetti
(reference [7] of README.md, eq. (37)), as used by Karataev et al.,
    beta_{lk}[dB] = -30.5 - 36.7 log10(d_{lk}) + z_{lk},   z ~ N(0, sigma_sf^2),
specialised to uncorrelated Rayleigh fading, R_{lk} = beta_{lk} I_{N_r}.  APs sit
on a corner-anchored square grid at height AP_HEIGHT_M above the UE plane, so the
link distance is three-dimensional and a UE can never land on top of an AP.  UEs
are dropped uniformly in the D x D square.

The geometry has its own RNG (config.geometry_seed), so a layout can be pinned
while the fading, drawn with the generator passed to draw()/apply(), is averaged
over it.

Conventions (PyTorch, batched):
    beta : (L, K)              large-scale fading
    H    : (batch, L, Nr, K)   small-scale channel, h_{l,r,k} ~ CN(0, beta_lk)
    X    : (batch, K, tau)     transmitted block
    Y    : (batch, L, Nr, tau) received block, Y = sqrt(p) H X + W
"""
from __future__ import annotations

import math

import torch

from config import AP_HEIGHT_M, MIN_DIST_M


class CellFreeChannel:

    def __init__(self, config, device: str = "cpu"):
        self.cfg = config
        self.device = device
        self.geo_rng = torch.Generator(device=device)
        self.geo_rng.manual_seed(config.geometry_seed)
        self.ap_pos = self._place_aps()                     # (L, 2), fixed for the whole run

    # ---------------- geometry ----------------
    def _place_aps(self) -> torch.Tensor:
        """Corner-anchored square grid: coordinates {i D/(g-1)}, i = 0..g-1, g = sqrt(L)."""
        c = self.cfg
        g = int(round(math.sqrt(c.L)))
        coords = torch.arange(g, device=self.device) * (c.D / (g - 1))   # (g,)
        gx, gy = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([gx.flatten(), gy.flatten()], dim=1)          # (L, 2)

    def _distance(self, ap_pos: torch.Tensor, ue_pos: torch.Tensor) -> torch.Tensor:
        delta = ap_pos[:, None, :] - ue_pos[None, :, :]     # (L, K, 2) horizontal offset
        d2d = torch.sqrt((delta ** 2).sum(-1))              # (L, K) horizontal distance
        # 3-D distance: the height difference floors d at AP_HEIGHT_M, so a UE on top of
        # an AP does not manufacture a link tens of dB stronger than the model intends.
        return torch.sqrt(d2d ** 2 + AP_HEIGHT_M ** 2)      # (L, K)

    def new_geometry(self) -> torch.Tensor:
        """Draw a UE placement and return the large-scale fading beta, shape (L, K)."""
        c = self.cfg
        ue_pos = torch.rand(c.K, 2, generator=self.geo_rng, device=self.device) * c.D  # (K, 2)
        self.ue_pos = ue_pos
        d = self._distance(self.ap_pos, ue_pos).clamp_min(MIN_DIST_M)   # (L, K)
        beta_db = -30.5 - 36.7 * torch.log10(d)                         # (L, K)
        if c.sigma_sf_db > 0:
            beta_db = beta_db + c.sigma_sf_db * torch.randn(
                c.L, c.K, generator=self.geo_rng, device=self.device)
        return (10 ** (beta_db / 10)).to(torch.float32)                 # (L, K) linear

    # ---------------- small-scale fading ----------------
    def draw(self, beta: torch.Tensor, batch: int,
             generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw a batch of channels H ~ CN(0, beta), shape (batch, L, Nr, K)."""
        c = self.cfg
        std = torch.sqrt(beta / 2.0)[None, :, None, :]      # (1, L, 1, K) per-real-dim std
        re = torch.randn(batch, c.L, c.Nr, c.K, generator=generator, device=self.device)
        im = torch.randn(batch, c.L, c.Nr, c.K, generator=generator, device=self.device)
        return (re * std + 1j * im * std).to(torch.complex64)           # (batch, L, Nr, K)

    # ---------------- mixing + noise ----------------
    def apply(self, H: torch.Tensor, X: torch.Tensor,
              generator: torch.Generator | None = None) -> torch.Tensor:
        """Pass X (batch, K, tau) through H and return Y = sqrt(p) H X + W (batch, L, Nr, tau)."""
        c = self.cfg
        # equal transmit power -> a scalar sqrt(p) in front (per-user powers would fold into X)
        Y = math.sqrt(c.p_tx) * torch.einsum("blrk,bkt->blrt", H, X.to(H.dtype))
        b, L, Nr, tau = Y.shape
        std = math.sqrt(c.N0 / 2.0)                         # W ~ CN(0, N0)
        re = torch.randn(b, L, Nr, tau, generator=generator, device=self.device)
        im = torch.randn(b, L, Nr, tau, generator=generator, device=self.device)
        return Y + (re + 1j * im).to(Y.dtype) * std         # (batch, L, Nr, tau)
