r"""
Transmitter.

Builds the uplink frame for all K users: tau_p orthogonal pilot symbols followed
by tau_d i.i.d. data symbols drawn uniformly from the constellation.  Symbols are
unit energy, and the transmit-power scaling sqrt(p) is applied by the channel.

By default the pilots are unit-modulus (DFT) sequences, phi_k[t] = exp(j 2 pi k t / tau_p):
they are mutually orthogonal for tau_p >= K and spend energy E_k = ||phi_k||^2 = tau_p per
user, the E_k of detectors/pilot_estimation.py.  This is the usual design.  With pilot_type = 'identity'
they are X_p = I as in Karataev et al.: user k transmits in slot k only, energy E_k = 1.
The pilot estimate depends on the design only through p E_k, so the two differ in
training energy, by a factor tau_p.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from utils.signal_toolbox import make_constellation, modulate


@dataclass
class Frame:
    data_idx: torch.Tensor          # (batch, K, tau_d) constellation indices (bit labels)
    data_sym: torch.Tensor          # (batch, K, tau_d) data symbols
    pilots: torch.Tensor            # (K, tau_p) known pilot symbols
    X: torch.Tensor                 # (batch, K, tau_c) full transmitted block


class Transmitter:

    def __init__(self, config, device: str = "cpu"):
        self.cfg = config
        self.device = device
        self.constellation = make_constellation(config.M, config.mod_type).to(device)  # (M,)
        self.pilots = self._make_pilots()                          # (K, tau_p)

    def _make_pilots(self) -> torch.Tensor:
        K, tp = self.cfg.K, self.cfg.tau_p
        if self.cfg.pilot_type == "identity":                     # X_p = I, user k in slot k
            phi = torch.zeros(K, tp, dtype=torch.complex64, device=self.device)   # (K, tau_p)
            phi[torch.arange(K), torch.arange(K)] = 1.0
            return phi
        k = torch.arange(K, device=self.device).unsqueeze(1)       # (K, 1)
        t = torch.arange(tp, device=self.device).unsqueeze(0)      # (1, tau_p)
        return torch.exp(1j * 2 * math.pi * k * t / tp).to(torch.complex64)   # (K, tau_p)

    def generate(self, batch: int, generator: torch.Generator | None = None) -> Frame:
        K, td, M = self.cfg.K, self.cfg.tau_d, self.cfg.M
        idx = torch.randint(0, M, (batch, K, td), generator=generator, device=self.device)
        data_sym = modulate(idx, self.constellation)               # (batch, K, tau_d)
        pilots_b = self.pilots.unsqueeze(0).expand(batch, -1, -1)  # (batch, K, tau_p)
        X = torch.cat([pilots_b, data_sym], dim=-1)                # (batch, K, tau_c)
        return Frame(data_idx=idx, data_sym=data_sym, pilots=self.pilots, X=X)
