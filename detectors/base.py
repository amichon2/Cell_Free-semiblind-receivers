"""
Detector interface.

Every receiver implements detect(obs) -> (x_hat, h_hat), so the Monte-Carlo loop
is agnostic to the receiver type.  The Observation bundles everything a detector
might need, and each uses only the fields relevant to it (the genie reads H_true,
the pilot-based receivers read y_pilot and pilots).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import torch


@dataclass
class Observation:
    y_data: torch.Tensor                # (batch, L, Nr, tau_d) received data signal
    y_pilot: torch.Tensor | None        # (batch, L, Nr, tau_p) received pilots
    pilots: torch.Tensor | None         # (K, tau_p) known pilot symbols
    constellation: torch.Tensor         # (M,) complex
    N0: float                           # noise variance, \sigma^2
    p_tx: float                         # linear TX power, p
    beta: torch.Tensor                  # (L, K) large-scale fading, \beta_{lk}
    H_true: torch.Tensor | None = None  # (batch, L, Nr, K) true channel, for the genie only


def to_double(obs: "Observation") -> "Observation":
    """Return a copy of obs in double precision.

    The EP recursions run in float64 throughout, as a safety margin.  The messages live
    at the scale of the received signal, where the powers and the noise are both ~1e-13
    in linear units, so the exchanged precisions are ~1e13 and a single-precision
    difference of two of them keeps few significant digits.  With the negative-precision
    guard in place, single precision gave the same results in our tests.
    """
    c = lambda t: None if t is None else t.to(torch.complex128)
    r = lambda t: None if t is None else t.to(torch.float64)
    return replace(obs, y_data=c(obs.y_data), y_pilot=c(obs.y_pilot),
                   pilots=c(obs.pilots), constellation=c(obs.constellation),
                   beta=r(obs.beta), H_true=c(obs.H_true))


class Detector(ABC):
    """Base receiver. Subclasses return (x_hat, h_hat)."""
    name: str = "base"

    @abstractmethod
    def detect(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (x_hat, h_hat).

        x_hat : (batch, K, tau_d) detected data symbols
        h_hat : (batch, L, Nr, K) channel estimate, or None if not estimated
        """
        raise NotImplementedError


# --- minimal name -> class registry, so config.detector resolves to a receiver ---
_REGISTRY: dict[str, type[Detector]] = {}


def register(name: str):
    def deco(cls: type[Detector]) -> type[Detector]:
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def build_detector(name: str, **kwargs) -> Detector:
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector '{name}', registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
