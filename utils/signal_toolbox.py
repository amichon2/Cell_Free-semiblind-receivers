"""
Signal toolbox: constellation construction and symbol (de)mapping.

Constellations are Gray-labelled by default: the integer index of a point is its
bit label, and nearest-neighbour points differ in exactly one bit.  This is
irrelevant for SER but matters for BER / soft decoding later.  All constellations
are normalised to unit average symbol energy, E{|x|^2} = 1.
"""
from __future__ import annotations

import math
import torch


def _gray(n: int) -> torch.Tensor:
    """Binary-reflected Gray code of 0..n-1:  g(m) = m XOR (m >> 1)."""
    m = torch.arange(n)
    return m ^ (m >> 1)


def make_constellation(M: int, mod_type: str, gray: bool = True) -> torch.Tensor:
    """Return the constellation as a (M,) complex64 tensor, unit average energy.

    With gray=True the array is indexed by Gray label, with gray=False it is in
    natural (raw binary) order.
    """
    if mod_type == "PSK":
        m = torch.arange(M)
        points = torch.exp(1j * 2 * math.pi * m / M)       # natural order, \exp(j 2 \pi m / M)
        const = torch.empty(M, dtype=torch.complex64)
        if gray:
            const[_gray(M)] = points.to(torch.complex64)   # const[gray(m)] = point(m)
        else:
            const = points.to(torch.complex64)

    elif mod_type == "QAM":
        nbits = int(round(math.log2(M)))                   # log2(M) bits per symbol
        if nbits % 2 != 0:
            raise ValueError("square QAM requires an even number of bits per symbol, log2(M)")
        s = 1 << (nbits // 2)                              # levels per axis, sqrt(M)
        hb = nbits // 2                                    # bits per axis
        levels = (2 * torch.arange(s) - (s - 1)).to(torch.float32)  # amplitudes by position
        amp = torch.empty(s, dtype=torch.float32)
        if gray:
            amp[_gray(s)] = levels                         # amp[gray(a)] = level(a), per-axis Gray
        else:
            amp = levels
        i = torch.arange(M)
        re = amp[i >> hb]                                  # high half -> in-phase
        im = amp[i & (s - 1)]                              # low half  -> quadrature
        const = (re + 1j * im).to(torch.complex64)
        const = const / torch.sqrt((const.abs() ** 2).mean())  # unit average energy

    else:
        raise ValueError(f"unknown mod_type '{mod_type}'")

    return const.to(torch.complex64)


def modulate(indices: torch.Tensor, constellation: torch.Tensor) -> torch.Tensor:
    """Map integer indices (bit labels) to constellation symbols."""
    return constellation[indices]


def hard_demap(symbols: torch.Tensor, constellation: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour symbol decision, returned as integer indices (bit labels).

    The constellation axis is reduced and the leading shape of symbols is kept.
    """
    dist = (symbols.unsqueeze(-1) - constellation).abs()   # (..., M)
    return dist.argmin(dim=-1)