r"""
Monte-Carlo runner and metrics.

One run is: for each of n_geometries UE drops, draw n_channels fading realisations,
transmit a frame, pass it through the channel, detect, and score.  Alongside the two
aggregate numbers it keeps the per-UE and per-link samples, because the figures are
distributions rather than averages.  The receivers behave very differently across the
near-far spread of a random deployment, and a single mean hides the tail that separates
them.

Metrics:
    SER   = fraction of wrongly detected symbols
    NMSE  = mean over links of ||hhat_{lk} - h_{lk}||^2 / ||h_{lk}||^2

Results are stored one file per receiver with its config embedded, which make_figure.py
uses to refuse curves from different operating points.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

from config import SimConfig
from detectors.base import Observation
from system.channel import CellFreeChannel
from system.transmitter import Transmitter
from utils.signal_toolbox import hard_demap

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


@dataclass
class Result:
    ser: float                  # aggregate symbol error rate
    nmse: float                 # aggregate channel NMSE
    ser_ue: np.ndarray          # (n_geometries * K,)     per-UE SER samples
    nmse_link: np.ndarray       # (n_geometries * L * K,) per-link NMSE samples
    beta_link: np.ndarray       # (n_geometries * L * K,) the matching beta_{lk}, for the weak-link cut
    cfg: SimConfig              # the config the run was produced with


def run(cfg: SimConfig, detector, verbose: bool = True, chunk: int = 20) -> Result:
    """Monte-Carlo over cfg.n_geometries drops of cfg.n_channels fading realisations.

    The realisations of one drop are processed in chunks of `chunk` and the per-UE and
    per-link scores accumulated across them, so that n_channels controls the *resolution*
    of a sample without controlling peak memory.  That matters, because the per-UE SER
    sample is an average over n_channels * tau_d symbols and is therefore quantised in
    steps of 1/(n_channels*tau_d).  At n_channels = 20 that is 5e-4, coarser than the SER
    of the better receivers, which turns the empirical cdf into a staircase.  Raising the
    number of drops does not help, only more symbols per sample does.  Meanwhile the JCD
    holds a (chunk, N, K, tau_d, |Omega|) array of categorical messages, so the chunk is
    what has to stay small.
    """
    cfg.validate()
    # Single-threaded on purpose.  Multithreaded BLAS changes the order of summation, and
    # the ICD recursion amplifies those last-digit differences into flipped symbols, so
    # repeated runs would differ.  One thread costs little at these sizes and makes every
    # run reproducible on a given machine.
    torch.set_num_threads(1)

    ch, tx = CellFreeChannel(cfg), Transmitter(cfg)
    gen = torch.Generator().manual_seed(cfg.seed)
    tp = cfg.tau_p

    n_err = n_sym = 0
    ser_ue, nmse_link, beta_link = [], [], []
    for g in range(cfg.n_geometries):
        beta = ch.new_geometry()                                    # (L, K)
        err_k = torch.zeros(cfg.K, dtype=torch.float64)             # (K,)   error count this drop
        sq_lk = torch.zeros(cfg.L, cfg.K, dtype=torch.float64)      # (L, K) summed NMSE ratios
        n_per_ue = n_real = 0
        for start in range(0, cfg.n_channels, chunk):
            b = min(chunk, cfg.n_channels - start)
            H = ch.draw(beta, b, generator=gen)                     # (b, L, Nr, K)
            fr = tx.generate(b, generator=gen)
            Y = ch.apply(H, fr.X, generator=gen)                    # (b, L, Nr, tau_c)
            obs = Observation(y_data=Y[..., tp:], y_pilot=Y[..., :tp], pilots=fr.pilots,
                              constellation=tx.constellation, N0=cfg.N0, p_tx=cfg.p_tx,
                              beta=beta, H_true=H)

            x_hat, h_hat = detector.detect(obs)
            err = hard_demap(x_hat, tx.constellation) != fr.data_idx  # (b, K, tau_d)
            n_err += int(err.sum())
            n_sym += err.numel()
            err_k += err.sum(dim=(0, 2)).double().cpu()             # (K,)
            n_per_ue += b * cfg.tau_d

            num = (h_hat - H).abs().pow(2).sum(dim=2)               # (b, L, K)
            den = H.abs().pow(2).sum(dim=2).clamp_min(1e-12)        # (b, L, K)
            sq_lk += (num / den).sum(dim=0).double().cpu()          # (L, K)
            n_real += b

        ser_ue.append((err_k / n_per_ue).numpy())                   # (K,)   per-UE SER
        nmse_link.append((sq_lk / n_real).numpy())                  # (L, K) per-link NMSE
        beta_link.append(beta.cpu().numpy())                        # (L, K)

        if verbose and (g + 1) % 5 == 0:
            print(f"  {cfg.detector}: {g + 1}/{cfg.n_geometries} drops, "
                  f"SER {n_err / n_sym:.3e}", flush=True)

    nmse_link = np.concatenate([a.ravel() for a in nmse_link])
    return Result(ser=n_err / n_sym,
                  nmse=float(nmse_link.mean()),
                  ser_ue=np.concatenate([a.ravel() for a in ser_ue]),
                  nmse_link=nmse_link,
                  beta_link=np.concatenate([a.ravel() for a in beta_link]),
                  cfg=cfg)


# ---------------------------------------------------------------- persistence
def save(res: Result, out_dir: str = RESULTS_DIR) -> str:
    """Write res to out_dir/<detector>.npz and return the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{res.cfg.detector}.npz")
    np.savez_compressed(path, config=json.dumps(res.cfg.to_dict()), ser=res.ser, nmse=res.nmse,
                        ser_ue=res.ser_ue, nmse_link=res.nmse_link, beta_link=res.beta_link)
    return path


def load(detector: str, results_dir: str = RESULTS_DIR) -> Result | None:
    """Load results_dir/<detector>.npz, or return None if it is missing."""
    path = os.path.join(results_dir, f"{detector}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=False)
    return Result(ser=float(d["ser"]), nmse=float(d["nmse"]), ser_ue=d["ser_ue"],
                  nmse_link=d["nmse_link"], beta_link=d["beta_link"],
                  cfg=SimConfig.from_dict(json.loads(str(d["config"]))))
