r"""
Configuration for the cell-free JCD simulator.

One SimConfig fully describes a run.  Only the quantities a study varies are fields.
Everything the simulator fixes is a module constant below rather than a knob.  Derived
quantities (N, tau_c, linear powers, ...) are properties, never stored, so they cannot
drift out of sync with the base fields.

Sweeping one of these constants is outside what this version supports, which is why
they are not fields.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any
import json
import math

# --- fixed, deliberately not parameters ------------------------------------------
LAYOUT       = "grid"       # APs on a square grid, corner-anchored at {i D/(g-1)}
WRAP         = False        # plain Euclidean distances (no torus/wrap-around)
AP_HEIGHT_M  = 10.0         # AP height above the UE plane [m] -> distances are 3-D
MIN_DIST_M   = 1.0          # floor on the AP-UE distance, avoids log10(0)
VAR_MIN      = 1e-6         # numerical floor on message variances
PRECISION    = "float64"    # every EP recursion runs in double precision
# (the variance *ceilings* differ between the two receivers and live next to the
#  code that needs them: VAR_MAX in detectors/icd.py and detectors/jcd.py.)


@dataclass
class SimConfig:
    # --- network geometry ---
    L: int = 16                     # number of APs
    Nr: int = 1                     # antennas per AP, N_r
    K: int = 8                      # single-antenna users
    D: float = 400.0                # square side length [m]

    # --- large-scale fading and noise (path loss in system/channel.py) ---
    sigma_sf_db: float = 0.0        # shadowing std [dB], 0 for path loss only
    p_tx_dbm: float = 14.0          # equal uplink power per user [dBm]
    noise_dbm: float = -96.0        # noise power [dBm]

    # --- modulation and frame structure ---
    M: int = 4                      # constellation size
    mod_type: str = "QAM"           # 'PSK' or 'QAM'
    tau_p: int = 8                  # pilot symbols (orthogonal, so tau_p >= K)
    pilot_type: str = "dft"         # 'dft': energy p tau_p per user, the usual design;
                                    # 'identity': X_p = I as Karataev et al., energy p per user
    tau_d: int = 100                # data symbols, \tau_d

    # --- receiver ---
    detector: str = "jcd"           # name resolved in the detectors registry
    damping: float = 0.7            # soft-update factor \eta in (0, 1]
    n_iters: int = 10               # JCD joint sweeps J, or ICD inner EP sweeps I
    n_outer: int = 3                # ICD outer detect<->re-estimate passes R (ICD only)

    # --- Monte-Carlo control ---
    seed: int = 42                  # fading / data RNG
    geometry_seed: int = 123        # separate RNG for the AP/UE layout
    n_geometries: int = 50          # random UE drops
    n_channels: int = 20            # small-scale realisations per drop

    # ---------------- derived ----------------
    @property
    def N(self) -> int:             # total receive antennas, N = L N_r
        return self.L * self.Nr

    @property
    def tau_c(self) -> int:         # coherence block, tau_c = tau_p + tau_d
        return self.tau_p + self.tau_d

    @property
    def p_tx(self) -> float:        # linear TX power [W]
        return 10 ** ((self.p_tx_dbm - 30) / 10)

    @property
    def N0(self) -> float:          # noise power [W], sigma^2
        return 10 ** ((self.noise_dbm - 30) / 10)

    # ---------------- validation / (de)serialisation ----------------
    def validate(self) -> None:
        assert math.log2(self.M).is_integer(), "M must be a power of two (log2(M) bits per symbol)"
        assert self.mod_type in ("PSK", "QAM"), "mod_type must be 'PSK' or 'QAM'"
        assert 0.0 < self.damping <= 1.0, "damping \\eta must lie in (0, 1]"
        assert self.tau_p >= self.K, "orthogonal pilots require \\tau_p >= K"
        assert self.pilot_type in ("dft", "identity"), "pilot_type must be 'dft' or 'identity'"
        g = int(round(self.L ** 0.5))
        assert g * g == self.L, "the square AP grid requires L to be a perfect square"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["_derived"] = {"N": self.N, "tau_c": self.tau_c}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SimConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
