r"""
Entry point.

Runs the listed receivers at the reference operating point and stores the results
under results/.  The point is the setting of Karataev et al.: L = 16 single-antenna
APs on a 400 m grid, K = 8 users, 4-QAM, tau_d = 100, p = 14 dBm, noise -96 dBm,
eta = 0.7.  It differs from theirs in the pilots (tau_p = K orthogonal DFT pilots, 8x
the energy of their X_p = I, which --pilots identity runs) and in the APs, 10 m above
the users, so it is not a reproduction of their curves.

    python main.py                    # all seven receivers (about 3.5 hours on one core)
    python main.py icd he             # a subset
    python main.py --drops 5          # a quick smoke run
    python main.py --reals 20         # fewer fading realisations per drop (coarser SER samples)
    python main.py --pilots identity  # Karataev et al.'s pilots X_p = I, saved in results/identity/

A run whose budget differs from the reference (--drops or --reals) is saved under
results/quick/, so it never replaces the reference results that make_figure.py reads.
Runs with identity pilots go to results/identity/ (and results/identity/quick/).

n_channels sets how many symbols each per-UE SER sample averages (n_channels * tau_d), and
therefore the resolution of that sample: at 20 realisations it is 5e-4, coarser than the
SER of the better receivers, which makes the empirical cdf a staircase.  200 gives 5e-5.

Receivers: 'lmmse' (pilot-based linear), 'genie' (EP detection with perfect CSI, the
reference), 'icd' (per-user message variance), 'he' (user-averaged variance, the ICD
baseline Karataev et al. compare against), 'jcd' (bilinear EP), 'bigabp' (bilinear Gaussian
BP) and 'bigamp' (bilinear AMP).  Keeping 'icd' and 'he' apart is the point: the difference
between them is most of the reported JCD-over-ICD gap.
"""
from __future__ import annotations

import argparse
import dataclasses
import os

from config import SimConfig
from detectors.base import build_detector
import detectors                                    # noqa: F401  registers the receivers
from utils.run import RESULTS_DIR, run, save

ALL = ["lmmse", "genie", "he", "icd", "jcd", "bigabp", "bigamp"]
EXTRA = ["bigabp_node"]           # ablations: run only when named on the command line

# The reference point.  Fields not named here take the defaults in config.py.
BASE = SimConfig(L=16, Nr=1, K=8, D=400.0, p_tx_dbm=14.0, noise_dbm=-96.0,
                 M=4, mod_type="QAM", tau_p=8, tau_d=100, damping=0.7,
                 n_geometries=50, n_channels=200)

# ICD is nested (I inner detection sweeps inside R outer passes) while JCD is a single
# joint loop of J sweeps, so the counts are not the same quantity.
ITERS = {"lmmse": 1, "genie": 5, "he": 5, "icd": 5, "jcd": 10, "bigabp": 15, "bigamp": 150,
         "bigabp_node": 15}

# The EP receivers and BiGaBP share eta = 0.7 so that none is advantaged by tuning.  BiG-AMP
# cannot: its first-order expansions leave a remainder that does not vanish at this problem
# size, and at 0.7 the recursion diverges outright, so it is given its own damping and
# iteration budget and the comparison is reported as such rather than as a like-for-like one.
# BiGaBP keeps the per-edge corrections instead of the Onsager term and converges at 0.7 in a
# handful of sweeps.
DAMPING = {"bigamp": 0.15}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("receivers", nargs="*", metavar="receiver",
                        help=f"receivers to run, any of {', '.join(ALL + EXTRA)} "
                             f"(default: all but the ablations {', '.join(EXTRA)})")
    parser.add_argument("--drops", type=int, default=BASE.n_geometries,
                        help=f"random user placements (default {BASE.n_geometries})")
    parser.add_argument("--reals", type=int, default=BASE.n_channels,
                        help=f"fading realisations per placement (default {BASE.n_channels})")
    parser.add_argument("--pilots", choices=["dft", "identity"], default=BASE.pilot_type,
                        help="pilot design: dft (energy p tau_p, default) or identity "
                             "(X_p = I as Karataev et al., energy p)")
    args = parser.parse_args()
    unknown = [r for r in args.receivers if r not in ALL + EXTRA]
    if unknown:
        parser.error(f"unknown receiver {', '.join(unknown)}, choose from {', '.join(ALL + EXTRA)}")
    if args.drops < 1 or args.reals < 1:
        parser.error("--drops and --reals must be at least 1")

    # a non-reference budget goes to results/quick/, out of make_figure.py's way
    quick = (args.drops, args.reals) != (BASE.n_geometries, BASE.n_channels)
    out_dir = RESULTS_DIR if args.pilots == BASE.pilot_type else os.path.join(RESULTS_DIR, args.pilots)
    if quick:
        out_dir = os.path.join(out_dir, "quick")

    for name in (args.receivers or ALL):
        cfg = dataclasses.replace(BASE, detector=name, n_iters=ITERS[name],
                                  damping=DAMPING.get(name, BASE.damping),
                                  n_geometries=args.drops, n_channels=args.reals,
                                  pilot_type=args.pilots)
        print(f"===== {name} =====", flush=True)
        res = run(cfg, build_detector(name, config=cfg))
        path = save(res, out_dir)
        print(f"  SER {res.ser:.4e}   NMSE {res.nmse:.4e}   -> {path}", flush=True)


if __name__ == "__main__":
    main()
