r"""
Build the results figure from results/ and write it into figures/.

Two panels, both distributions rather than averages:
  (a) the per-UE symbol error rate over all drops and users,
  (b) the per-link channel NMSE, restricted to links the receiver can actually see.

The weak-link restriction in (b) is the cut stated by Karataev et al.: a link is kept
when its received power exceeds the noise, p*beta_{lk} > sigma^2.  Without it the
distribution is dominated by links that carry no signal, where every estimator returns
the prior mean and the NMSE saturates at one, which says nothing about the receiver.
The genie is omitted from (b), where its NMSE is zero by construction.

Only results/ is read, never results/quick/, and every curve must come from the same
operating point (the receivers may differ only in their iterations and damping).

Run (after main.py):  python make_figure.py                     # results/ -> figures/reproduction.pdf
                      python make_figure.py --pilots identity   # results/identity/ -> reproduction_identity.pdf
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from config import SimConfig
from utils import figstyle  # noqa: F401  sets the PDF/font options
from utils.run import RESULTS_DIR, load

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# label, colour, line style.  'he' and 'icd' share a colour family on purpose, since
# the point of the figure is that they differ only in the message variance.
CURVES = [
    ("lmmse",  "L-MMSE (pilots only)",      "tab:gray",   "-"),
    ("he",     "ICD, averaged variance",    "tab:red",    "--"),
    ("icd",    "ICD, per-user variance",    "tab:orange", "-"),
    ("jcd",    "JCD (bilinear EP)",         "tab:blue",   "-"),
    ("bigabp", "BiGaBP (BP)",               "tab:purple", "-."),
    ("bigamp", "BiG-AMP",                   "tab:green",  "-."),
    ("genie",  "EP detection, perfect CSI", "black",      ":"),
]

# config fields that legitimately differ between the receivers of one operating point
PER_RECEIVER = ("detector", "n_iters", "damping")


def operating_point(cfg: SimConfig) -> dict:
    """The config without its per-receiver fields, to check that all curves share one point."""
    d = cfg.to_dict()
    for k in PER_RECEIVER:
        d.pop(k)
    return d


def ecdf(x):
    """Empirical cdf of x, returned as (sorted values, cumulative probability)."""
    x = np.sort(np.asarray(x).ravel())
    return x, np.arange(1, x.size + 1) / x.size


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="results/ -> summary table and figures/reproduction.pdf")
    parser.add_argument("--pilots", choices=["dft", "identity"], default="dft")
    pilots = parser.parse_args().pilots
    results_dir = RESULTS_DIR if pilots == "dft" else os.path.join(RESULTS_DIR, pilots)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    print(f"{'receiver':>26}   {'SER (median)':>13} {'SER (mean)':>11} {'NMSE (median)':>14}"
          f" {'zeros':>7}")
    floor = point = None
    for name, label, colour, style in CURVES:
        res = load(name, results_dir)
        if res is None:
            print(f"  {name}: no results/{name}.npz, run main.py first")
            continue
        c = res.cfg                                     # the config the run was produced with
        if point is None:
            point = operating_point(c)
        elif operating_point(c) != point:
            raise SystemExit(f"results/{name}.npz was run at a different operating point from "
                             f"the receivers above, re-run main.py for all of them")

        # A per-UE sample is an average over n_channels * tau_d symbols, so it is quantised
        # in steps of `resolution` and a UE with no errors sits at exactly zero, which a log
        # axis cannot show.  Put those at half a step and say so in the caption.
        resolution = 1.0 / (c.n_channels * c.tau_d)
        floor = resolution / 2
        zeros = float((res.ser_ue == 0).mean())

        x, F = ecdf(res.ser_ue)
        ax[0].semilogx(np.maximum(x, floor), F, color=colour, ls=style, label=label, lw=1.6)

        keep = c.p_tx * res.beta_link > c.N0            # the weak-link cut
        med_nmse = float("nan")
        if name != "genie":
            xn, Fn = ecdf(res.nmse_link[keep])
            ax[1].semilogx(xn, Fn, color=colour, ls=style, label=label, lw=1.6)
            med_nmse = float(np.median(res.nmse_link[keep]))
        print(f"{label:>26}   {np.median(res.ser_ue):13.3e} {res.ser:11.3e} {med_nmse:14.3e}"
              f" {100 * zeros:6.1f}%")

    if floor is not None:
        ax[0].set_xlim(left=floor * 0.7)
    ax[0].set_xlabel("per-UE symbol error rate")
    ax[0].set_ylabel("empirical cdf")
    ax[0].set_title("(a) detection")
    ax[1].set_xlabel(r"per-link channel NMSE,  $p\beta_{lk} > \sigma^2$")
    ax[1].set_title("(b) channel estimation")
    for a in ax:
        a.grid(True, which="both", alpha=0.3)
        a.set_ylim(0, 1)
        a.legend(loc="lower right", fontsize=8)

    os.makedirs(FIGDIR, exist_ok=True)
    stem = "reproduction" if pilots == "dft" else f"reproduction_{pilots}"
    out = os.path.join(FIGDIR, f"{stem}.{figstyle.FIG_EXT}")
    fig.tight_layout()
    fig.savefig(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
