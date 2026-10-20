# Semi-blind channel estimation and data detection in cell-free massive MIMO

This repository simulates the uplink of a cell-free massive MIMO system and compares classical (non-learned) iterative receivers that estimate the channel and detect the data from the same block.
Each receiver is written to be read next to the paper it comes from, with the paper's symbols where possible and the same order of operations.

Written by Arthur Michon.

The default operating point is the one of Karataev et al. [1]: 16 single-antenna access points (APs), 8 users, 4-QAM and 100 data symbols per block.
It differs from theirs in two ways: 8 orthogonal DFT pilots, which carry 8 times the energy of their pilots $X_p = I$ (available with `--pilots identity`, see below), and APs 10 m above the users.
So it is their setting, not a reproduction of their curves.

## Receivers

| name     | idea | reference |
|----------|------|-----------|
| `lmmse`  | Pilot-only MMSE channel estimate, local L-MMSE combining at each AP, large-scale fading decoding (LSFD) at the CPU. Not iterative. | Level 3 of Björnson and Sanguinetti [7] |
| `icd`    | Iterative channel estimation and detection (ICD): distributed EP detection with the channel fixed, then data-aided L-MMSE re-estimation with the symbols fixed, repeated. Each user keeps its own message variance. | He et al. [2, 3], with per-user variance |
| `he`     | The same ICD with the message variance averaged over users, as He et al. do. This is the ICD baseline Karataev et al. compare against. | He et al. [2, 3] |
| `jcd`    | Joint channel estimation and data detection (JCD) by bilinear EP on the product of channel and symbol, with categorical symbol messages. | Karataev et al. [1] |
| `bigamp` | Bilinear generalised AMP: Gaussian messages, with an Onsager correction in place of per-edge messages (a large-system approximation of belief propagation on the bilinear graph). | Parker et al. [4], applied to joint channel and data estimation by Wen et al. [6] |
| `bigabp` | Bilinear Gaussian belief propagation (BiGaBP): one Gaussian message per edge, each combination leaving the target edge out. Without the paper's belief scaling. | Ito et al. [5] |
| `genie`  | The EP detector of `icd` run on the true channel, with no estimation. | He et al. [2], the perfect-CSI baseline of [1] |

Every receiver splits between the APs and the CPU: each AP processes its own antennas (observation and channel), and the CPU combines the symbol information of all APs.
The iterative ones exchange messages over the fronthaul at every sweep, so their numbers of sweeps (see Operating point) are also their numbers of fronthaul round trips.

`icd` and `he` differ only in the variance averaging (three places in `detectors/icd.py`, marked `[avg_var]`).
Keeping them apart is the point of the comparison, since that one choice accounts for most of the published gap between ICD and JCD.
Unlike He et al., `he` is damped, with the damping shared by all the EP receivers.

`genie` is a reference, not a bound: EP is an approximate detector even with the true channel.

`bigabp_node` is an ablation of `bigabp`, run only when named (`python main.py bigabp_node`).
It keeps one belief per node instead of one message per edge, so each update re-uses its own information.

## Install

The code runs on the CPU and needs only PyTorch, NumPy and Matplotlib.
It was tested with Python 3.11, PyTorch 2.11, NumPy 2.4 and Matplotlib 3.10, and with Python 3.10, PyTorch 2.5 and NumPy 2.2.

```bash
pip install torch numpy matplotlib
```

All commands below are run from the root of the repository.

## Quick start

Run one receiver on a small budget (about 20 s for `jcd`):

```bash
python main.py jcd --drops 2 --reals 20
```

Positional arguments name the receivers to run, and no name runs all seven.
`--drops` sets the number of random user placements and `--reals` the number of fading realisations per placement.
`python main.py --help` lists the options.

A run with a smaller (or any non-reference) budget is saved in `results/quick/`, so it never replaces the committed results.

## Reproducing the results

```bash
python main.py
python make_figure.py
```

`main.py` runs every receiver on 50 user placements of 200 fading realisations each.
This takes about 3.5 hours on one CPU core: about 55 min for `jcd`, 50 min each for `icd` and `he`, 20 min for `bigabp`, 15 min each for `genie` and `bigamp`, and under a minute for `lmmse` (scaled from a one-placement run on a laptop).

`make_figure.py` reads `results/` (never `results/quick/`), prints a summary table and writes the two-panel figure to `figures/reproduction.pdf`.
It stops if the receivers were run at different operating points.
Panel (a) is the distribution of the per-user symbol error rate (SER), panel (b) the distribution of the per-link channel NMSE over the links whose received power exceeds the noise, $p\beta_{lk} > \sigma^2$ (the cut stated by Karataev et al.).

The committed results give:

| receiver | mean SER | median per-user SER | median link NMSE |
|----------|----------|---------------------|------------------|
| `lmmse`  | 1.0e-1   | 7.5e-2  | 1.2e-1 |
| `he`     | 3.0e-2   | 1.6e-2  | 2.0e-2 |
| `icd`    | 1.6e-3   | 1.3e-3  | 1.05e-2 |
| `bigabp` | 4.4e-3   | 2.7e-3  | 1.30e-2 |
| `bigamp` | 1.2e-3   | 8.8e-4  | 1.06e-2 |
| `jcd`    | 1.2e-3   | 1.0e-3  | 1.02e-2 |
| `genie`  | 1.2e-3   | 9.3e-4  | (true channel) |
| `bigabp_node` (ablation) | 2.2e-3 | 1.5e-3 | 1.05e-2 |

## Identity pilots

```bash
python main.py --pilots identity
python make_figure.py --pilots identity
```

The same receivers with the pilots of Karataev et al. [1], $X_p = I$: one slot per user, so each user's pilot carries the energy $p$ instead of $\tau_p p$ with the DFT pilots, 8 times less here.
Results go to `results/identity/` and the figure to `figures/reproduction_identity.pdf`.
The committed results give:

| receiver | mean SER | median per-user SER |
|----------|----------|---------------------|
| `lmmse`  | 1.4e-1   | 1.1e-1  |
| `he`     | 4.2e-2   | 2.4e-2  |
| `icd`    | 3.8e-3   | 3.1e-3  |
| `jcd`    | 6.2e-3   | 4.3e-3  |
| `bigabp` | 9.3e-3   | 5.5e-3  |
| `bigamp` | 6.5e-1 (diverges) | 6.6e-1 |
| `genie`  | 1.2e-3   | 9.3e-4  |

With these weaker pilots the per-user ICD detects better than the JCD, while the JCD is ahead with the DFT pilots.
`bigamp` diverges on every placement.
This implementation uses a fixed damping where Parker et al. [4] adapt it, and starts from the priors.
Started from the pilot MMSE channel estimate instead (an experiment not included here), it no longer diverges but stays far behind the other receivers.
So its identity-pilot result is a result of this implementation, not a verdict on BiG-AMP.

## Operating point

Set in `BASE` at the top of `main.py`.
Fields not named there take the defaults of `SimConfig` in `config.py`.

| quantity | value |
|----------|-------|
| APs | $L = 16$, one antenna each ($N_r = 1$), on a $4\times4$ grid with corners at the corners of a 400 m square, 10 m above the users |
| users | $K = 8$, dropped uniformly in the square, equal transmit power $p = 14$ dBm |
| large-scale fading | $\beta_{lk}$ [dB] $= -30.5 - 36.7\log_{10} d_{lk}$, no shadowing, $d_{lk}$ the 3-D distance in metres |
| small-scale fading | uncorrelated Rayleigh, $h \sim \mathcal{CN}(0, \beta_{lk})$ |
| noise | $-96$ dBm |
| frame | $\tau_p = 8$ orthogonal DFT pilots, then $\tau_d = 100$ data symbols, 4-QAM with unit average energy |
| iterations | `icd`/`he`: $R = 3$ detect and re-estimate passes of $I = 5$ EP sweeps, then a final detection. `genie`: 5 EP sweeps. `jcd`: $J = 10$ sweeps. `bigabp`: 15 sweeps. `bigamp`: 150 sweeps |
| damping | $\eta = 0.7$ for the EP receivers and `bigabp`, $0.15$ for `bigamp` (it diverges at 0.7) |
| Monte Carlo | 50 user placements, 200 fading realisations each |

The quantities meant to be varied are fields of `SimConfig`.
The ones kept fixed (grid layout, AP height, arithmetic precision) are module constants at the top of `config.py`, deliberately not parameters.

## Code layout

```
config.py                  SimConfig: one dataclass describes a run, derived quantities are properties
main.py                    runs the receivers at the operating point, saves results/<name>.npz
make_figure.py             results/ -> summary table + figures/reproduction.pdf
system/
  channel.py               geometry, large-scale fading, Rayleigh channel, Y = sqrt(p) H X + W
  transmitter.py           pilots and data symbols
detectors/
  base.py                  Observation (what a receiver sees), Detector interface, name registry
  pilot_estimation.py      MMSE channel estimate from orthogonal pilots
  lsfd.py                  local combining and LSFD fusion (Level 3 of [7]), used by lmmse
  lmmse.py                 lmmse
  icd.py                   icd, he, genie
  jcd.py                   jcd
  bigabp.py                bigabp, bigabp_node
  bigamp.py                bigamp
utils/
  run.py                   Monte-Carlo loop, metrics, saving and loading results
  signal_toolbox.py        Gray-labelled QAM/PSK constellations, mapping and hard decision
  figstyle.py              PDF output settings for the figures
results/                   one .npz per receiver at the reference budget, quick runs in results/quick/
figures/                   written by make_figure.py
```

## Conventions

- Tensor shapes are written in a comment at the end of the line: the channel `H` is `(batch, L, Nr, K)`, the received block `Y` is `(batch, L, Nr, tau)` and the transmitted block `X` is `(batch, K, tau)`.
- Comments refer to the equations of the papers by their own numbers, with the paper given by its number in the reference list below, or write the formula out where no paper equation exists.
- Following Karataev et al., $\sigma$ in a message denotes a variance, not a standard deviation. Many Gaussian messages are stored in natural parameters, precision $1/\sigma$ and natural mean $\rho = \mu/\sigma$, so that an uninformative message is zero.
- The EP receivers run in double precision as a safety margin. The received powers and the noise are both around $10^{-13}$ W, so the exchanged precisions are around $10^{13}$. With the negative-precision guard in place, single precision gave the same results in our tests.
- Every run is single-threaded (`torch.set_num_threads(1)` in `utils/run.py`). The ICD amplifies the last-digit differences of multithreaded summation into flipped symbols, so repeated runs would otherwise differ. Runs on different machines or library versions agree statistically but not to the last digit.

## Adding a receiver

1. Subclass `Detector` in a new file under `detectors/` and implement `detect(obs) -> (x_hat, h_hat)`, with `x_hat` of shape `(batch, K, tau_d)` and `h_hat` of shape `(batch, L, Nr, K)` or `None`.
2. Decorate the class with `@register("name")` and import the file in `detectors/__init__.py`.
3. Add `"name"` to `ALL` and `ITERS` in `main.py`, and to `CURVES` in `make_figure.py` to plot it.

The `Observation` passed to `detect` holds the received pilots and data, the pilot sequences, the constellation, the noise variance, the transmit power and the large-scale fading $\beta_{lk}$.
It also holds the true channel, which only `genie` may read.

## Results format

`results/<name>.npz` holds the configuration of the run (as JSON, key `config`), the aggregate `ser` and `nmse`, and the samples behind the figures: `ser_ue` (per-user SER, one value per user and placement), `nmse_link` (per-link NMSE, one value per AP, user and placement) and `beta_link` (the matching $\beta_{lk}$, for the weak-link cut).

## References

[1] A. Karataev, C. Forsch and L. Cottatellucci, "Bilinear expectation propagation for distributed semi-blind joint channel estimation and data detection in cell-free massive MIMO," *IEEE Open Journal of Signal Processing*, vol. 5, pp. 284-293, 2024, doi: 10.1109/OJSP.2023.3348343.

[2] H. He, H. Wang, X. Yu, J. Zhang, S. H. Song and K. B. Letaief, "Distributed expectation propagation detection for cell-free massive MIMO," *IEEE Global Communications Conference (GLOBECOM)*, 2021.

[3] H. He, X. Yu, J. Zhang, S. Song, R. D. Murch and K. B. Letaief, "Cell-free massive MIMO detection: a distributed expectation propagation approach," *IEEE Transactions on Mobile Computing*, vol. 24, no. 11, pp. 12149-12161, 2025, doi: 10.1109/TMC.2025.3583019.

[4] J. T. Parker, P. Schniter and V. Cevher, "Bilinear generalized approximate message passing, Part I: Derivation," *IEEE Transactions on Signal Processing*, vol. 62, no. 22, pp. 5839-5853, 2014, doi: 10.1109/TSP.2014.2357776.

[5] K. Ito, T. Takahashi, S. Ibi and S. Sampei, "Bilinear Gaussian belief propagation for massive MIMO detection with non-orthogonal pilots," *IEEE Transactions on Communications*, vol. 72, no. 2, pp. 1045-1061, 2024, doi: 10.1109/TCOMM.2023.3325479. An earlier conference version is "Bilinear Gaussian belief propagation for large MIMO channel and data estimation," *IEEE GLOBECOM*, 2020, doi: 10.1109/GLOBECOM42002.2020.9347952.

[6] C.-K. Wen, C.-J. Wang, S. Jin, K.-K. Wong and P. Ting, "Bayes-optimal joint channel-and-data estimation for massive MIMO with low-precision ADCs," *IEEE Transactions on Signal Processing*, vol. 64, no. 10, pp. 2541-2556, 2016, doi: 10.1109/TSP.2015.2508786.

[7] E. Björnson and L. Sanguinetti, "Making cell-free massive MIMO competitive with MMSE processing and centralized implementation," *IEEE Transactions on Wireless Communications*, vol. 19, no. 1, pp. 77-90, 2020, doi: 10.1109/TWC.2019.2941478.

## License

Not chosen yet.
