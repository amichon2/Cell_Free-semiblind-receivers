r"""
Iterative channel estimation and data detection (ICD) of He et al. [3], Sec. IV and
Algorithm 2.  Numbers in brackets refer to the reference list of README.md.

The model is linear in the channel given the symbols, and in the symbols given the
channel.  ICD alternates between the two: the distributed-EP detector detects the data
with the channel estimate held fixed, then a linear MMSE estimator re-estimates the
channel with the detected symbols used as extra pilots.  Besides the point estimates,
the two modules exchange two second-order corrections, once per outer pass: the channel
error variance enters the detector noise (V_det here, [3] eq. (26)), and the symbol
variance enters the estimator noise (V_est here, [3] eq. (27)).  He et al. name the two
the other way round, after the error each one carries: their V_est is our V_det.

    EP detection, channel fixed       ->  _ep_detect   (inner loop of I sweeps)
    data-aided re-estimation          ->  _reestimate  (outer loop of R passes)

The fork between icd and he.  Module A, the AP-local L-MMSE, takes the incoming symbol
messages as its prior, and there is a choice of how much of them to keep:

    avg_var = False  (icd)  keep the per-user variance, S_{l,t} = diag(sigma_{k->l,t})
    avg_var = True   (he)   average it over the users, prior lambda_l I

The second is the scalar-variance approximation, which makes the scheme behave like
vector AMP rather than full EP, and it is the ICD baseline Karataev et al. compare
against.  It discards the per-user information, so the per-user form is the stronger
baseline.  Most of the reported JCD-over-ICD gap comes from this choice rather than from
the bilinear coupling (see the results in README.md).  The two receivers differ only at
the three places marked [avg_var] below: the incoming and the posterior variance of
Module A, and the posterior variance in the return message, which He et al. average over
users before dividing out the AP's extrinsic ([3] eqs. (20)-(21)).
Everything else is shared, which is the point.
"""
from __future__ import annotations

import torch

from config import VAR_MIN
from detectors.base import Detector, Observation, register, to_double
from detectors.pilot_estimation import pilot_mmse
from utils.signal_toolbox import hard_demap

VAR_MAX = 1.0 / VAR_MIN                 # variance ceiling of the symbol-domain messages


def _cavity(mu_post, var_post, mu_in, var_in):
    """EP extrinsic: divide the posterior by the incoming message, in natural parameters.
    Returns the extrinsic (mean, variance).

    The difference of precisions can come out non-positive, which no Gaussian can carry,
    and the message is then made uninformative: its precision is floored at 1/VAR_MAX and
    its natural mean set to zero.  Flooring the precision alone would leave an improper
    message, whose implied mean rho/prec ~ rho*VAR_MAX blows up and corrupts whatever
    uses it.
    """
    raw = 1.0 / var_post - 1.0 / var_in                  # extrinsic precision (may be <= 0)
    floored = raw <= 1.0 / VAR_MAX                       # non-positive: make uninformative
    var_e = (1.0 / raw.clamp_min(1.0 / VAR_MAX)).clamp(VAR_MIN, VAR_MAX)
    rho_e = mu_post / var_post - mu_in / var_in          # natural mean of the extrinsic
    rho_e = torch.where(floored, torch.zeros_like(rho_e), rho_e)   # zero it with the precision
    return var_e * rho_e, var_e


@register("icd")
class ICDDetector(Detector):
    """ICD of He et al. with the per-user message variance (full EP)."""

    avg_var = False                     # see the module docstring

    def __init__(self, config):
        self.cfg = config
        self.n_iters = config.n_iters   # inner EP detection sweeps, I
        self.n_outer = config.n_outer   # outer detect <-> re-estimate passes, R
        self.eta = config.damping       # damping: new = (1 - eta) old + eta update, eta = 1 undamped

    # ---------------------------------------------------------------- detection
    def _ep_detect(self, h_hat, c, obs: Observation):
        """EP detection with the channel estimate h_hat (B, L, Nr, K) held fixed, and its
        per-(l,k) error variance c (B, L, K).  Returns the posterior mean and variance of
        the symbols, xhat and v, each (B, K, tau_d)."""
        p, N0 = obs.p_tx, obs.N0
        y = obs.y_data                                       # (B, L, Nr, tau_d)
        B, L, Nr, Td = y.shape
        K = h_hat.shape[-1]
        Ex = (obs.constellation.abs() ** 2).mean().item()    # = 1 for unit-energy symbols
        eta = self.eta

        G = (p ** 0.5) * h_hat                               # effective channel  (B, L, Nr, K)
        GhG = torch.einsum("blrk,blrj->blkj", G.conj(), G)   # (B, L, K, K)
        Ghy = torch.einsum("blrk,blrt->blkt", G.conj(), y)   # (B, L, K, tau_d)

        sig2 = N0 + p * c.sum(dim=-1)                        # (B, L) detector noise V_det, [3] eq. (26)
        inv_sig2 = (1.0 / sig2)[..., None, None]             # (B, L, 1, 1)
        M0 = inv_sig2 * GhG                                  # (B, L, K, K)
        Ghy_s = inv_sig2 * Ghy                               # (B, L, K, tau_d)

        # symbol -> factor messages, initialised to the moments of the uniform prior
        mu_s = torch.zeros(B, L, K, Td, dtype=G.dtype, device=G.device)          # (B, L, K, tau_d)
        var_s = torch.full((B, L, K, Td), Ex, dtype=G.real.dtype, device=G.device)
        xhat = torch.zeros(B, K, Td, dtype=G.dtype, device=G.device)             # (B, K, tau_d)
        v = torch.full((B, K, Td), Ex, dtype=G.real.dtype, device=G.device)      # (B, K, tau_d)

        for _ in range(self.n_iters):
            var_in = (var_s.mean(dim=2, keepdim=True).expand_as(var_s)   # [avg_var]
                      if self.avg_var else var_s)            # (B, L, K, tau_d)
            prec_s = 1.0 / var_in                            # (B, L, K, tau_d)

            # --- Module A: AP-local L-MMSE posterior over x_{:,t} ---
            A = M0.unsqueeze(2) + torch.diag_embed(prec_s.transpose(-1, -2))   # (B, L, tau_d, K, K)
            Sigma = torch.linalg.inv(A)                      # (B, L, tau_d, K, K)
            rhs = (Ghy_s + prec_s * mu_s).transpose(-1, -2)  # (B, L, tau_d, K)
            mu_post = torch.einsum("bltij,bltj->blti", Sigma, rhs)          # (B, L, tau_d, K)
            var_post = torch.diagonal(Sigma, dim1=-2, dim2=-1).real.clamp_min(VAR_MIN)
            mu_post = mu_post.transpose(-1, -2)              # (B, L, K, tau_d)
            var_post = var_post.transpose(-1, -2)            # (B, L, K, tau_d)
            if self.avg_var:                                 # he: the extrinsic stays scalar too
                var_post = var_post.mean(dim=2, keepdim=True).expand_as(var_post)   # [avg_var]

            # --- extrinsic, factor -> variable (l -> k) ---
            mu_lk, var_lk = _cavity(mu_post, var_post, mu_s, var_in)   # (B, L, K, tau_d)

            # --- Module B: CPU combining over all APs (precision-weighted) ---
            prec_k = (1.0 / var_lk).sum(dim=1)               # (B, K, tau_d)
            rho_k = (mu_lk / var_lk).sum(dim=1)              # (B, K, tau_d)
            var_k = (1.0 / prec_k).clamp(VAR_MIN, VAR_MAX)   # (B, K, tau_d)
            mu_k = var_k * rho_k                             # (B, K, tau_d)

            # --- constellation projection against that scalar AWGN channel ---
            xhat, v = self._project(mu_k, var_k, obs.constellation)

            # --- return message, variable -> factor (k -> l), damped in natural parameters.
            #     he: average the posterior variance over users first, then divide out the AP's
            #     extrinsic, as He et al. eqs. (14)-(15) ---
            v_ret = v.mean(dim=1, keepdim=True).expand_as(v) if self.avg_var else v   # [avg_var]
            mu_kl, var_kl = _cavity(xhat.unsqueeze(1), v_ret.unsqueeze(1), mu_lk, var_lk)
            prec_d = (1 - eta) / var_s + eta / var_kl        # (B, L, K, tau_d)
            rho_d = (1 - eta) * mu_s / var_s + eta * mu_kl / var_kl
            var_s = (1.0 / prec_d).clamp(VAR_MIN, VAR_MAX)   # (B, L, K, tau_d)
            mu_s = rho_d * var_s                             # (B, L, K, tau_d)
        return xhat, v

    def _project(self, mu, var, constellation):
        """Posterior mean and variance of x under the uniform constellation prior, given the
        scalar AWGN observation mu = x + noise of variance var.  mu and var are (B, K, tau_d),
        and so are the returned xhat and v."""
        d = constellation                                    # (M,)
        logw = -(mu.unsqueeze(-1) - d).abs() ** 2 / var.unsqueeze(-1)   # (B, K, tau_d, M)
        logw = logw - logw.max(dim=-1, keepdim=True).values
        w = torch.softmax(logw, dim=-1)                      # the uniform prior is a constant and drops out
        xhat = (w * d).sum(dim=-1)                           # (B, K, tau_d)
        e2 = (w * d.abs() ** 2).sum(dim=-1)                  # (B, K, tau_d)
        return xhat, (e2 - xhat.abs() ** 2).real.clamp_min(VAR_MIN)

    # --------------------------------------------------------- re-estimation
    def _reestimate(self, obs: Observation, xhat, v, h_hat):
        """Data-aided joint L-MMSE on the augmented block [Y^p Y^d] ([3] eq. (28)).

        The detected symbols are not orthogonal, which is why this is a joint solve over
        all K users rather than the per-user scalar estimator of pilot_estimation.py.  Returns the
        new estimate (B, L, Nr, K) and its per-(l,k) error variance (B, L, K).
        """
        p, N0 = obs.p_tx, obs.N0
        pilots = obs.pilots                                  # (K, tau_p)
        B, L, Nr, _ = h_hat.shape
        K, tp = pilots.shape

        X = torch.cat([pilots.unsqueeze(0).expand(B, -1, -1), xhat], dim=-1)  # (B, K, tau_c)
        Y = torch.cat([obs.y_pilot, obs.y_data], dim=-1)     # (B, L, Nr, tau_c)

        # Per-use noise V_est: N0 on the pilots, N0 + p sum_k |hhat_lk|^2 v_kt on the data, the
        # covariance of the equivalent noise of [3] eq. (27).  Two differences from [3]: the
        # symbol error is weighted by the channel estimate |hhat_lk|^2 where He et al. use the
        # channel statistics (measured immaterial here), and the diagonal is averaged over the
        # Nr antennas of the AP, exact for Nr = 1 and dropping off-diagonal terms for Nr > 1.
        ehat = (h_hat.abs() ** 2).mean(dim=2)                # (B, L, K)
        D = Y.new_full((B, L, X.shape[-1]), N0, dtype=Y.real.dtype)           # (B, L, tau_c)
        D[..., tp:] = N0 + p * torch.einsum("blk,bkt->blt", ehat, v)
        Dinv = (1.0 / D).to(X.dtype)                         # (B, L, tau_c)

        # Lambda_l = diag(1/beta) + p X^* Dinv X^T   (K x K, shared across the Nr antennas)
        XDX = torch.einsum("bkt,blt,bjt->blkj", X.conj(), Dinv, X)           # (B, L, K, K)
        prior = torch.diag_embed((1.0 / obs.beta).to(X.dtype)).unsqueeze(0)  # (1, L, K, K)
        Linv = torch.linalg.inv(p * XDX + prior)             # (B, L, K, K)

        rhs = (p ** 0.5) * torch.einsum("bkt,blt,blrt->blrk", X.conj(), Dinv, Y)
        h_new = torch.einsum("blkj,blrj->blrk", Linv, rhs)   # (B, L, Nr, K)
        # a channel-domain variance (of the order of beta), so not floored at the symbol-domain VAR_MIN
        c_new = torch.diagonal(Linv, dim1=-2, dim2=-1).real.clamp_min(0.0)   # (B, L, K)
        return h_new, c_new

    # ------------------------------------------------------------------ driver
    def detect(self, obs: Observation):
        if obs.y_pilot is None or obs.pilots is None:
            raise ValueError("ICD needs pilots (tau_p > 0), it cannot run blind")
        in_cdtype = obs.y_data.dtype                         # returned in the caller's dtype
        obs = to_double(obs)

        h_hat, c = pilot_mmse(obs.y_pilot, obs.pilots, obs.beta, obs.p_tx, obs.N0)
        c = c.unsqueeze(0).expand(h_hat.shape[0], -1, -1)    # (L, K) -> (B, L, K)

        for _ in range(self.n_outer):
            xhat, v = self._ep_detect(h_hat, c, obs)
            h_hat, c = self._reestimate(obs, xhat, v, h_hat)
        xhat, _ = self._ep_detect(h_hat, c, obs)             # final detection

        idx = hard_demap(xhat, obs.constellation)            # (B, K, tau_d)
        return obs.constellation[idx].to(in_cdtype), h_hat.to(in_cdtype)


@register("he")
class HeDetector(ICDDetector):
    """ICD with the user-averaged (scalar) Module-A variance of He et al.

    Identical to ICDDetector except at the three places marked [avg_var] in _ep_detect.
    He et al. do not damp, here eta is the shared EP damping (undamped, the published
    receiver reaches only SER ~0.14 at the reference point).
    """
    avg_var = True


@register("genie")
class GenieEPDetector(ICDDetector):
    """The EP *detector* run on the true channel: perfect CSI, no pilots, no
    re-estimation (c = 0).  This is the perfect-CSI reference.  Unlike a *linear*
    perfect-CSI receiver it is the right reference for the EP detectors, which
    exploit the discrete alphabet.  It is not a bound: EP is approximate even with
    the true channel, and the JCD lands on it within the Monte-Carlo spread."""

    def detect(self, obs: Observation):
        if obs.H_true is None:
            raise ValueError("the genie needs the true channel (perfect CSI)")
        in_cdtype = obs.y_data.dtype
        obs = to_double(obs)
        H = obs.H_true                                       # (B, L, Nr, K)
        B, L, Nr, K = H.shape
        c = torch.zeros(B, L, K, dtype=torch.float64, device=H.device)   # no channel error
        xhat, _ = self._ep_detect(H, c, obs)
        idx = hard_demap(xhat, obs.constellation)
        return obs.constellation[idx].to(in_cdtype), H.to(in_cdtype)
