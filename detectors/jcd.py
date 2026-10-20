r"""
Bilinear joint channel estimation and data detection (JCD) of Karataev et al. [1], Sec. IV.
Numbers in brackets refer to the reference list of README.md.

The graph is built on the auxiliary variable z_{(l,r),k,t} = sqrt(p) x_{k,t} h_{(l,r),k}
([1] eq. (12), with the transmit power folded in).  The observation is linear in z, and
all the bilinearity sits in one coupling factor.  The graph has four kinds of factor:

    f_{(l,r),t} = CN(y ; sum_k z, N0)                observation likelihood
    f_Z         = delta(z - sqrt(p) x h)             bilinear coupling factor
    f_H         = the pilot-MMSE posterior           standing channel prior, [1] Sec. IV-A
    f_X         = uniform over the constellation     data prior

The symbol stays discrete.  Given a constellation point s the coupling factor is
linear, so the beliefs around it are Gaussian mixtures with one component per point.
The message to the symbol is categorical ([1] eq. (24)) and the decision is MAP.
The two Gaussian messages, forward to the observation and backward to the channel, are
standard EP updates: match the moments of the mixture marginal, then divide out the
incoming message.  Once a symbol belief has settled on one point s the mixture has a
single component, and the two updates reduce to the linear ones for that point: the
forward message is CN(z; a s mu_h, |a s|^2 var_h) and the backward one is
CN(g; mu_O / (a s), var_O / |a s|^2), with (mu_h, var_h) the incoming channel message,
(mu_O, var_O) the observation message and a, g as in the conventions below.

One sweep runs three blocks in order, each using the messages the previous block has
just updated:

    A. forward     f_Z -> z     [1] eqs. (17)-(23)
    B. observation y -> z       [1] Sec. IV-C, update of m_{Psi_0 -> z} (unnumbered)
    C. backward    f_Z -> h     [1] Sec. IV-C, update of m_{Psi_1 -> h} (unnumbered),
                   f_Z -> x     [1] eq. (24)

Within a block the updates are independent across ((l,r), k, t), so each block is a
few tensor operations.

Conventions.  Antennas are treated as independent scalars (uncorrelated Rayleigh gives
the per-user covariance beta_lk I), so every message is a scalar per antenna.  This is
Karataev et al.'s scheme exactly for Nr = 1, the case run here.  For Nr > 1 their
messages are Nr-dimensional with full covariance: the forward covariance of z = x h
contains Var(x) mu_h mu_h^H, which correlates the antennas of an AP, and the per-antenna
treatment drops it.  The channel is normalised to unit prior variance, g = h/sqrt(beta), with
gain a = sqrt(p beta), so that z = a g x.  The estimate is returned as
h_hat = g_hat sqrt(beta).

Gaussian messages are stored in natural parameters, precision 1/sigma and natural mean
rho = mu/sigma (sigma is a variance, as in Karataev et al.), so that an uninformative
message is zero.  This is what makes the negative-precision guard expressible.  An EP
division can return a non-positive precision, and the guard then makes the message
uninformative by zeroing rho together with the precision.  Zeroing the precision alone
would leave an improper message: its implied mean rho/prec blows up, is subtracted into
the observation cavity (block B) and corrupts every other user at that AP.  Karataev et
al. set the negative eigenvalues to zero ([1] Sec. IV-E) and say nothing of the mean.
Zeroing rho with them is the completion that keeps the message proper.
"""
from __future__ import annotations

import torch

from config import VAR_MIN
from detectors.base import Detector, Observation, register
from detectors.pilot_estimation import pilot_mmse

VAR_MAX = 1e12                          # variance ceiling
PINV = 1.0 / VAR_MAX                    # precision floor, what counts as zero precision


@register("jcd")
class JCDDetector(Detector):
    """Bilinear-EP JCD of Karataev et al., with categorical symbol messages."""

    def __init__(self, config):
        self.cfg = config
        self.n_iters = config.n_iters   # joint sweeps J: one channel + one symbol update each
        self.eta = config.damping       # damping: new = (1 - eta) old + eta update, eta = 1 undamped

    def _moment_match(self, prec_ytoz, rho_ytoz, prec_htoz, rho_htoz, L_xtoz, a, cstv, s2v, N0):
        """Moments of the z and g marginals at the coupling factor, over the constellation.

        Inputs are the observation message (prec_ytoz, rho_ytoz) of block B, the channel
        extrinsic (prec_htoz, rho_htoz) of [1] eqs. (15)-(16), the symbol extrinsic L_xtoz
        (log-weights over the constellation, [1] eqs. (26)-(27)), the gain a = sqrt(p beta),
        and the constellation cstv with its energies s2v.  Returns
            (zhat, Zz)   mean and variance of the z marginal, for the forward message (A)
            (hhat, Zh)   mean and variance of the g marginal, for the message to g (C)
            ev           log-evidence of each point s, the message m_{Z->x}(s) of [1] eq. (24)
        All broadcast over (B, N, K, Td), with a trailing constellation axis of size S.
        """
        var_htoz = 1.0 / prec_htoz                              # (B, N, K, Td)
        mu_htoz = rho_htoz * var_htoz                           # (B, N, K, Td)
        asv = a.unsqueeze(-1) * cstv                            # a*s      (1, N, K, 1, S)
        as2 = (a.unsqueeze(-1) ** 2) * s2v                      # |a s|^2  (1, N, K, 1, S)
        mzs = asv * mu_htoz.unsqueeze(-1)                       # per-symbol z mean  (B, N, K, Td, S)
        vzs = as2 * var_htoz.unsqueeze(-1)                      # per-symbol z var   (B, N, K, Td, S)
        sO = 1.0 / prec_ytoz.clamp_min(PINV)                    # observation variance  (B, N, K, Td)
        mO = rho_ytoz * sO                                      # observation mean      (B, N, K, Td)

        tot = sO.unsqueeze(-1) + vzs                            # (B, N, K, Td, S)
        ev = -((mO.unsqueeze(-1) - mzs).abs() ** 2) / tot - torch.log(tot)   # log-evidence of s, [1] eq. (24)
        logw = L_xtoz + ev                                      # (B, N, K, Td, S)
        logw = logw - logw.max(dim=-1, keepdim=True).values
        w = torch.softmax(logw, dim=-1)                         # mixture weights (B, N, K, Td, S)

        precZs = prec_ytoz.unsqueeze(-1) + 1.0 / vzs            # z marginal given s  (B, N, K, Td, S)
        mZs = (rho_ytoz.unsqueeze(-1) + mzs / vzs) / precZs     # (B, N, K, Td, S)
        vZs = 1.0 / precZs                                      # (B, N, K, Td, S)
        precHs = prec_htoz.unsqueeze(-1) + as2 * prec_ytoz.unsqueeze(-1)     # g marginal given s
        mHs = (rho_htoz.unsqueeze(-1) + asv.conj() * rho_ytoz.unsqueeze(-1)) / precHs
        vHs = 1.0 / precHs                                      # (B, N, K, Td, S)

        zhat = (w * mZs).sum(-1)                                # (B, N, K, Td)
        Zz = ((w * (mZs.abs() ** 2 + vZs)).sum(-1) - zhat.abs() ** 2).clamp(VAR_MIN * N0, VAR_MAX)
        hhat = (w * mHs).sum(-1)                                # (B, N, K, Td)
        Zh = ((w * (mHs.abs() ** 2 + vHs)).sum(-1) - hhat.abs() ** 2).clamp(VAR_MIN, VAR_MAX)
        return zhat, Zz, hhat, Zh, ev

    @staticmethod
    def _guard(raw, rho):
        """Negative-precision guard ([1] Sec. IV-E, completed).  Where the precision raw is negative,
        zero the natural mean too, so that the message is uninformative and not improper.
        The precision is floored at PINV everywhere."""
        neg = raw < 0
        return raw.clamp_min(PINV), torch.where(neg, torch.zeros_like(rho), rho)

    def detect(self, obs: Observation):
        if obs.y_pilot is None or obs.pilots is None:
            raise ValueError("bilinear-EP JCD needs the pilot MMSE prior (tau_p > 0)")
        p, N0 = obs.p_tx, obs.N0
        dev = obs.y_data.device
        B, L, Nr, Td = obs.y_data.shape
        K = obs.beta.shape[1]
        N = L * Nr
        rdtype, cdtype = torch.float64, torch.complex128       # see detectors/base.to_double
        in_cdtype = obs.y_data.dtype                           # returned in the caller's dtype
        cst = obs.constellation.to(cdtype)                     # (S,)
        S = cst.numel()
        eta = self.eta

        y = obs.y_data.reshape(B, N, Td).to(cdtype)            # (B, N, Td)
        beta = obs.beta.repeat_interleave(Nr, dim=0).to(dev).to(rdtype)   # (N, K)
        a = (p * beta).sqrt().unsqueeze(0).unsqueeze(-1)       # sqrt(p beta)  (1, N, K, 1)
        cstv = cst.view(1, 1, 1, 1, S)                         # (1, 1, 1, 1, S)
        s2v = (cst.abs() ** 2).view(1, 1, 1, 1, S)             # (1, 1, 1, 1, S)

        # Standing channel prior, the pilot-MMSE posterior on g = h/sqrt(beta) ([1] Sec. IV-A).
        # It enters again at every sweep and is never re-estimated from detected symbols,
        # which is why the per-user scalar estimator suffices here (ICD needs a joint one).
        h0, c0 = pilot_mmse(obs.y_pilot.to(cdtype), obs.pilots.to(cdtype),
                            obs.beta.to(rdtype), p, N0)
        mu_h0 = h0.reshape(B, N, K) / beta.sqrt()              # (B, N, K)
        var_h0 = (c0.repeat_interleave(Nr, dim=0) / beta).unsqueeze(0).expand(B, N, K)
        var_h0 = var_h0.clamp(VAR_MIN, VAR_MAX)                # (B, N, K)
        prec_h0 = 1.0 / var_h0                                 # (B, N, K)
        rho_h0 = mu_h0 / var_h0                                # (B, N, K)

        # persistent factor -> variable messages, initialised uninformative (zero)
        prec_ztoy = torch.zeros(B, N, K, Td, dtype=rdtype, device=dev)      # (B, N, K, Td)
        rho_ztoy = torch.zeros(B, N, K, Td, dtype=cdtype, device=dev)       # (B, N, K, Td)
        prec_ztoh = torch.zeros(B, N, K, Td, dtype=rdtype, device=dev)      # (B, N, K, Td)
        rho_ztoh = torch.zeros(B, N, K, Td, dtype=cdtype, device=dev)       # (B, N, K, Td)
        prec_ytoz = torch.zeros(B, N, K, Td, dtype=rdtype, device=dev)      # (B, N, K, Td)
        rho_ytoz = torch.zeros(B, N, K, Td, dtype=cdtype, device=dev)       # (B, N, K, Td)
        m_ztox = torch.zeros(B, N, K, Td, S, dtype=rdtype, device=dev)      # log message f_Z -> x

        for _ in range(self.n_iters):
            # channel node: posterior ([1] Sec. IV-D), and the extrinsic without use t
            prec_h = prec_h0.unsqueeze(-1) + prec_ztoh.sum(dim=3, keepdim=True)   # (B, N, K, 1)
            rho_h = rho_h0.unsqueeze(-1) + rho_ztoh.sum(dim=3, keepdim=True)      # (B, N, K, 1)
            prec_htoz = (prec_h - prec_ztoh).clamp_min(PINV)   # (B, N, K, Td)  [1] eq. (15)
            rho_htoz = rho_h - rho_ztoh                        # (B, N, K, Td)
            # symbol node: the extrinsic is the product of the other antennas' categorical
            # messages, a sum in the log domain
            L_xtoz = m_ztox.sum(dim=1, keepdim=True) - m_ztox  # (B, N, K, Td, S)  [1] eq. (26)
            L_xtoz = L_xtoz - L_xtoz.max(dim=-1, keepdim=True).values

            # --- A. forward f_Z -> z: z marginal against the previous sweep's observation
            #     message, which is then divided out ---
            zhat, Zz, _, _, _ = self._moment_match(prec_ytoz, rho_ytoz, prec_htoz, rho_htoz,
                                                   L_xtoz, a, cstv, s2v, N0)
            raw = 1.0 / Zz - prec_ytoz                         # extrinsic precision (may be < 0)
            rho_n = zhat / Zz - rho_ytoz                       # (B, N, K, Td)
            prec_n, rho_n = self._guard(raw, rho_n)
            prec_ztoy = (1 - eta) * prec_ztoy + eta * prec_n   # damped, in natural parameters
            rho_ztoy = (1 - eta) * rho_ztoy + eta * rho_n

            # --- B. observation message ([1] Sec. IV-C), rebuilt from the forwards just updated.
            #     The other users enter by subtracting their means and adding their variances,
            #     each with its own variance, never collapsed to one scalar ---
            on = prec_ztoy > 0                                 # informative forwards  (B, N, K, Td)
            var_ztoy = 1.0 / prec_ztoy.clamp_min(PINV)         # (B, N, K, Td)
            mu_ztoy = torch.where(on, rho_ztoy * var_ztoy, torch.zeros_like(rho_ztoy))
            sumv = N0 + (on * var_ztoy).sum(dim=2, keepdim=True)   # over users  (B, N, 1, Td)
            summ = (on * mu_ztoy).sum(dim=2, keepdim=True)         # (B, N, 1, Td)
            sO = (sumv - on * var_ztoy).clamp_min(N0 * 1e-6)       # leave user k out  (B,N,K,Td)
            mO = y.unsqueeze(2) - (summ - on * mu_ztoy)            # (B, N, K, Td)
            prec_ytoz = (1 - eta) * prec_ytoz + eta * (1.0 / sO)
            rho_ytoz = (1 - eta) * rho_ytoz + eta * (mO / sO)

            # --- C. backward f_Z -> h ([1] Sec. IV-C) and f_Z -> x ([1] eq. (24)) ---
            _, _, hhat, Zh, ev = self._moment_match(prec_ytoz, rho_ytoz, prec_htoz, rho_htoz,
                                                    L_xtoz, a, cstv, s2v, N0)
            raw_h = 1.0 / Zh - prec_htoz                       # extrinsic precision (may be < 0)
            rho_hn = hhat / Zh - rho_htoz                      # (B, N, K, Td)
            prec_hn, rho_hn = self._guard(raw_h, rho_hn)
            prec_ztoh = (1 - eta) * prec_ztoh + eta * prec_hn
            rho_ztoh = (1 - eta) * rho_ztoh + eta * rho_hn
            m_ztox = (1 - eta) * m_ztox + eta * ev           # damped in the log domain

        belief = m_ztox.sum(dim=1)                             # combine over antennas (B, K, Td, S)
        x_hat = cst[belief.argmax(dim=-1)]                     # MAP decision  (B, K, Td)
        prec_h = prec_h0 + prec_ztoh.sum(dim=3)                # channel posterior, [1] Sec. IV-D
        mu_h = (rho_h0 + rho_ztoh.sum(dim=3)) / prec_h         # posterior mean on g  (B, N, K)
        h_hat = (mu_h * beta.sqrt()).reshape(B, L, Nr, K)      # de-normalise g -> h
        return x_hat.to(in_cdtype), h_hat.to(in_cdtype)
