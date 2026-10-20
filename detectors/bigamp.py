r"""
Bilinear generalised AMP (BiG-AMP) of Parker, Schniter and Cevher [4], Table III, with an
AWGN output.  Numbers in brackets refer to the reference list of README.md.

The symbols follow their Table III, with the step that computes each one:

    a_{(l,r),k}, nu^a   effective channel sqrt(p) h and its variance   (R15-R16)
    x_{k,t},     v      symbol mean and variance                       (R13-R14)
    nubar_p             sum_k (T1 + T2), which scales the Onsager term (R1)
    m_{(l,r),t}         Onsager-corrected prediction, their p-hat      (R2, R4)
    nu_p                nubar_p + sum_k T3 = sum_k (T1+T2+T3)          (R3)
    shat, nu_s          scaled residual and inverse residual variance  (R5-R8)
    rhat, nu_r          pseudo-measurement of a symbol                 (R9-R10)
    qhat, nu_q          pseudo-measurement of a channel coefficient    (R11-R12)

with T1 = nu^a |x|^2, T2 = |a|^2 v and T3 = nu^a v for each user.  For an AWGN output,
R5-R8 reduce to nu_s = 1/(nu_p + N0) and shat = (y - m)/(nu_p + N0).

The whole coherence block is processed at once, pilots included.  A pilot is a
symbol whose prior is a point mass instead of the uniform constellation prior, and
that is the entire semi-blind machinery here.  With the zero
initialisation below, the first sweep correlates the received pilots with each
user's pilot, as the pilot MMSE estimator does, but it is not that estimator.  The
scalar residual variance charges the estimate with the other users' received power,
sum_k p beta_k + N0 instead of N0, because nothing in AMP uses the orthogonality of
the pilots.  The first-sweep estimate is therefore shrunk harder than the pilot MMSE
one (a median of 0.57 times it per link, on one drop at the reference point).

Parker et al. derive BiG-AMP from loopy belief propagation on the bilinear graph by
large-system approximations ([4] Sec. II).  The observation node is a linear sum over
the K users, and the leave-one-out expansion of that sum drops the O(1/K) correction in
the cavity variance (so nu_p and nu_s carry no user index) while keeping the O(1/sqrt K)
correction in the cavity mean, which is the Onsager term in m.  Next to jcd, the messages
are Gaussian: the categorical symbol message is collapsed to its two moments, the
discrete prior being applied once per (k,t) at the denoiser.

The pseudo-measurements are stored in natural parameters (precision and natural
mean).  At the first sweep ahat = 0 makes nu_r infinite in R9, and on a pilot
column the message would then have zero precision and a non-zero natural mean, an
improper message (see the guard in jcd.py).  In natural parameters both are zero, and
both denoisers reduce to adding precisions.

The receiver splits between the APs and the CPU as jcd does.  R1-R8 and the channel half
(R11-R12, R15-R16) use only the antenna's own observation and the current symbol
estimates, so they run at the AP.  R9-R10 sum per-antenna terms over the N = L*Nr
antennas, which the APs send to the CPU, and the symbol denoiser (R13-R14) runs there,
as the symbol node of jcd does.  Each sweep is one fronthaul round trip, so the 150
sweeps of main.py cost 150 round trips where jcd needs 10.

Damping is part of the algorithm, not a numerical afterthought ([4] Sec. IV), and
BiG-AMP needs far more of it than the EP receivers.  At the eta = 0.7 they share, it
diverges outright, so main.py gives it its own value.  The damping is that of Parker et
al. (Part I, Sec. IV-A, eqs. 93-98): the steps computing nubar_p, nu_p, nu_s and shat
(their R1, R3, R7, R8) are damped, and damped copies xbar, abar of the estimates replace
them in the pseudo-measurements (their R9-R12) but not in the prediction (R1-R2).  Two
differences: the damping factor is fixed (their adaptive rule, Sec. IV-B, is not
implemented), and nubar_p, nu_p, nu_s and shat are undamped on the first sweep, where
they have no previous value.
"""
from __future__ import annotations

import torch

from detectors.base import Detector, Observation, register


@register("bigamp")
class BiGAMPDetector(Detector):
    """BiG-AMP of Parker et al. [4], Table III, with an AWGN output."""

    def __init__(self, config):
        self.cfg = config
        self.n_iters = config.n_iters       # sweeps I_A, which play the role of the JCD's J
        self.eta = config.damping           # damping: new = (1 - eta) old + eta update

    def detect(self, obs: Observation):
        if obs.y_pilot is None or obs.pilots is None:
            raise ValueError("BiG-AMP needs pilots to break the scale ambiguity (tau_p > 0)")
        p, N0 = obs.p_tx, obs.N0
        dev = obs.y_data.device
        B, L, Nr, Td = obs.y_data.shape
        K = obs.beta.shape[1]
        N, Tp = L * Nr, obs.pilots.shape[1]
        Tc = Tp + Td
        rdtype, cdtype = torch.float64, torch.complex128       # see detectors/base.to_double
        in_cdtype = obs.y_data.dtype                           # answer in the caller's dtype
        cst = obs.constellation.to(cdtype)                     # (S,)
        s2 = (cst.abs() ** 2).to(rdtype)                       # |s|^2  (S,)
        eta = self.eta

        # the whole block at once, pilots first
        y = torch.cat([obs.y_pilot, obs.y_data], dim=-1).reshape(B, N, Tc).to(cdtype)  # (B, N, Tc)
        beta = obs.beta.repeat_interleave(Nr, dim=0).to(dev).to(rdtype)                # (N, K)
        tau_a = (p * beta).unsqueeze(0).expand(B, N, K).contiguous()   # prior var of a  (B, N, K)
        pilots = obs.pilots.to(cdtype).unsqueeze(0).expand(B, K, Tp)   # (B, K, Tp)

        # --- initialisation from the priors, with shat = 0 so that sweep 1 has no Onsager term ---
        shat = torch.zeros(B, N, Tc, dtype=cdtype, device=dev)         # (B, N, Tc)
        ahat = torch.zeros(B, N, K, dtype=cdtype, device=dev)          # (B, N, K)  E{a} = 0
        nu_a = tau_a.clone()                                           # (B, N, K)  p beta
        xhat = torch.zeros(B, K, Tc, dtype=cdtype, device=dev)         # (B, K, Tc)
        v_x = torch.full((B, K, Tc), float(s2.mean()), dtype=rdtype, device=dev)   # (B, K, Tc) E_x
        xhat[:, :, :Tp] = pilots                     # point-mass prior on the pilot columns
        v_x[:, :, :Tp] = 0.0
        abar, xbar = ahat, xhat                      # damped copies used in R9-R12
        nubar_p = nu_p = nu_s = None                 # damped from sweep 2 on
        w = None                                     # last constellation posterior weights

        for i in range(self.n_iters):
            first = (i == 0)
            dmp = lambda new, old: new if first else eta * new + (1 - eta) * old

            # --- prediction and Onsager correction (R1-R4) ---
            # nubar_p collects T1 + T2, the two cross terms of the product expansion, and
            # scales the Onsager correction.  nu_p adds T3 = nu^a v, which is second order in
            # that expansion and therefore reaches the algorithm only as a variance.
            nubar_p = dmp(torch.bmm(ahat.abs() ** 2, v_x)                    # sum_k T2 = |a|^2 v
                          + torch.bmm(nu_a, xhat.abs() ** 2), nubar_p)       # sum_k T1 = nu^a |x|^2  (B, N, Tc)
            nu_p = dmp(nubar_p + torch.bmm(nu_a, v_x), nu_p)                 # + sum_k T3  (B, N, Tc)
            pbar = torch.bmm(ahat, xhat)                                     # (B, N, Tc)
            m = pbar - shat * nubar_p                                        # (B, N, Tc)

            # --- observation node (R5-R8), two lines under AWGN ---
            nu_s = dmp(1.0 / (nu_p + N0), nu_s)                              # (B, N, Tc)
            shat = dmp((y - m) / (nu_p + N0), shat)                          # (B, N, Tc)

            # --- pseudo-measurements (R9-R12), in natural parameters.
            # prec = 1/nu, and xi = rhat/nu^r for a symbol, qhat/nu^q for a channel coefficient.
            # The brackets are the variable-side reaction correction, the accumulated
            # uncertainty over the accumulated energy.
            prec_r = torch.bmm((abar.abs() ** 2).transpose(1, 2), nu_s)      # 1/nu^r  (B, K, Tc)
            xi_r = (xbar * (prec_r - torch.bmm(nu_a.transpose(1, 2), nu_s))
                    + torch.bmm(abar.conj().transpose(1, 2), shat))          # (B, K, Tc)
            prec_q = torch.bmm(nu_s, (xbar.abs() ** 2).transpose(1, 2))      # 1/nu^q  (B, N, K)
            xi_q = (abar * (prec_q - torch.bmm(nu_s, v_x.transpose(1, 2)))
                    + torch.bmm(shat, xbar.conj().transpose(1, 2)))          # (B, N, K)

            # --- symbol denoiser (R13-R14), as log-weights so that prec_r = 0
            # (sweep 1, nu^r infinite) returns the prior instead of 0/0 ---
            logw = (2.0 * (xi_r.unsqueeze(-1) * cst.conj()).real
                    - prec_r.unsqueeze(-1) * s2)                             # (B, K, Tc, S)
            w = torch.softmax(logw, dim=-1)                                  # (B, K, Tc, S)
            xhat = (w * cst).sum(-1)                                         # (B, K, Tc)
            v_x = ((w * s2).sum(-1) - xhat.abs() ** 2).clamp_min(0.0)        # (B, K, Tc)
            xhat[:, :, :Tp] = pilots                 # the point mass overrides the observation
            v_x[:, :, :Tp] = 0.0

            # --- channel denoiser (R15-R16): the Gaussian prior combined with the
            # pseudo-measurement, which in natural parameters is adding precisions ---
            prec_post = 1.0 / tau_a + prec_q                                 # (B, N, K)
            ahat = xi_q / prec_post                                          # (B, N, K)
            nu_a = 1.0 / prec_post                                           # (B, N, K)

            # Parker (97)-(98): damped from the first sweep, unlike the four above, which
            # have no previous value on sweep 1
            xbar = eta * xhat + (1 - eta) * xbar                             # (B, K, Tc)
            abar = eta * ahat + (1 - eta) * abar                             # (B, N, K)

        x_hat = cst[w[:, :, Tp:, :].argmax(dim=-1)]              # MAP on the data columns (B,K,Td)
        h_hat = (ahat / p ** 0.5).reshape(B, L, Nr, K)           # undo a = sqrt(p) h  (B,L,Nr,K)
        return x_hat.to(in_cdtype), h_hat.to(in_cdtype)
