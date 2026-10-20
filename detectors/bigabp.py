r"""
Bilinear Gaussian belief propagation (BiGaBP) of Ito, Takahashi, Ibi and Sampei
(IEEE Trans. Commun. 72(2), 2024, reference [5] of README.md), Algorithm 1.

BiGaBP is Gaussian belief propagation on the same bilinear factor graph as the JCD.  It
does not take the large-system limit: where BiG-AMP corrects the self-feedback with an
Onsager term, BiGaBP keeps one message per edge and leaves the target edge out of every
combination.  Their indices (m user, n antenna, k channel use) are ours (k, n, t).

Every quantity lives on an edge, stored as (B, N, K, Tc):
    xhat[n,k,t], psi_x   symbol of user k at use t as seen by antenna n
    ahat[n,k,t], psi_a   channel of (n,k) as seen by use t, a = sqrt(p) h

One sweep (line numbers of their Algorithm 1):
    factor nodes (5-8)   soft interference cancellation with the per-edge replicas,
                         ytil = y - sum_{k'!=k} ahat xhat, and the cavity noises
                         xi_y = N0 + sum_{k'!=k} (|ahat|^2 psi_x + psi_a (|xhat|^2 + psi_x)),
                         xi_x = xi_y + E_s psi_a,   xi_a = xi_y + phi psi_x
    symbols (9-12)       combine the other antennas i != n, then the soft demapper
    channels (13-16)     combine the other uses i != t, then the shrinkage
                         phi q / (psi_q + phi) of their eq. (21)
    damping              new = eta update + (1 - eta) old on all four, from the first sweep
The decision (lines 20-24) combines over all antennas with the last sweep's quantities,
takes the nearest constellation point, and re-estimates the channel from the pilots and
the decided symbols.

Differences from the paper, all deliberate:
- Cell-free prior: the channel prior variance phi is p beta_{lk} per link, where the
  paper has one phi = 1/N for every link (co-located array).
- No belief scaling.  Their adaptive scaled belief (gamma(t) in the demapper, alpha(t)
  and beta(t) on the pilot and data terms of the channel combine) targets short
  non-orthogonal pilots.  At the orthogonal-pilot operating point it is switched off:
  the demapper is the exact posterior of their eq. (18) and alpha = beta = 1.
- A fixed number of sweeps instead of their stopping test (line 17), as for every other
  receiver here.

The initial channel is their least-squares estimate with orthogonal pilots, eq. (4), and
the pilots are point-mass symbols throughout.  The receiver splits between the APs and
the CPU as jcd does: the factor nodes and the channel nodes use only the antenna's own
observation and run at the AP, and the symbol nodes, which combine the other antennas,
run at the CPU, as the symbol node of jcd does.  Each sweep is one fronthaul round trip.
"""
from __future__ import annotations

import torch

from detectors.base import Detector, Observation, register


@register("bigabp")
class BiGaBPDetector(Detector):
    """Bilinear Gaussian BP with per-edge messages (Ito et al., Algorithm 1), no belief scaling."""

    def __init__(self, config):
        self.cfg = config
        self.n_iters = config.n_iters       # BP sweeps T
        self.eta = config.damping           # damping: new = (1 - eta) old + eta update

    def detect(self, obs: Observation):
        if obs.y_pilot is None or obs.pilots is None:
            raise ValueError("BiGaBP needs pilots to break the scale ambiguity (tau_p > 0)")
        p, N0 = obs.p_tx, obs.N0
        dev = obs.y_data.device
        B, L, Nr, Td = obs.y_data.shape
        K = obs.beta.shape[1]
        N, Tp = L * Nr, obs.pilots.shape[1]
        Tc = Tp + Td
        rdtype, cdtype = torch.float64, torch.complex128       # see detectors/base.to_double
        in_cdtype = obs.y_data.dtype                           # returned in the caller's dtype
        cst = obs.constellation.to(cdtype)                     # (S,)
        s2 = (cst.abs() ** 2).to(rdtype)                       # |s|^2  (S,)
        Ex = float(s2.mean())                                  # symbol energy E_s
        eta = self.eta

        # the whole block at once, pilots first
        y = torch.cat([obs.y_pilot, obs.y_data], dim=-1).reshape(B, N, Tc).to(cdtype)  # (B, N, Tc)
        beta = obs.beta.repeat_interleave(Nr, dim=0).to(dev).to(rdtype)                # (N, K)
        phi = (p * beta).view(1, N, K, 1)                      # prior variance of a  (1, N, K, 1)
        pil = obs.pilots.to(cdtype)                            # (K, Tp)
        E = (pil.abs() ** 2).sum(dim=1).to(rdtype)             # pilot energy per user  (K,)

        # --- initialisation (lines 1-3): least-squares channel, their eq. (4) ---
        a_ls = torch.einsum("bnt,kt->bnk", y[..., :Tp], pil.conj()) / E           # (B, N, K)
        ahat = a_ls.unsqueeze(-1).expand(B, N, K, Tc).clone()                     # (B, N, K, Tc)
        psi_a = (N0 / E).view(1, 1, K, 1).expand(B, N, K, Tc).clone()             # (B, N, K, Tc)
        xhat = torch.zeros(B, N, K, Tc, dtype=cdtype, device=dev)                # (B, N, K, Tc)
        psi_x = torch.full((B, N, K, Tc), Ex, dtype=rdtype, device=dev)          # (B, N, K, Tc)
        xhat[..., :Tp] = pil.view(1, 1, K, Tp)                 # pilots are point masses
        psi_x[..., :Tp] = 0.0

        for _ in range(self.n_iters):
            # --- factor nodes (lines 5-8): soft IC keeping user k, and the cavity noises ---
            s = ahat * xhat                                    # per-edge replica  (B, N, K, Tc)
            ytil = y.unsqueeze(2) - s.sum(dim=2, keepdim=True) + s                  # (B, N, K, Tc)
            e = ahat.abs() ** 2 * psi_x + psi_a * (xhat.abs() ** 2 + psi_x)        # (B, N, K, Tc)
            xi_y = N0 + e.sum(dim=2, keepdim=True) - e         # other users only  (B, N, K, Tc)
            xi_x = xi_y + Ex * psi_a                           # symbol side  (B, N, K, Tc)
            xi_a = xi_y + phi * psi_x                          # channel side  (B, N, K, Tc)

            # --- symbol nodes (lines 9-12): combine the other antennas i != n ---
            w_x = ahat.abs() ** 2 / xi_x                       # (B, N, K, Tc)
            m_x = ahat.conj() * ytil / xi_x                    # (B, N, K, Tc)
            prec_r = w_x.sum(dim=1, keepdim=True) - w_x        # 1/psi_r, leave antenna n out
            num_r = m_x.sum(dim=1, keepdim=True) - m_x         # rhat/psi_r
            logw = (2.0 * (num_r.unsqueeze(-1) * cst.conj()).real
                    - prec_r.unsqueeze(-1) * s2)               # (B, N, K, Tc, S)
            w = torch.softmax(logw, dim=-1)                    # posterior over the constellation
            xhat_new = (w * cst).sum(-1)                       # their eq. (18a)  (B, N, K, Tc)
            psi_x_new = ((w * s2).sum(-1) - xhat_new.abs() ** 2).clamp_min(0.0)    # eq. (18b)

            # --- channel nodes (lines 13-16): combine the other uses i != t ---
            w_a = xhat.abs() ** 2 / xi_a                       # (B, N, K, Tc)
            m_a = xhat.conj() * ytil / xi_a                    # (B, N, K, Tc)
            prec_q = w_a.sum(dim=3, keepdim=True) - w_a        # 1/psi_q, leave use t out
            num_q = m_a.sum(dim=3, keepdim=True) - m_a         # qhat/psi_q
            prec_post = 1.0 / phi + prec_q                     # eq. (21) in natural form
            ahat_new = num_q / prec_post                       # = phi q / (psi_q + phi)
            psi_a_new = 1.0 / prec_post                        # = phi psi_q / (psi_q + phi)

            # keep this sweep's factor-node quantities for the decision (lines 20-24)
            ahat_T, ytil_T, xi_x_T, xi_a_T = ahat, ytil, xi_x, xi_a

            # --- damping, the pilots stay point masses ---
            xhat = eta * xhat_new + (1 - eta) * xhat
            psi_x = eta * psi_x_new + (1 - eta) * psi_x
            xhat[..., :Tp] = pil.view(1, 1, K, Tp)
            psi_x[..., :Tp] = 0.0
            ahat = eta * ahat_new + (1 - eta) * ahat
            psi_a = eta * psi_a_new + (1 - eta) * psi_a

        # --- decision (lines 20-21): combine all antennas, nearest constellation point ---
        prec = (ahat_T.abs() ** 2 / xi_x_T).sum(dim=1)                          # (B, K, Tc)
        num = (ahat_T.conj() * ytil_T / xi_x_T).sum(dim=1)                      # (B, K, Tc)
        logw = 2.0 * (num.unsqueeze(-1) * cst.conj()).real - prec.unsqueeze(-1) * s2
        x_dec = cst[logw.argmax(dim=-1)]                       # nearest point to num/prec  (B, K, Tc)
        x_dec[..., :Tp] = pil                                  # pilot columns are known

        # --- final channel (lines 22-24): pilots and decided symbols, all uses ---
        xd = x_dec.unsqueeze(1)                                # (B, 1, K, Tc)
        prec_q = (xd.abs() ** 2 / xi_a_T).sum(dim=3)           # (B, N, K)
        num_q = (xd.conj() * ytil_T / xi_a_T).sum(dim=3)       # (B, N, K)
        a_fin = num_q / (1.0 / phi.squeeze(-1) + prec_q)       # (B, N, K)

        x_hat = x_dec[..., Tp:]                                # data columns  (B, K, Td)
        h_hat = (a_fin / p ** 0.5).reshape(B, L, Nr, K)        # undo a = sqrt(p) h  (B, L, Nr, K)
        return x_hat.to(in_cdtype), h_hat.to(in_cdtype)


@register("bigabp_node")
class NodewiseBiGaBPDetector(Detector):
    """Ablation: an earlier node-wise implementation of BiGaBP, kept for comparison.

    Same factor-node equations, but one belief per node instead of one message per edge:
    ahat is (B, N, K) and xhat (B, K, Tc), and the symbol and channel combines run over
    all antennas and all uses, so each update re-uses its own information.  The only
    leave-one-out is over users, in the cavity variance.  The channel-side cavity uses
    the current estimate |ahat|^2 where the paper uses the prior phi, and the channel
    starts from the prior (ahat = 0) rather than the least-squares estimate.  The first
    sweep is therefore not the pilot estimate: it shrinks each link to about 0.6 of the
    pilot MMSE estimate (one drop at the reference point).  Kept to measure what the
    per-edge exclusion of bigabp is worth.
    """

    def __init__(self, config):
        self.cfg = config
        self.n_iters = config.n_iters       # BP sweeps
        self.eta = config.damping           # damping: new = (1 - eta) old + eta update

    def detect(self, obs: Observation):
        if obs.y_pilot is None or obs.pilots is None:
            raise ValueError("BiGaBP needs pilots to break the scale ambiguity (tau_p > 0)")
        p, N0 = obs.p_tx, obs.N0
        dev = obs.y_data.device
        B, L, Nr, Td = obs.y_data.shape
        K = obs.beta.shape[1]
        N, Tp = L * Nr, obs.pilots.shape[1]
        Tc = Tp + Td
        rdtype, cdtype = torch.float64, torch.complex128       # see detectors/base.to_double
        in_cdtype = obs.y_data.dtype
        cst = obs.constellation.to(cdtype)                     # (S,)
        s2 = (cst.abs() ** 2).to(rdtype)                       # |s|^2  (S,)
        Ex = float(s2.mean())                                  # average symbol energy
        eta = self.eta

        # whole block at once, pilots first (a pilot is a point-mass symbol)
        y = torch.cat([obs.y_pilot, obs.y_data], dim=-1).reshape(B, N, Tc).to(cdtype)  # (B,N,Tc)
        beta = obs.beta.repeat_interleave(Nr, dim=0).to(dev).to(rdtype)                # (N,K)
        tau_a = (p * beta).unsqueeze(0).expand(B, N, K).contiguous()   # prior var of a (B,N,K)
        pilots = obs.pilots.to(cdtype).unsqueeze(0).expand(B, K, Tp)   # (B,K,Tp)

        # --- initialisation from the priors, a = 0 ---
        ahat = torch.zeros(B, N, K, dtype=cdtype, device=dev)          # (B,N,K)  E{a}=0
        nu_a = tau_a.clone()                                           # (B,N,K)  p*beta
        xhat = torch.zeros(B, K, Tc, dtype=cdtype, device=dev)         # (B,K,Tc)
        v_x = torch.full((B, K, Tc), Ex, dtype=rdtype, device=dev)     # (B,K,Tc)
        xhat[:, :, :Tp] = pilots                                       # point-mass pilots
        v_x[:, :, :Tp] = 0.0
        w = None                                                       # last symbol weights

        for i in range(self.n_iters):
            first = (i == 0)
            dmp = lambda new, old: new if first else eta * new + (1 - eta) * old

            a2 = ahat.abs() ** 2                                       # (B,N,K)
            x2 = xhat.abs() ** 2                                       # (B,K,Tc)

            # --- per-antenna variance of the z = a x sum, and the per-user leave-one-out ---
            # V_{n,t} = sum_k [ |a_{n,k}|^2 v_{k,t} + (|x_{k,t}|^2 + v_{k,t}) nu^a_{n,k} ]
            V = torch.bmm(a2, v_x) + torch.bmm(nu_a, x2 + v_x)         # (B,N,Tc)
            # own contribution of user k at (n,t):  (B,N,K,Tc)
            own = a2.unsqueeze(-1) * v_x.unsqueeze(1) \
                + nu_a.unsqueeze(-1) * (x2 + v_x).unsqueeze(1)
            common = N0 + V.unsqueeze(2) - own                        # (B,N,K,Tc)  leave user k out
            xi_x = common + Ex * nu_a.unsqueeze(-1)                    # symbol-side cavity noise
            xi_h = common + a2.unsqueeze(-1) * v_x.unsqueeze(1)        # channel-side cavity noise

            # --- soft interference cancellation: residual keeping user k (Algorithm 1, lines
            #     5-8, with node-wise instead of per-edge replicas) ---
            pbar = torch.bmm(ahat, xhat)                              # (B,N,Tc)  sum_k a x
            ytil = y.unsqueeze(2) - pbar.unsqueeze(2) \
                + ahat.unsqueeze(-1) * xhat.unsqueeze(1)              # (B,N,K,Tc)

            # --- symbol belief: combine the residuals over all N antennas ---
            prec_r = (a2.unsqueeze(-1) / xi_x).sum(dim=1)            # 1/nu^r  (B,K,Tc)
            xi_r = (ahat.conj().unsqueeze(-1) * ytil / xi_x).sum(dim=1)  # rhat/nu^r  (B,K,Tc)
            # log-weights (natural form, so prec_r = 0 on sweep 1 returns the prior)
            logw = (2.0 * (xi_r.unsqueeze(-1) * cst.conj()).real
                    - prec_r.unsqueeze(-1) * s2)                      # (B,K,Tc,S)
            w = torch.softmax(logw, dim=-1)                          # (B,K,Tc,S)
            xhat_new = (w * cst).sum(-1)                             # (B,K,Tc)
            v_x_new = ((w * s2).sum(-1) - xhat_new.abs() ** 2).clamp_min(0.0)
            xhat_new[:, :, :Tp] = pilots                            # point mass overrides
            v_x_new[:, :, :Tp] = 0.0

            # --- channel belief: combine the residuals over all tau_c uses, then shrink ---
            prec_q = (x2.unsqueeze(1) / xi_h).sum(dim=-1)           # 1/nu^q  (B,N,K)
            xi_q = (xhat.conj().unsqueeze(1) * ytil / xi_h).sum(dim=-1)  # qhat/nu^q  (B,N,K)
            prec_post = 1.0 / tau_a + prec_q                        # (B,N,K)
            ahat_new = xi_q / prec_post                             # (B,N,K)
            nu_a_new = 1.0 / prec_post                              # (B,N,K)

            # --- damping in the moments ---
            xhat = dmp(xhat_new, xhat)
            v_x = dmp(v_x_new, v_x)
            xhat[:, :, :Tp] = pilots
            v_x[:, :, :Tp] = 0.0
            ahat = dmp(ahat_new, ahat)
            nu_a = dmp(nu_a_new, nu_a)

        x_hat = cst[w[:, :, Tp:, :].argmax(dim=-1)]              # MAP on the data columns (B,K,Td)
        h_hat = (ahat / p ** 0.5).reshape(B, L, Nr, K)           # undo a = sqrt(p) h  (B,L,Nr,K)
        return x_hat.to(in_cdtype), h_hat.to(in_cdtype)
