import time
import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss

from baselines import _gmm_params

def samplePG(n, n_terms=20):
    k = np.arange(1, n_terms + 1)
    g = np.random.exponential(1.0, (n, n_terms))
    return (g / (k - 0.5) ** 2).sum(1) / (2 * np.pi ** 2)   # (n,)

def _pg_gh_expectation(weights, m, v, c, n_pg, n_gh):
    t = 0.5
    z, gh_w = hermgauss(n_gh)

    omega = samplePG(n_pg)
    omega = np.clip(omega, 1e-8, None)

    cf_scale  = np.sqrt(2.0 * omega)   # (P,) — argument to ϕ_Y
    exp_scale = np.sqrt(2.0 / omega)   # (P,) — factor in exponential

    # s[p, h] = z[h] * cf_scale[p]  — where to evaluate the CF
    s = z[None, :] * cf_scale[:, None]    # (P, H)

    # ϕ_Y(s) = e^{isc} · Σ_k wk exp(i s mk - ½ s² vk)
    s3 = s[:, :, None]                             # (P, H, 1)
    log_phi_k = 1j * s3 * m - 0.5 * s3**2 * v     # (P, H, K)
    phi_y = np.exp(1j * s * c) * (weights * np.exp(log_phi_k)).sum(-1)   # (P, H)

    # e^{-iz t sqrt(2/ω)}
    herm_factor = np.exp(-1j * z[None, :] * t * exp_scale[:, None])      # (P, H)

    prefactor = np.exp(t**2 / (2 * omega)) / np.sqrt(np.pi)   # (P,)
    G = prefactor * (gh_w * herm_factor * phi_y).sum(-1).real  # (P,)

    return float(0.5 * G.mean())


def _get_mixture_params(log_pi, means, covs, w, b, x=None, mask=None):
    """Return (weights, m, v, c) as numpy — sufficient stats for the CF of Y = w⊤x."""
    pi   = log_pi.exp().cpu().numpy()
    mu   = means.detach().cpu().numpy()   # (K, D)
    cov  = covs.detach().cpu().numpy()    # (K, D, D)
    w_np = w.cpu().numpy()               # (D,)
    K    = len(pi)

    if mask is None:
        m = mu @ w_np
        v = np.array([w_np @ cov[k] @ w_np for k in range(K)])
        return pi, m, v, float(b)

    obs_idx = np.where(~mask.cpu().numpy())[0]
    mis_idx = np.where( mask.cpu().numpy())[0]
    x_np    = x.cpu().numpy()
    c       = w_np[obs_idx] @ x_np[obs_idx] + float(b)

    # Posterior p(k | x_obs)
    log_post = np.log(pi + 1e-300)
    for k in range(K):
        mu_o  = mu[k][obs_idx]
        cov_o = cov[k][obs_idx][:, obs_idx]
        diff  = x_np[obs_idx] - mu_o
        _, logdet = np.linalg.slogdet(cov_o)
        log_post[k] -= 0.5 * (diff @ np.linalg.solve(cov_o, diff) + logdet
                               + len(obs_idx) * np.log(2 * np.pi))
    log_post -= log_post.max()
    weights = np.exp(log_post); weights /= weights.sum()

    # Conditional mean/var of Y_miss = w_mis⊤x_miss | k, x_obs  (Schur complement)
    m = np.zeros(K); v = np.zeros(K)
    w_mis = w_np[mis_idx]
    for k in range(K):
        cov_oo  = cov[k][obs_idx][:, obs_idx]
        cov_mo  = cov[k][mis_idx][:, obs_idx]
        cov_mm  = cov[k][mis_idx][:, mis_idx]
        gain    = cov_mo @ np.linalg.solve(cov_oo, np.eye(len(obs_idx)))
        cond_mu  = mu[k][mis_idx] + gain @ (x_np[obs_idx] - mu[k][obs_idx])
        cond_cov = cov_mm - gain @ cov_mo.T
        m[k] = w_mis @ cond_mu
        v[k] = w_mis @ cond_cov @ w_mis

    return weights, m, v, c


class PCPG:
    def __init__(self, gmm, n_pg=500, n_gh=32):
        self.gmm  = gmm
        self.n_pg = n_pg
        self.n_gh = n_gh


    @torch.no_grad()
    def pred(self, x_test, missing_mask, w, b=0.0):
        log_pi, means, covs = _gmm_params(self.gmm)
        probs = torch.zeros(len(x_test))

        for i, (x, mask) in enumerate(zip(x_test, missing_mask)):
            weights, m, v, c = _get_mixture_params(log_pi, means, covs, w, b, x, mask)
            probs[i] = float(np.clip(_pg_gh_expectation(weights, m, v, c, self.n_pg, self.n_gh), 0.0, 1.0))

        return probs.to(self.gmm.device)