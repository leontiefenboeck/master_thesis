import torch
from polyagamma import random_polyagamma
import numpy as np

class PCPG:
    def __init__(self, model, n_pg=100, n_gauss=100, n_hermite=30, seed=42):
        self.model = model
        self.n_pg = n_pg
        self.n_gauss = n_gauss
        self.n_hermite = n_hermite
        self.seed = seed

        nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_hermite)
        self.hermite_nodes   = torch.tensor(nodes_np,   dtype=torch.float32, device=model.device)
        self.hermite_weights = torch.tensor(weights_np, dtype=torch.float32, device=model.device)

    def sample_pg(self, n_samples):
        omega = random_polyagamma(1.0, np.zeros(n_samples), random_state=self.seed)
        return torch.tensor(omega, dtype=torch.float32, device=self.model.device)

    @torch.no_grad()
    def _expectation(self, w, x_partial, u, gamma, h_weights=None):
        obs_mask = ~torch.isnan(x_partial)
        wo, wm = w[obs_mask], w[~obs_mask]
        wo_xo = wo @ x_partial[obs_mask]
        inv_gamma = 1.0 / gamma

        real = (inv_gamma / 8).expand_as(u)
        im = u * (0.5 * inv_gamma - wo_xo)
        factor = torch.exp(torch.complex(real, im))

        s = torch.outer(-u.flatten(), wm)
        cf = self.model.characteristic_function(s, x_obs=x_partial)
        cf = cf.view_as(factor)

        integrand = factor * cf
        if h_weights is None:
            inner = integrand.mean(dim=-1)
        else:
            inner = (integrand * h_weights).sum(dim=-1) / np.sqrt(np.pi)
        return float(0.5 * inner.mean().real)

    def pg_expectation(self, w, x_partial):
        # ω ~ PG(1, 0), then u | ω ~ N(0, ω).
        gamma = self.sample_pg(self.n_pg)[:, None]
        u = torch.randn(self.n_pg, self.n_gauss, device=self.model.device) * gamma.sqrt()
        return self._expectation(w, x_partial, u, gamma)

    def pg_expectation_hermite(self, w, x_partial):
        # ω ~ PG(1, 0), then u_i = √(2ω)·t_i at Gauss-Hermite nodes.
        gamma = self.sample_pg(self.n_pg)[:, None]
        u = (2 * gamma).sqrt() * self.hermite_nodes
        return self._expectation(w, x_partial, u, gamma, h_weights=self.hermite_weights)

    def __call__(self, w, x_partial):
        return self.pg_expectation(w, x_partial)


class PCPGHermite(PCPG):
    def __call__(self, w, x_partial):
        return self.pg_expectation_hermite(w, x_partial)
