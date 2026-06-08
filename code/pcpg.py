import torch
from polyagamma import random_polyagamma
import numpy as np

class PCPG:
    def __init__(self, model, n_pg=100, n_gauss=100, n_gauss_quad=30, seed=42):
        self.model = model
        self.n_pg = n_pg
        self.n_gauss = n_gauss
        self.n_gauss_quad = n_gauss_quad
        self._rng = np.random.default_rng(seed)

        nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_gauss_quad)
        self.hermite_nodes   = torch.tensor(nodes_np,   dtype=torch.float32, device=model.device)
        self.hermite_weights = torch.tensor(weights_np, dtype=torch.float32, device=model.device)

    def sample_pg(self, n_samples):
        omega = random_polyagamma(1.0, np.zeros(n_samples), random_state=self._rng)
        return torch.tensor(omega, dtype=torch.float32, device=self.model.device)

    @torch.no_grad()
    def _expectation(self, w, x_partial, u, gamma, h_weights=None, pg_weights=None):
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

        if pg_weights is None:
            result = inner.mean()
        else:
            result = (inner * pg_weights).sum()
        return float(0.5 * result.real)

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


class PCPGMCHermite(PCPG):
    def __call__(self, w, x_partial):
        return self.pg_expectation_hermite(w, x_partial)

def pg_quadrature(n, dps=200):
    import mpmath
    mpmath.mp.dps = dps

    def L(t):
        return 1 / mpmath.cosh(mpmath.sqrt(t / 2))

    mu = [mpmath.re(((-1) ** k) * mpmath.diff(L, mpmath.mpf(0), k))
          for k in range(2 * n)]

    H  = mpmath.matrix([[mu[i + j]     for j in range(n)] for i in range(n)])
    H1 = mpmath.matrix([[mu[i + j + 1] for j in range(n)] for i in range(n)])

    R    = mpmath.cholesky(H).T
    Rinv = mpmath.inverse(R)
    J    = Rinv.T * H1 * Rinv
    J    = (J + J.T) / 2

    eigenvalues, eigenvectors = mpmath.eigh(J)
    nodes   = np.array([float(eigenvalues[k])                for k in range(n)])
    weights = np.array([float(mu[0] * eigenvectors[0, k]**2) for k in range(n)])
    return nodes, weights

class PCPGQuadrature(PCPG):
    def __init__(self, model, n_pg_quad=20, n_gauss_quad=30, seed=42):
        super().__init__(model, n_gauss_quad=n_gauss_quad, seed=seed)
        pg_nodes_np, pg_weights_np = pg_quadrature(n_pg_quad)
        self.pg_nodes   = torch.tensor(pg_nodes_np,   dtype=torch.float32, device=model.device)
        self.pg_weights = torch.tensor(pg_weights_np, dtype=torch.float32, device=model.device)

    def pg_expectation_quadrature(self, w, x_partial):
        gamma = self.pg_nodes[:, None]
        u = (2 * gamma).sqrt() * self.hermite_nodes
        return self._expectation(w, x_partial, u, gamma,
                                 h_weights=self.hermite_weights,
                                 pg_weights=self.pg_weights)

    def __call__(self, w, x_partial):
        return self.pg_expectation_quadrature(w, x_partial)
