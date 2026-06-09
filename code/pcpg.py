from polyagamma import random_polyagamma
from scipy.optimize import nnls
import numpy as np
import mpmath
import torch

class _PCPGBase:
    def __init__(self, model, seed=42):
        self.model = model
        self._rng = np.random.default_rng(seed)

    def sample_pg(self, n_samples):
        omega = random_polyagamma(1.0, np.zeros(n_samples), random_state=self._rng)
        return torch.tensor(omega, dtype=torch.float32, device=self.model.device)

    def _expectation_diff(self, w, x_partial, u, gamma, h_weights=None, pg_weights=None):
        """Compute E[σ(wᵀx) | x_obs] as a scalar tensor; gradients flow through w."""
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
        return 0.5 * result.real

    @torch.no_grad()
    def _expectation(self, w, x_partial, u, gamma, h_weights=None, pg_weights=None):
        return float(self._expectation_diff(w, x_partial, u, gamma, h_weights, pg_weights))


class PCPGMC(_PCPGBase):
    """Both integrals (PG outer, Gaussian inner) estimated by Monte Carlo."""

    def __call__(self, w, x_partial, n_pg=100, n_gauss=100):
        gamma = self.sample_pg(n_pg)[:, None]
        u = torch.randn(n_pg, n_gauss, device=self.model.device) * gamma.sqrt()
        return self._expectation(w, x_partial, u, gamma)


class PCPGMCHermite(_PCPGBase):
    """PG outer integral by MC; Gaussian inner integral by Gauss-Hermite quadrature."""

    def __init__(self, model, n_gauss_quad=30, seed=42):
        super().__init__(model, seed=seed)
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_gauss_quad)
        self.hermite_nodes   = torch.tensor(nodes_np,   dtype=torch.float32, device=model.device)
        self.hermite_weights = torch.tensor(weights_np, dtype=torch.float32, device=model.device)

    def __call__(self, w, x_partial, n_pg=100):
        gamma = self.sample_pg(n_pg)[:, None]
        u = (2 * gamma).sqrt() * self.hermite_nodes
        return self._expectation(w, x_partial, u, gamma, h_weights=self.hermite_weights)

def pg_gauss_quadrature(n):
    mpmath.mp.dps = 200

    def laplace_transform(t):
        return 1 / mpmath.cosh(mpmath.sqrt(t / 2))

    moments = [mpmath.re(((-1) ** k) * mpmath.diff(laplace_transform, mpmath.mpf(0), k)) for k in range(2 * n)]

    def hankel_matrix(offset):
        return mpmath.matrix([[moments[i + j + offset] for j in range(n)] for i in range(n)])

    H = hankel_matrix(0)
    H1 = hankel_matrix(1)

    R = mpmath.cholesky(H).T
    Rinv = mpmath.inverse(R)
    J = Rinv.T * H1 * Rinv
    J = (J + J.T) / 2

    eigenvalues, eigenvectors = mpmath.eigh(J)
    nodes = np.array([float(v) for v in eigenvalues])
    weights = np.array([float(moments[0] * eigenvectors[0, k] ** 2) for k in range(n)])
    return nodes, weights

def pg_nnls_quadrature(M=500, R=1500, gamma_min=0.00001, gamma_max=8.0, z_max=20.0):
    z       = np.linspace(0, z_max, R)
    gamma   = np.logspace(np.log10(gamma_min), np.log10(gamma_max), M)

    b = 1.0 / np.cosh(np.sqrt(z / 2.0))
    A = np.exp(-np.outer(z, gamma))

    weights, _ = nnls(A, b)
    weights = weights.astype(np.float64)

    total = weights.sum()
    if total > 0:
        weights /= total

    selected = weights > 1e-14
    if not selected.any():
        selected[np.argmax(weights)] = True

    return gamma[selected], weights[selected]

class PCPGQuadrature(_PCPGBase):
    """Both integrals fully deterministic: Gauss-PG outer, Gauss-Hermite inner."""

    def __init__(self, model, n_pg_quad=10, n_gauss_quad=30, seed=42):
        super().__init__(model, seed=seed)
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(n_gauss_quad)
        self.hermite_nodes   = torch.tensor(nodes_np,   dtype=torch.float32, device=model.device)
        self.hermite_weights = torch.tensor(weights_np, dtype=torch.float32, device=model.device)

        pg_nodes_np, pg_weights_np = pg_gauss_quadrature(n_pg_quad)
        self.pg_nodes   = torch.tensor(pg_nodes_np,   dtype=torch.float32, device=model.device)
        self.pg_weights = torch.tensor(pg_weights_np, dtype=torch.float32, device=model.device)

    def __call__(self, w, x_partial):
        gamma = self.pg_nodes[:, None]
        u = (2 * gamma).sqrt() * self.hermite_nodes
        return self._expectation(w, x_partial, u, gamma, h_weights=self.hermite_weights, pg_weights=self.pg_weights)


class PCPGMCHermiteDiff(PCPGMCHermite):
    """PCPGMCHermite with gradient support for training w.

    Identical to PCPGMCHermite but exposes a forward() that returns a scalar
    tensor so gradients can flow through w during classifier training.
    """

    def forward(self, w, x_partial, n_pg: int = 50):
        gamma = self.sample_pg(n_pg)[:, None]
        u = (2 * gamma).sqrt() * self.hermite_nodes
        return self._expectation_diff(w, x_partial, u, gamma, h_weights=self.hermite_weights)
