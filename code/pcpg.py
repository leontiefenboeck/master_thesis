import torch
from polyagamma import random_polyagamma
import numpy as np

class PCPG:
    def __init__(self, model, n_pg=20, n_hermite=10, seed=42):
        self.model = model
        self.n_pg = n_pg # number of polygamma samples, TODO: how many?
        self.seed = seed
        
        # Hermite quadrature 
        nodes, weights = np.polynomial.hermite.hermgauss(n_hermite)
        self.nodes = torch.tensor(nodes, dtype=torch.float32, device=self.model.device)
        self.weights = torch.tensor(weights, dtype=torch.float32, device=self.model.device)

    # TODO: am I really supposed to sample this
    def sample_pg(self, n_samples):
        omega = random_polyagamma(1.0, np.zeros(n_samples), random_state=self.seed)
        return torch.tensor(omega, dtype=torch.float32, device=self.model.device)

    def pg_expectation(self, w, shift=0.0, component_weights=None):
        w = w.to(self.model.device)
        omegas = self.sample_pg(self.n_pg)          # (n_pg,)
        t = 0.5

        a = t - omegas * shift
        factor_term = torch.exp(t * shift - 0.5 * omegas * shift**2 + a**2 / (2.0 * omegas))
        factor_term = factor_term / torch.sqrt(torch.tensor(torch.pi, device=omegas.device))

        s = torch.sqrt(2.0 * omegas).unsqueeze(1) * self.nodes               # (n_pg, n_hermite)
        s_vector = s.unsqueeze(-1) * w                                         # (n_pg, n_hermite, d)

        phi = self.model.characteristic_function(s_vector, weights=component_weights)

        exp_term = torch.exp(-1j * self.nodes * a.unsqueeze(1) * torch.sqrt(2.0 / omegas).unsqueeze(1))
        integrand = self.weights.to(phi.device) * exp_term * phi              # (n_pg, n_hermite)
        g = factor_term * integrand.sum(dim=-1)                               # (n_pg,)

        return 0.5 * g.real.mean()

    @torch.no_grad()
    def marg(self, w):
        return self.pg_expectation(w)

    @torch.no_grad()
    def cond(self, w, x_partial):
        x_partial = torch.as_tensor(x_partial, dtype=torch.float32, device=self.model.device)
        observed_mask = ~torch.isnan(x_partial)

        if observed_mask.all():
            return float(torch.sigmoid((w.to(self.model.device) * x_partial).sum()).item())

        missing_mask = ~observed_mask
        shift = (w.to(self.model.device)[observed_mask] * x_partial[observed_mask]).sum()

        w_projected = torch.zeros_like(w, device=self.model.device)
        w_projected[missing_mask] = w.to(self.model.device)[missing_mask]

        component_weights = self.model.conditional_mixture_weights(x_partial, observed_mask)
        return self.pg_expectation(w_projected, shift=shift, component_weights=component_weights)


 