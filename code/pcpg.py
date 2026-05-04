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

    @torch.no_grad()
    def forward(self, w):
        omegas = self.sample_pg(self.n_pg)          # (n_pg,)
        t = 0.5

        factor_term = torch.exp(t**2 / (2 * omegas)) / torch.tensor(torch.pi, device=omegas.device).sqrt()
        exp_term = torch.exp(-1j * self.nodes * t * torch.sqrt(2.0 / omegas).unsqueeze(1))      # (n_pg, n_hermite)

        # TODO: where to put w
        s = torch.sqrt(2.0 * omegas).unsqueeze(1) * self.nodes               # (n_pg, n_hermite)
        s_vector = s.unsqueeze(-1) * w                                       # (n_pg, n_hermite, d)

        phi = self.model.characteristic_function(s_vector)

        integrand = self.weights * exp_term * phi                            # (n_pg, n_hermite)
        g = factor_term * integrand.sum(dim=-1)                              # (n_pg,)

        return 0.5 * g.real.mean()

    def __call__(self, w):
        return self.forward(w)

 