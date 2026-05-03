import torch
from polyagamma import random_polyagamma
import numpy as np

class PCPG:
    def __init__(self, gmm, n_pg=100, n_mc=1000):
        self.gmm = gmm
        self.n_pg = n_pg
        self.n_mc = n_mc

    def sample_pg(self, n_samples):
        omega = random_polyagamma(1.0, np.zeros(n_samples))
        return torch.tensor(omega, dtype=torch.float32, device=self.gmm.device)

    @torch.no_grad()
    def forward(self, w, b=0.0):
        omegas = self.sample_pg(self.n_pg)
        
        expectation = 0.0
        for omega in omegas:
            samples = self.gmm.sample(self.n_mc)
            y = samples @ w + b 
            
            inner_exp = torch.exp(y / 2 - omega * y**2 / 2)
            inner_expectation = inner_exp.mean()
            
            expectation += inner_expectation
        
        expectation = expectation / self.n_pg        
        return 0.5 * expectation

    def __call__(self, w, b=0.0):
        return self.forward(w, b)

 