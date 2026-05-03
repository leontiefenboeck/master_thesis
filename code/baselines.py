import torch
from numpy.polynomial.hermite import hermgauss

class MCBaseline:
    def __init__(self, gmm, n_samples=2000):
        self.gmm = gmm
        self.n_samples = n_samples

    @torch.no_grad()
    def forward(self, w, b=0.0):
        samples = self.gmm.sample(self.n_samples)
        logits = samples @ w + b
        return float(torch.sigmoid(logits).mean())

    def __call__(self, w, b=0.0):
        return self.forward(w, b)

class GaussHermiteBaseline:
    def __init__(self, gmm, n_points=20):
        self.gmm = gmm

        nodes, weights = hermgauss(n_points)
        self.nodes   = torch.tensor(nodes,   dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, w, b=0.0):
        device = w.device
        nodes   = self.nodes.to(device)    # (J,)
        weights = self.weights.to(device)  # (J,)

        pi = torch.softmax(self.gmm.pi, dim=-1)  # (K,)


        K = self.gmm.K
        m = torch.zeros(K, device=device)
        v = torch.zeros(K, device=device)

        for k in range(K):
            L_k  = torch.tril(self.gmm.chol_var[k])                          # (D, D)
            m[k] = w @ self.gmm.means[k] + b                                 # scalar
            v[k] = (L_k.t() @ w).pow(2).sum()                               # scalar

        z = m[:, None] + torch.sqrt(2 * v[:, None]) * nodes[None, :]        # (K, J)
        component_expectations = (weights * torch.sigmoid(z)).sum(dim=1) \
                                 / torch.tensor(torch.pi, device=device).sqrt()  # (K,)

        return float((pi * component_expectations).sum())

    def __call__(self, w, b=0.0):
        return self.forward(w, b)