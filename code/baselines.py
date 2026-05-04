import torch
import numpy as np

# TODO: what if missing data - conditional?
class MCBaseline:
    def __init__(self, model, n_samples=2000):
        self.model = model
        self.n_samples = n_samples

    @torch.no_grad()
    def forward(self, w):
        samples = self.model.sample(self.n_samples)
        logits = samples @ w
        return float(torch.sigmoid(logits).mean())

    def __call__(self, w):
        return self.forward(w)

class GaussHermiteBaseline:
    def __init__(self, gmm, n_points=20):
        self.gmm = gmm

        nodes, weights = np.polynomial.hermite.hermgauss(n_points)
        self.nodes   = torch.tensor(nodes,   dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, w):
        device = w.device
        nodes   = self.nodes.to(device)    # (J,)
        weights = self.weights.to(device)  # (J,)

        pi = torch.softmax(self.gmm.pi, dim=-1)  # (K,)

        variances = torch.exp(self.gmm.log_vars) + 1e-6  # (K, D)

        K = self.gmm.K
        m = torch.zeros(K, device=device)
        v = torch.zeros(K, device=device)

        for k in range(K):
            m[k] = w @ self.gmm.means[k]                                # scalar
            L_k_t_w = w * torch.sqrt(variances[k])                          # (D,)
            v[k] = L_k_t_w.pow(2).sum()                                     # scalar

        z = m[:, None] + torch.sqrt(2 * v[:, None]) * nodes[None, :]        # (K, J)
        component_expectations = (weights * torch.sigmoid(z)).sum(dim=1) \
                                 / torch.sqrt(torch.tensor(torch.pi, device=device))  # (K,)

        return float((pi * component_expectations).sum())

    def __call__(self, w):
        return self.forward(w)