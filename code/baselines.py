import torch
import numpy as np
import torch.distributions as dist

# TODO: what if missing data - conditional?
class MCBaseline:
    def __init__(self, model, n_samples=2000):
        self.model = model
        self.n_samples = n_samples

    @torch.no_grad()
    def marg(self, w):
        samples = self.model.sample(self.n_samples)
        logits = samples @ w
        return float(torch.sigmoid(logits).mean())

    @torch.no_grad()
    def cond(self, w, x_partial):
        w = w.to(self.model.device)
        x_partial = torch.as_tensor(x_partial, dtype=torch.float32, device=self.model.device)
        observed_mask = ~torch.isnan(x_partial)
        missing_mask = ~observed_mask

        if observed_mask.all():
            logits = (w * x_partial).sum()
            return float(torch.sigmoid(logits).item())

        component_weights = self.model.conditional_mixture_weights(x_partial, observed_mask)
        cat_dist = dist.Categorical(component_weights)
        indices = cat_dist.sample((self.n_samples,))

        component_means = self.model.means[indices]
        component_vars = torch.exp(self.model.log_vars[indices]) + 1e-6

        if missing_mask.sum() > 0:
            mu_miss = component_means[:, missing_mask]
            sigma_miss = component_vars[:, missing_mask].sqrt()
            noise = torch.randn(self.n_samples, missing_mask.sum(), device=self.model.device)
            samples_miss = mu_miss + noise * sigma_miss

            x_full = x_partial.unsqueeze(0).repeat(self.n_samples, 1)
            x_full[:, missing_mask] = samples_miss
        else:
            x_full = x_partial.unsqueeze(0).repeat(self.n_samples, 1)

        logits = x_full @ w
        return float(torch.sigmoid(logits).mean().item())

class GaussHermiteBaseline:
    def __init__(self, gmm, n_points=20):
        self.gmm = gmm

        nodes, weights = np.polynomial.hermite.hermgauss(n_points)
        self.nodes   = torch.tensor(nodes,   dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    @torch.no_grad()
    def marg(self, w):
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