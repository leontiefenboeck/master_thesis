import torch

class MCBaseline:
    def __init__(self, model, n_samples=2000):
        self.model = model
        self.n_samples = n_samples

    @torch.no_grad()
    def mc_expectation(self, w, x_partial):
        x_samples = self.model.sample(self.n_samples, x_obs=x_partial)
        return float(torch.sigmoid(x_samples @ w).mean())

    def __call__(self, w, x_partial):
        return self.mc_expectation(w, x_partial)


class MeanImputation:
    def __init__(self, mean):
        self.mean = mean

    @torch.no_grad()
    def __call__(self, w, x_partial):
        obs_mask = ~torch.isnan(x_partial)
        x = torch.where(obs_mask, x_partial, self.mean.to(x_partial))
        return float(torch.sigmoid((w * x).sum()))

