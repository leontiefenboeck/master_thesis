import torch

class MCBaseline:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, w, x_partial, n_samples=1000):
        x_samples = self.model.sample(n_samples, x_obs=x_partial)
        return float(torch.sigmoid(x_samples @ w).mean())


class MeanImputation:
    def __init__(self, mean):
        self.mean = mean

    @torch.no_grad()
    def __call__(self, w, x_partial):
        obs_mask = ~torch.isnan(x_partial)
        x = torch.where(obs_mask, x_partial, self.mean.to(x_partial))
        return float(torch.sigmoid((w * x).sum()))
