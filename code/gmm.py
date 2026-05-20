import time
import torch
import torch.nn as nn
import torch.distributions as dist

class GMM(nn.Module):
    def __init__(self, device, K, means_init, random_weights=False, n_features=2):
        super(GMM, self).__init__()
        self.device = device
        self.K = K
        self.n_features = n_features

        if random_weights:
            pi_init = torch.rand(K, dtype=torch.float32)  
            self.pi = nn.Parameter(pi_init)
        else:
            self.pi = nn.Parameter(torch.ones(K))

        self.means = nn.Parameter(means_init.clone().detach())
        self.log_vars = nn.Parameter(torch.zeros(K, n_features))

    def forward(self, x):
        weights_log_pdf = torch.log_softmax(self.pi, dim=-1)
        variances = torch.exp(self.log_vars) + 1e-6
        log_probs = dist.Normal(self.means, variances.sqrt()).log_prob(x.unsqueeze(1)).sum(dim=-1)

        weighted_log_probs = log_probs + weights_log_pdf
        return torch.logsumexp(weighted_log_probs, dim=1)

    def _component_weights(self, x_obs=None):
        # Prior softmax(pi) if nothing observed, else posterior p(k | x_obs).
        if x_obs is None:
            return torch.softmax(self.pi, dim=-1)

        obs_mask = ~torch.isnan(x_obs)
        stds_obs = torch.sqrt(torch.exp(self.log_vars[:, obs_mask]) + 1e-6)
        log_lik  = dist.Normal(self.means[:, obs_mask], stds_obs).log_prob(x_obs[obs_mask]).sum(dim=-1)

        return torch.softmax(torch.log_softmax(self.pi, dim=-1) + log_lik, dim=-1)

    def sample(self, num_samples, x_obs=None):
        with torch.no_grad():
            weights = self._component_weights(x_obs)
            indices = dist.Categorical(weights).sample((num_samples,))

            means = self.means[indices]
            stds  = torch.sqrt(torch.exp(self.log_vars[indices]) + 1e-6)
            eps   = torch.randn(num_samples, self.n_features, device=self.device)
            samples = means + eps * stds

            if x_obs is not None:
                obs_mask = ~torch.isnan(x_obs)
                samples[:, obs_mask] = x_obs[obs_mask]
            return samples

    def characteristic_function(self, t, x_obs=None):
        weights = self._component_weights(x_obs)
        if x_obs is None:
            means, variances = self.means, torch.exp(self.log_vars) + 1e-6
        else:
            mis_mask  = torch.isnan(x_obs)
            means     = self.means[:, mis_mask]
            variances = torch.exp(self.log_vars[:, mis_mask]) + 1e-6

        squeeze = t.dim() == 1
        if squeeze:
            t = t.unsqueeze(0)

        t_exp    = t.unsqueeze(1)                          # (N, 1, D_eff)
        t_dot_mu = (t_exp * means).sum(dim=-1)             # (N, K)
        t_sq_var = (t_exp**2 * variances).sum(dim=-1)      # (N, K)

        log_cf = torch.complex(-0.5 * t_sq_var, t_dot_mu)
        cf = (weights * torch.exp(log_cf)).sum(dim=-1)
        return cf.squeeze(0) if squeeze else cf


    def fit(self, loader, epochs=50, lr=0.01):
        print("Fitting GMM...", end=" ", flush=True)
        t0 = time.perf_counter()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for _ in range(epochs):
            for batch in loader:
                if isinstance(batch, (tuple, list)): batch = batch[0]
                batch = batch.to(self.device)

                loss = -torch.mean(self(batch))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"done ({time.perf_counter() - t0:.1f}s)")