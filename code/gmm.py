import torch 
import torch.nn as nn
import torch.distributions as dist

class GMM(nn.Module):

    def __init__(self, device, K, means_init, random_weights = False, n_features=2):
        super(GMM, self).__init__()
        self.device = device
        self.K = K
        self.n_features = n_features

        if random_weights:
            pi_init = torch.rand(K, dtype=torch.float32)  
            self.pi = nn.Parameter(pi_init / pi_init.sum())
        else:
            self.pi = nn.Parameter(torch.ones(K) / K)

        self.means = nn.Parameter(means_init.clone().detach())
        self.chol_var = nn.Parameter(torch.eye(n_features).repeat(K, 1, 1))

    def forward(self, x):
        log_likelihoods = torch.zeros(x.size(0), self.K, device=x.device) 

        for k in range(self.K):
            lower_triangular = torch.tril(self.chol_var[k])
            var_k = lower_triangular @ lower_triangular.t() + 1e-6 * torch.eye(self.n_features, device=x.device)  
            log_probs = dist.MultivariateNormal(self.means[k], var_k).log_prob(x)
            log_likelihoods[:, k] = log_probs

        weighted_log_likelihoods = log_likelihoods + torch.log_softmax(self.pi, dim=-1)
        log_likelihood = torch.logsumexp(weighted_log_likelihoods, dim=1)

        return log_likelihood

    def log_likelihood(self, x):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            return torch.mean(self(x)).item()
        
    def sample(self, num_samples):
        cat_dist = dist.Categorical(torch.softmax(self.pi, dim=-1))
        component_indices = cat_dist.sample((num_samples,))

        samples = torch.zeros(num_samples, self.n_features, device=self.device)  
        for k in range(self.K): 
            mask = (component_indices == k)
            num_component_samples = mask.sum()
            
            if num_component_samples > 0:
                var_k = self.chol_var[k] @ self.chol_var[k].t() + 1e-6 * torch.eye(self.n_features, device=self.device)  
                component_samples = dist.MultivariateNormal(self.means[k].float(), var_k.float()).sample((num_component_samples,))
                samples[mask] = component_samples

        return samples
        
    def fit(self, loader, epochs=50, lr=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            for i, batch in enumerate(loader):
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                batch = batch.to(self.device)

                log_likelihoods = self(batch)
                loss = -torch.mean(log_likelihoods)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
