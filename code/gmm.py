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

    def get_covariances(self):
        variances = torch.exp(self.log_vars) + 1e-6
        return torch.diag_embed(variances)

    def forward(self, x):
        weights_log_pdf = torch.log_softmax(self.pi, dim=-1)
        variances = torch.exp(self.log_vars) + 1e-6
        log_probs = dist.Normal(self.means, variances.sqrt()).log_prob(x.unsqueeze(1)).sum(dim=-1)

        weighted_log_probs = log_probs + weights_log_pdf
        return torch.logsumexp(weighted_log_probs, dim=1)

    def characteristic_function(self, s):
        weights = torch.softmax(self.pi, dim=-1)
        variances = torch.exp(self.log_vars) + 1e-6 # [K, D]
        
        linear_term = torch.matmul(s, self.means.T)
        quad_term = torch.matmul(s**2, variances.T)
        phi_components = torch.exp(1j * linear_term - 0.5 * quad_term)

        return torch.sum(weights * phi_components, dim=-1)

    def sample(self, num_samples):
        with torch.no_grad():
            weights = torch.softmax(self.pi, dim=-1)
            cat_dist = dist.Categorical(weights)
            indices = cat_dist.sample((num_samples,))

            eps = torch.randn(num_samples, self.n_features, device=self.device)
            
            m = self.means[indices]
            s = torch.exp(0.5 * self.log_vars[indices])
            
            return m + eps * s

    def fit(self, loader, epochs=50, lr=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            for batch in loader:
                if isinstance(batch, (tuple, list)): batch = batch[0]
                batch = batch.to(self.device)

                loss = -torch.mean(self(batch))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()