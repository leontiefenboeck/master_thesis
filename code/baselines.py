import torch
import torch.distributions as dist

def _gmm_params(gmm):
    log_pi = torch.log_softmax(gmm.pi, dim=-1)          # (K,)
    covs = []
    for k in range(gmm.K):
        L = torch.tril(gmm.chol_var[k])
        covs.append(L @ L.t() + 1e-6 * torch.eye(gmm.n_features, device=gmm.device))
    return log_pi, gmm.means, torch.stack(covs)          # (K,), (K,D), (K,D,D)


def _component_posterior(x_obs, obs_idx, log_pi, means, covs):
    log_post = log_pi.clone()
    for k in range(len(log_pi)):
        mvn = dist.MultivariateNormal(means[k][obs_idx], covs[k][obs_idx][:, obs_idx])
        log_post[k] += mvn.log_prob(x_obs)
    return (log_post - log_post.logsumexp(0)).exp()      # (K,)


def _conditional_gaussian(k, x_obs, obs_idx, mis_idx, means, covs):
    """p(x_miss | x_obs, component k) via Schur complement."""
    mu, cov = means[k], covs[k]
    cov_oo_inv = torch.linalg.inv(cov[obs_idx][:, obs_idx])
    gain       = cov[mis_idx][:, obs_idx] @ cov_oo_inv
    cond_mean  = mu[mis_idx] + gain @ (x_obs - mu[obs_idx])
    cond_cov   = cov[mis_idx][:, mis_idx] - gain @ cov[obs_idx][:, mis_idx]
    cond_cov   = (cond_cov + cond_cov.t()) / 2 + 1e-6 * torch.eye(len(mis_idx), device=mu.device)
    return cond_mean, cond_cov


class MCBaseline:
    def __init__(self, gmm, n_samples=2000):
        self.gmm = gmm
        self.n_samples = n_samples

    @torch.no_grad()
    def pred(self, x_test, missing_mask, w, b=0.0):
        """x_test: (N,D), missing_mask: (N,D) bool. Returns (N,) expected sigmoid."""
        log_pi, means, covs = _gmm_params(self.gmm)
        probs = torch.zeros(len(x_test), device=self.gmm.device)

        for i, (x, mask) in enumerate(zip(x_test, missing_mask)):
            obs_idx = (~mask).nonzero(as_tuple=True)[0]
            mis_idx =   mask .nonzero(as_tuple=True)[0]

            if len(mis_idx) == 0:           # nothing missing
                probs[i] = torch.sigmoid(w @ x + b)
                continue

            post = _component_posterior(x[obs_idx], obs_idx, log_pi, means, covs)
            k_samples = torch.multinomial(post, self.n_samples, replacement=True)

            x_full = x.unsqueeze(0).repeat(self.n_samples, 1)
            for k in range(self.gmm.K):
                sel = (k_samples == k).nonzero(as_tuple=True)[0]
                if len(sel) == 0:
                    continue
                cm, cc = _conditional_gaussian(k, x[obs_idx], obs_idx, mis_idx, means, covs)
                x_full[sel[:, None], mis_idx] = dist.MultivariateNormal(cm, cc).sample((len(sel),))

            probs[i] = torch.sigmoid(x_full @ w + b).mean()

        return probs


class MeanImputationBaseline:
    def __init__(self, train_means):
        self.train_means = torch.tensor(train_means, dtype=torch.float32) if not isinstance(train_means, torch.Tensor) else train_means

    def pred(self, x_test, missing_mask, w, b=0.0):
        means = self.train_means.to(x_test.device)
        x_imp = x_test.clone()
        x_imp[missing_mask] = means.expand_as(x_test)[missing_mask]
        return torch.sigmoid(x_imp @ w + b)
