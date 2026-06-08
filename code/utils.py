import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import time

os.makedirs('figures', exist_ok=True)

def evaluate(name, method, w, x_partial, n_runs=10):
    vals, times = [], []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        vals.append(method(w, x_partial))
        times.append(time.perf_counter() - t0)
    mean, std = np.mean(vals), np.std(vals)
    mean_t = np.mean(times) * 1000
    print(f"{name:<10}  p(y=1|x_obs) = {mean:.4f} ± {std:.4f}   ({mean_t:.1f} ms/run) (num runs = {n_runs})")

def plot_data(X, y=None, title=None, xlabel='Feature 1', ylabel='Feature 2'):
    plt.figure(figsize=(6, 6))
    if y is not None:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Label')
    else:
        plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.axis('equal')
    filename = title.replace(' ', '_') + '.png' if title else 'plot_data.png'
    plt.savefig(f'figures/{filename}')
    plt.close()

def plot_gmm_results(gmm_model, X, title=None, num_samples=1000):
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(gmm_model.device)
        variances = torch.exp(gmm_model.log_vars) + 1e-6
        log_probs = torch.distributions.Normal(gmm_model.means, variances.sqrt()).log_prob(X_tensor.unsqueeze(1)).sum(dim=-1)
        weighted_log_probs = log_probs + torch.log_softmax(gmm_model.pi, dim=-1)
        clusters = torch.argmax(weighted_log_probs, dim=1).cpu().numpy()

    samples = gmm_model.sample(num_samples).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].scatter(X[:, 0], X[:, 1], c=clusters, cmap='tab10', alpha=0.5, s=15)
    axes[0].set_title(f"Real Data: Learned Clusters\n({title if title else ''})")
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].axis('equal')

    axes[1].scatter(samples[:, 0], samples[:, 1], color='crimson', alpha=0.4, s=15)
    axes[1].set_title(f"Generated Samples\n(What the GMM 'sees')")
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].axis('equal')

    os.makedirs('figures', exist_ok=True)
    filename = title.replace(' ', '_') + '_combined.png' if title else 'gmm_combined.png'
    plt.tight_layout()
    plt.savefig(f'figures/{filename}')
    plt.close()

def plot_classifier_results(clf, X, y, title="Logistic Regression"):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid_points)[:, 1].reshape(xx.shape)

    y_pred = clf.predict(X)
    correct = (y_pred == y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    contour = axes[0].contourf(xx, yy, probs, alpha=0.8, cmap='RdYlBu_r', levels=20)
    axes[0].scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis', s=30, alpha=0.8)
    axes[0].set_title(f"{title}: Probabilities")
    fig.colorbar(contour, ax=axes[0], label='P(class=1)')

    axes[1].contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)

    axes[1].scatter(X[correct, 0], X[correct, 1], c='green', label='Correct', alpha=0.6, s=25)
    axes[1].scatter(X[~correct, 0], X[~correct, 1], c='red', label='Incorrect', marker='x', s=50)

    axes[1].set_title(f"{title}: Hits vs Misses")
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')

    os.makedirs('figures', exist_ok=True)
    filename = f"{title.lower().replace(' ', '_')}_results.png"
    plt.tight_layout()
    plt.savefig(f'figures/{filename}')
    plt.close()

def plot_convergence(results, sample_counts, filename='convergence.png'):
    colors = {'MC': 'steelblue', 'PCPG': 'darkorange', 'PCPG-GH': 'forestgreen'}
    sample_counts_arr = np.array(sample_counts)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, data in results.items():
        means = np.array(data['means'])
        stds  = np.array(data['stds'])
        color = colors.get(name, None)
        ax.plot(sample_counts_arr, means, marker='o', label=name, color=color)
        ax.fill_between(sample_counts_arr, means - stds, means + stds, alpha=0.2, color=color)

    ax.set_xscale('log')
    ax.set_xlabel('Total budget $N$')
    ax.set_ylabel(r'$\mathbb{E}[\sigma(w^\top x) \mid x_\mathrm{obs}]$')
    ax.set_title('Convergence: MC vs PCPG  (mean ± 1 std)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved -> figures/{filename}")

def plot_variance_analysis(sigma2_mc, sigma2_B, sigma2_A, budgets, results,
                           filename='variance_analysis.png'):
    N = np.logspace(np.log10(budgets[0]), np.log10(budgets[-1]), 300)
    var_mc_pred   = sigma2_mc / N
    var_pcpg_pred = 0.25 * (sigma2_B / np.sqrt(N) + sigma2_A / N)

    budgets_arr        = np.array(budgets)
    var_mc_empirical   = np.array(results['MC']['stds'])   ** 2
    var_pcpg_empirical = np.array(results['PCPG']['stds']) ** 2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(N, var_mc_pred,   color='steelblue', linestyle='-',
              label=r'MC analytical: $\sigma^2_{MC}/N$')
    ax.loglog(N, var_pcpg_pred, color='darkorange', linestyle='-',
              label=r'PCPG analytical: $\frac{1}{4}\!\left(\frac{\sigma^2_B}{\sqrt{N}}+\frac{\sigma^2_A}{N}\right)$')
    ax.loglog(budgets_arr, var_mc_empirical,   'o', color='steelblue',  markersize=6, label='MC empirical')
    ax.loglog(budgets_arr, var_pcpg_empirical, 'o', color='darkorange', markersize=6, label='PCPG empirical')

    ax.set_xlabel('Total budget $N$')
    ax.set_ylabel('Variance')
    ax.set_title(r'Analytical vs empirical variance  ($n_{pg}=n_{gauss}=\sqrt{N}$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved -> figures/{filename}")
