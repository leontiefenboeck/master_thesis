import matplotlib.pyplot as plt
import torch
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

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

def plot_convergence(results, sample_counts, filename='convergence.png', ylim=None, title=None):
    _MARKERS = ['o', 's', '^', 'D', 'P', 'X']
    xs = np.array(sample_counts)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, data) in enumerate(results.items()):
        means = np.array(data['means'])
        stds  = np.array(data['stds'])
        line, = ax.plot(xs, means, marker=_MARKERS[i % len(_MARKERS)], markersize=7,
                        linewidth=2, label=name, zorder=3)
        ax.fill_between(xs, means - stds, means + stds,
                        alpha=0.12, color=line.get_color(), zorder=2)

    ax.set_xscale('log')
    ax.set_xlabel('Total budget $N$', fontsize=12)
    ax.set_ylabel(r'$\mathbb{E}[\sigma(w^\top x) \mid x_\mathrm{obs}]$', fontsize=12)
    ax.set_title(title or r'Estimator convergence  (mean $\pm$ 1 std over runs)', fontsize=13)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(framealpha=0.95, fontsize=11)
    ax.grid(True, which='major', alpha=0.25)
    ax.grid(True, which='minor', alpha=0.10, linestyle=':')
    ax.tick_params(labelsize=10)

    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=150, bbox_inches='tight')
    plt.close()

def print_variance_components(sigma2_mc, sigma2_B, sigma2_A, pcpg11_vals):
    arr = np.asarray(pcpg11_vals)
    print(f"σ²_MC = {sigma2_mc:.4f}  (bound: 0.25)")
    print(f"σ²_B  = {sigma2_B:.4f}  (outer PG, finite)")
    print(f"σ²_A  = {sigma2_A:.4f}  (inner Gaussian, ∞ in theory)")
    print(f"PCPG(1,1) range: [{arr.min():.3f}, {arr.max():.3f}]  "
          f"({(arr < 0).mean():.1%} < 0,  {(arr > 1).mean():.1%} > 1)")

def plot_variance_analysis(sigma2_mc, sigma2_B, sigma2_A, budgets, results,
                           n_gauss_quad=30, pcpg_ratio=1,
                           filename='variance_analysis.png', title=None):
    N = np.logspace(np.log10(budgets[0]), np.log10(budgets[-1]), 300)
    var_mc_formula   = sigma2_mc / N
    var_pcpg_formula = 0.25 * (sigma2_B * np.sqrt(pcpg_ratio) / np.sqrt(N) + sigma2_A / N)
    var_gh_formula   = 0.25 * sigma2_B * n_gauss_quad / N

    budgets_arr    = np.array(budgets)
    var_mc_meas    = np.array(results['MC']['stds'])      ** 2
    var_pcpg_meas  = np.array(results['PCPG']['stds'])    ** 2
    var_gh_meas    = np.array(results['PCPG-GH']['stds']) ** 2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(N, var_mc_formula,   color='#2196F3', linestyle='--')
    ax.loglog(N, var_pcpg_formula, color='#FF7043', linestyle='--')
    ax.loglog(N, var_gh_formula,   color='#43A047', linestyle='--')
    ax.loglog(budgets_arr, var_mc_meas,   'o', color='#2196F3', markersize=6, label='MC')
    ax.loglog(budgets_arr, var_pcpg_meas, 's', color='#FF7043', markersize=6, label='PCPG')
    ax.loglog(budgets_arr, var_gh_meas,   '^', color='#43A047', markersize=6, label='PCPG-GH')

    # proxy for dashed "formula" entry
    ax.plot([], [], 'k--', linewidth=1.5, label='formula')

    ax.set_xlabel('Total budget $N$', fontsize=12)
    ax.set_ylabel('Variance', fontsize=12)
    ax.set_title(title or r'Formula vs measured variance', fontsize=13)
    ax.legend(fontsize=10, framealpha=0.95, loc='upper right')
    ax.grid(True, which='major', alpha=0.25)
    ax.grid(True, which='minor', alpha=0.10, linestyle=':')
    ax.tick_params(labelsize=10)

    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
