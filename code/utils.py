import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

def plot_2D(*fns, title=None, xmin=-2.5, xmax=2.5, nbins=15):

    x_min, x_max = xmin, xmax
    y_min, y_max = xmin, xmax

    dx, dy = 0.01, 0.01

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(x_min, x_max + dy, dy),
                    slice(y_min, y_max + dx, dx)]
    xy = torch.from_numpy(np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))).float()

    fns = [fn for fn in fns if fn is not None]

    ncols = len(fns)
    if ncols == 0:
        return

    fig, axs = plt.subplots(ncols=ncols, figsize=(5*ncols, 5))
    
    if ncols == 1:
        axs = [axs]

    for fn, ax in zip(fns, axs):
        with torch.no_grad():
            z = fn(xy)
        z = z.cpu().view(y.shape).numpy()
        z = z[:-1, :-1]
        
        cmap = plt.colormaps['PiYG']
        
        levels = MaxNLocator(nbins=nbins).tick_values(z.min(), z.max())
        cf = ax.contourf(
            x[:-1, :-1] + dx/2.,
            y[:-1, :-1] + dy/2., 
            z, 
            levels=levels, cmap=cmap
        )
        ax.set_aspect('equal', 'box')
        
    fig.colorbar(cf, ax=axs)
    if title is not None:
        if ncols == 1:
            axs[-1].set_title(title)
        else:
            fig.suptitle(title)

    filename = title.replace(' ', '_').replace('-', '_') + '.png' if title else 'plot_2D.png'
    plt.savefig(f'figures/{filename}')
    plt.close()

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

def plot_gmm_results(gmm_model, X, title=None):
    # Predict cluster assignments
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(gmm_model.device)
        log_likelihoods = torch.zeros(X_tensor.size(0), gmm_model.K, device=X_tensor.device)
        for k in range(gmm_model.K):
            lower_triangular = torch.tril(gmm_model.chol_var[k])
            var_k = lower_triangular @ lower_triangular.t() + 1e-6 * torch.eye(gmm_model.n_features, device=X_tensor.device)
            log_probs = torch.distributions.MultivariateNormal(gmm_model.means[k], var_k).log_prob(X_tensor)
            log_likelihoods[:, k] = log_probs
        weighted_log_likelihoods = log_likelihoods + torch.log_softmax(gmm_model.pi, dim=-1)
        clusters = torch.argmax(weighted_log_likelihoods, dim=1).cpu().numpy()

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    if title:
        plt.title(title)
    plt.axis('equal')
    filename = title.replace(' ', '_') + '_clusters.png' if title else 'gmm_clusters.png'
    plt.savefig(f'figures/{filename}')
    plt.close()

    # Plot GMM density contours
    def gmm_density(xy):
        with torch.no_grad():
            return torch.exp(gmm_model(xy.to(gmm_model.device)))

    plot_2D(gmm_density, title=f'GMM Density Contours - {title}' if title else 'GMM Density Contours')

def plot_logistic_correctness(clf, X, y, title=None):
    y_pred = clf.predict(X)
    correctness = (y == y_pred).astype(int)  # 1 for correct, 0 for incorrect

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=correctness, cmap='RdYlGn', edgecolor='k', alpha=0.7)
    plt.colorbar(scatter, label='Correctly Classified (1=Yes, 0=No)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    if title:
        plt.title(title)
    plt.axis('equal')
    filename = title.replace(' ', '_') + '.png' if title else 'logistic_correctness.png'
    plt.savefig(f'figures/{filename}')
    plt.close()

def plot_logistic_probabilities(clf, X, y, title=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu_r', levels=20)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='True Label')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    if title:
        plt.title(title)
    plt.axis('equal')
    filename = title.replace(' ', '_') + '.png' if title else 'logistic_probabilities.png'
    plt.savefig(f'figures/{filename}')
    plt.close()