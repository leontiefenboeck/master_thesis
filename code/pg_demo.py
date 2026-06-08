import numpy as np
from polyagamma import random_polyagamma
import matplotlib.pyplot as plt
import os
from pcpg import pg_quadrature

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _mc_estimate(x_vals, pg_samples):
    out = []
    for x in x_vals:
        integrand = np.exp(-pg_samples * x**2 / 2)
        out.append(0.5 * np.exp(x / 2) * integrand.mean())
    return np.array(out)

def _quad_estimate(x_vals, nodes, weights):
    out = []
    for x in x_vals:
        integrand = np.exp(-nodes * x**2 / 2)
        out.append(0.5 * np.exp(x / 2) * (weights * integrand).sum())
    return np.array(out)

def _build_convergence_data(mc_ns, quad_ns, n_runs=1):
    x     = np.linspace(-6, 6, 300)
    truth = sigmoid(x)
    rng   = np.random.default_rng(0)

    mc_curves = {}
    for n in mc_ns:
        mc_curves[n] = []
        for _ in range(n_runs):
            samples = random_polyagamma(1.0, np.zeros(n), random_state=rng)
            est = _mc_estimate(x, samples)
            mc_curves[n].append(est)

    quad_curves = {}
    for n in quad_ns:
        nodes_n, weights_n = pg_quadrature(n)
        quad_curves[n] = _quad_estimate(x, nodes_n, weights_n)

    return x, truth, mc_curves, quad_curves


def plot_convergence_functions(mc_ns, quad_ns, data=None, out_path="figures/pg_convergence_functions.pdf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x, truth, mc_curves, quad_curves = data or _build_convergence_data(mc_ns, quad_ns)

    colors_mc   = plt.cm.Blues(np.linspace(0.35, 0.9, len(mc_ns)))
    colors_quad = plt.cm.Reds( np.linspace(0.35, 0.9, len(quad_ns)))

    fig, (ax_mc, ax_quad) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax in (ax_mc, ax_quad):
        ax.plot(x, truth, color="0.2", lw=1.5, zorder=10, label="exact")
        ax.set_xlabel("$x$")
        ax.set_ylim(-0.05, 1.15)

    for color, n in zip(colors_mc, mc_ns):
        for i, est in enumerate(mc_curves[n]):
            ax_mc.plot(x, est, color=color, alpha=0.4, lw=0.9,
                       label=f"$N={n}$" if i == 0 else None)

    for color, n in zip(colors_quad, quad_ns):
        ax_quad.plot(x, quad_curves[n], color=color, lw=1.8, label=f"$n={n}$")

    ax_mc.set_title("MC sampling")
    ax_mc.set_ylabel("$\\sigma(x)$")
    ax_mc.legend(fontsize=8, title="samples $N$")
    ax_quad.set_title("Gauss-PG quadrature")
    ax_quad.legend(fontsize=8, title="nodes $n$")

    fig.tight_layout()
    fig.savefig(out_path)


def plot_convergence_errors(mc_ns, quad_ns, data=None, out_path="figures/pg_convergence_errors.pdf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x, truth, mc_curves, quad_curves = data or _build_convergence_data(mc_ns, quad_ns)

    colors_mc   = plt.cm.Blues(np.linspace(0.35, 0.9, len(mc_ns)))
    colors_quad = plt.cm.Reds( np.linspace(0.35, 0.9, len(quad_ns)))

    fig, (ax_mc, ax_quad) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for color, n in zip(colors_mc, mc_ns):
        for i, est in enumerate(mc_curves[n]):
            ax_mc.plot(x, np.abs(est - truth), color=color, alpha=0.4, lw=0.9,
                       label=f"$N={n}$" if i == 0 else None)

    for color, n in zip(colors_quad, quad_ns):
        ax_quad.plot(x, np.abs(quad_curves[n] - truth), color=color, lw=1.8,
                     label=f"$n={n}$")

    for ax in (ax_mc, ax_quad):
        ax.set_xlabel("$x$")
        ax.set_yscale("log")

    ax_mc.set_title("MC sampling")
    ax_mc.set_ylabel("|error|")
    ax_mc.legend(fontsize=8, title="samples $N$")
    ax_quad.set_title("Gauss-PG quadrature")
    ax_quad.legend(fontsize=8, title="nodes $n$")

    fig.tight_layout()
    fig.savefig(out_path)

if __name__ == "__main__":
    mc_ns   = [5, 10, 100, 500]
    quad_ns = [3, 5, 10, 15, 20]
    data = _build_convergence_data(mc_ns, quad_ns)
    plot_convergence_functions(mc_ns, quad_ns, data=data)
    plot_convergence_errors(mc_ns, quad_ns, data=data)
