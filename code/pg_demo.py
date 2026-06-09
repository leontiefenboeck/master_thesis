import numpy as np
from polyagamma import random_polyagamma
import matplotlib.pyplot as plt
import os
from pcpg import pg_gauss_quadrature as pg_quadrature, pg_nnls_quadrature

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

def _pg_pdf(omega, n_terms=50):
    """PG(1,0) density via the alternating-series representation:
       p(ω) = (π/2) Σ_{k=0}^∞ (-1)^k (2k+1) e^{-(2k+1)²π²ω/8}
    """
    omega = np.asarray(omega, dtype=float)
    result = np.zeros_like(omega)
    for k in range(n_terms):
        coeff = (-1)**k * (2*k + 1)
        result += coeff * np.exp(-(2*k + 1)**2 * np.pi**2 * omega / 8)
    return (np.pi / 2) * result


def plot_pg_distribution(n_samples=50_000, out_path="figures/pg_distribution.pdf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rng     = np.random.default_rng(0)
    samples = random_polyagamma(1.0, 0.0, size=n_samples, random_state=rng)

    omega   = np.linspace(0.02, 2.5, 500)
    pdf     = _pg_pdf(omega)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(samples, bins=200, density=True, color="#90CAF9", label=f"samples ($N={n_samples:,}$)")
    ax.plot(omega, pdf, color="#1565C0", lw=2, label="exact PDF")

    ax.set_xlabel(r"$\omega$", fontsize=12)
    ax.set_ylabel("density", fontsize=12)
    ax.set_title(r"Pólya-Gamma distribution  $\mathrm{PG}(1,\,0)$", fontsize=13)
    ax.set_xlim(0, 2.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")

def plot_pg_samples(ns=(5, 20, 100, 500), out_path="figures/pg_samples.pdf"):
    """Histogram of PG(1,0) draws at increasing N, overlaid with the exact PDF."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rng   = np.random.default_rng(0)
    omega = np.linspace(0.02, 2.5, 500)
    pdf   = _pg_pdf(omega)

    fig, axes = plt.subplots(1, len(ns), figsize=(3.5 * len(ns), 4), sharey=True)

    for ax, n in zip(axes, ns):
        samples = random_polyagamma(1.0, 0.0, size=n, random_state=rng)

        ax.hist(samples, bins=max(10, n // 5), density=True,
                color="#90CAF9", alpha=0.8, label=f"{n} draws")
        ax.plot(omega, pdf, color="#1565C0", lw=2, label="exact PDF")

        ax.set_xlim(0, 2.5)
        ax.set_xlabel(r"$\omega$", fontsize=12)
        ax.set_title(f"$N = {n}$", fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.20)

    axes[0].set_ylabel("density", fontsize=12)
    fig.suptitle(r"Draws from $\mathrm{PG}(1,\,0)$ via \texttt{random\_polyagamma}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")

def plot_quadrature_comparison(quad_ns, out_path="figures/pg_quadrature_comparison.pdf"):
    """Two-panel comparison: approximated σ(x) and pointwise error, GW vs NNLS."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    x     = np.linspace(-6, 6, 300)
    truth = sigmoid(x)

    colors_gw = plt.cm.Reds(np.linspace(0.4, 0.9, len(quad_ns)))

    nodes_nnls, weights_nnls = pg_nnls_quadrature()
    n_out    = len(nodes_nnls)
    est_nnls = _quad_estimate(x, nodes_nnls, weights_nnls)
    nnls_label = f"NNLS ({n_out} nodes)"

    fig, (ax_fn, ax_err) = plt.subplots(1, 2, figsize=(11, 4))

    ax_fn.plot(x, truth, color="0.2", lw=1.5, zorder=10, label="exact")
    for color, n in zip(colors_gw, quad_ns):
        nodes, weights = pg_quadrature(n)
        est = _quad_estimate(x, nodes, weights)
        ax_fn.plot(x, est, color=color, lw=1.6, label=f"GW $n={n}$")
    ax_fn.plot(x, est_nnls, color="#2E7D32", lw=2.0, ls="--", label=nnls_label)
    ax_fn.set_xlabel("$x$")
    ax_fn.set_ylabel("$\\sigma(x)$")
    ax_fn.set_ylim(-0.05, 1.15)
    ax_fn.legend(fontsize=8)

    for color, n in zip(colors_gw, quad_ns):
        nodes, weights = pg_quadrature(n)
        est = _quad_estimate(x, nodes, weights)
        ax_err.plot(x, np.abs(est - truth), color=color, lw=1.6, label=f"GW $n={n}$")
    ax_err.plot(x, np.abs(est_nnls - truth), color="#2E7D32", lw=2.0, ls="--",
                label=nnls_label)
    ax_err.set_xlabel("$x$")
    ax_err.set_ylabel("|error|")
    ax_err.set_yscale("log")
    ax_err.legend(fontsize=8)

    fig.suptitle("PG quadrature: Golub–Welsch vs NNLS", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")

if __name__ == "__main__":
    mc_ns   = [5, 10, 100, 500]
    quad_ns = [3, 5, 10, 15, 20]
    # data = _build_convergence_data(mc_ns, quad_ns)
    # plot_convergence_functions(mc_ns, quad_ns, data=data)
    # plot_convergence_errors(mc_ns, quad_ns, data=data)
    # plot_pg_distribution()
    # plot_pg_samples()
    plot_quadrature_comparison([5, 10, 19])
