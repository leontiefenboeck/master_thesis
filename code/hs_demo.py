"""
Hubbard-Stratonovich trick demo
================================

Core identity:
    e^{-ω t²/2}  =  E_{u ~ N(0, ω)}[ e^{iut} ]           (HS)

The right-hand side is the characteristic function (CF) of N(0, ω) evaluated
at t.  This replaces a quadratic function of t with a linear phase, which is
useful when t = wᵀx_mis is unknown: the expectation over x_mis can then be
pulled inside and identified as the CF of p(x_mis | x_obs):

    E_{x_mis}[ e^{-ω(wᵀx_mis)²/2} ]
        = E_{u ~ N(0,ω)}[ E_{x_mis}[ e^{iu wᵀx_mis} ] ]
        = E_{u ~ N(0,ω)}[ φ(u w_mis) ]

where φ(s) = E[e^{is x_mis}].  No samples of x_mis needed on the right.

This file demonstrates:
  1.  The HS identity itself: direct computation vs MC vs Gauss-Hermite
  2.  The decoupling application:
        E_x[ e^{-ω(wx)²/2} ]  via direct MC over x  vs  CF-based trick
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

OMEGA = 1.5   # fixed ω used throughout

def hs_true(t_vals):
    """Left-hand side: exact."""
    return np.exp(-OMEGA * t_vals**2 / 2)


def hs_mc(t_vals, n_samples, rng):
    """Right-hand side via MC: mean of Re[e^{iut}] over u ~ N(0, ω)."""
    u = rng.normal(0, np.sqrt(OMEGA), n_samples)          # (N,)
    return np.cos(np.outer(u, t_vals)).mean(axis=0)        # Re[e^{iut}] = cos(ut)


def hs_hermite(t_vals, n_nodes):
    """Right-hand side via Gauss-Hermite quadrature.

    Change of variables: u = √(2ω) · z, du = √(2ω) dz
      ∫ e^{iut} N(u; 0, ω) du = (1/√π) ∫ e^{i√(2ω) z t} e^{-z²} dz
                               ≈ (1/√π) Σ_k w_k e^{i√(2ω) z_k t}
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    u_nodes = np.sqrt(2 * OMEGA) * nodes                   # (n_nodes,)
    # outer product: phase[k, j] = u_nodes[k] * t_vals[j]
    phases  = np.outer(u_nodes, t_vals)                    # (n_nodes, len(t))
    return (weights[:, None] * np.cos(phases)).sum(axis=0) / np.sqrt(np.pi)

W        = 1.2
SIGMA_X  = 0.8

def app_true(mu_vals):
    """Closed-form E[e^{-ω(wx)²/2}] for x ~ N(μ, σ²)."""
    # completing the square: N(x; μ, σ²) · e^{-ω w² x²/2}
    # = Z · N(x; μ/(1+ωw²σ²), σ²/(1+ωw²σ²))
    # where Z = exp(-ω w² μ² / (2(1+ω w²σ²))) / sqrt(1 + ω w²σ²)
    denom = 1 + OMEGA * W**2 * SIGMA_X**2
    return np.exp(-OMEGA * W**2 * mu_vals**2 / (2 * denom)) / np.sqrt(denom)


def app_mc(mu_vals, n_samples, rng):
    """Direct MC over x ~ N(μ, σ²)."""
    out = np.empty(len(mu_vals))
    for i, mu in enumerate(mu_vals):
        x = rng.normal(mu, SIGMA_X, n_samples)
        out[i] = np.exp(-OMEGA * (W * x)**2 / 2).mean()
    return out


def app_hs_mc(mu_vals, n_samples, rng):
    """CF-based trick: E_u[φ_x(uw)] where φ_x(s) = e^{iμs - σ²s²/2}."""
    u = rng.normal(0, np.sqrt(OMEGA), n_samples)           # u ~ N(0, ω)
    s = u * W                                               # frequencies: (N,)
    # φ_x(s; μ) = e^{iμs - σ²s²/2}  →  Re part for the real integral
    gauss_factor = np.exp(-0.5 * SIGMA_X**2 * s**2)        # (N,)  real, ≥ 0
    # E_u[Re[e^{iμs}] · gauss_factor] = E_u[cos(μs) · gauss_factor]
    out = np.empty(len(mu_vals))
    for i, mu in enumerate(mu_vals):
        out[i] = (np.cos(mu * s) * gauss_factor).mean()
    return out


def app_hs_hermite(mu_vals, n_nodes):
    """CF-based trick, u integral via Gauss-Hermite quadrature."""
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    u_nodes = np.sqrt(2 * OMEGA) * nodes
    s       = u_nodes * W
    gauss_factor = np.exp(-0.5 * SIGMA_X**2 * s**2)        # (n_nodes,)
    out = np.empty(len(mu_vals))
    for i, mu in enumerate(mu_vals):
        integrand = np.cos(mu * s) * gauss_factor
        out[i]    = (weights * integrand).sum() / np.sqrt(np.pi)
    return out


def _build_data(mc_ns, gh_ns, n_runs=5):
    t_vals  = np.linspace(-3, 3, 200)
    mu_vals = np.linspace(-3, 3, 200)
    rng     = np.random.default_rng(0)

    hs_exact = hs_true(t_vals)
    app_exact = app_true(mu_vals)

    hs_mc_curves  = {n: [hs_mc(t_vals, n, rng)      for _ in range(n_runs)] for n in mc_ns}
    hs_gh_curves  = {n: hs_hermite(t_vals, n)                               for n in gh_ns}
    app_mc_curves = {n: [app_mc(mu_vals, n, rng)     for _ in range(n_runs)] for n in mc_ns}
    app_hs_curves = {n: [app_hs_mc(mu_vals, n, rng)  for _ in range(n_runs)] for n in mc_ns}
    app_gh_curves = {n: app_hs_hermite(mu_vals, n)                           for n in gh_ns}

    return (t_vals, mu_vals, hs_exact, app_exact,
            hs_mc_curves, hs_gh_curves,
            app_mc_curves, app_hs_curves, app_gh_curves)


def plot_hs_identity(mc_ns, gh_ns, data=None,
                     out_path="figures/hs_identity.pdf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    (t_vals, _, hs_exact, _, hs_mc_curves, hs_gh_curves, *_) = (
        data or _build_data(mc_ns, gh_ns)
    )

    colors_mc = plt.cm.Blues( np.linspace(0.35, 0.90, len(mc_ns)))
    colors_gh = plt.cm.Reds(  np.linspace(0.35, 0.90, len(gh_ns)))

    fig, (ax_fn, ax_err) = plt.subplots(1, 2, figsize=(11, 4))

    # function panel
    ax_fn.plot(t_vals, hs_exact, color="0.15", lw=2, zorder=10, label="exact $e^{-\\omega t^2/2}$")
    for color, n in zip(colors_mc, mc_ns):
        for i, curve in enumerate(hs_mc_curves[n]):
            ax_fn.plot(t_vals, curve, color=color, alpha=0.45, lw=0.9,
                       label=f"MC  $N={n}$" if i == 0 else None)
    for color, n in zip(colors_gh, gh_ns):
        ax_fn.plot(t_vals, hs_gh_curves[n], color=color, lw=1.8,
                   label=f"GH  $n={n}$")

    # error panel
    for color, n in zip(colors_mc, mc_ns):
        for i, curve in enumerate(hs_mc_curves[n]):
            ax_err.plot(t_vals, np.abs(curve - hs_exact), color=color, alpha=0.45, lw=0.9,
                        label=f"MC  $N={n}$" if i == 0 else None)
    for color, n in zip(colors_gh, gh_ns):
        ax_err.plot(t_vals, np.abs(hs_gh_curves[n] - hs_exact), color=color, lw=1.8,
                    label=f"GH  $n={n}$")

    ax_fn.set_xlabel("$t$", fontsize=11)
    ax_fn.set_ylabel(r"$e^{-\omega t^2/2}$", fontsize=11)
    ax_fn.set_title(r"HS identity:  $e^{-\omega t^2/2} = \mathbb{E}_{u\sim\mathcal{N}(0,\omega)}[e^{iut}]$",
                    fontsize=11)
    ax_fn.legend(fontsize=8)

    ax_err.set_xlabel("$t$", fontsize=11)
    ax_err.set_ylabel("|error|", fontsize=11)
    ax_err.set_yscale("log")
    ax_err.set_title("Absolute error", fontsize=11)
    ax_err.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved -> {out_path}")


def plot_hs_application(mc_ns, gh_ns, data=None,
                        out_path="figures/hs_application.pdf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    (_, mu_vals, _, app_exact, _, _,
     app_mc_curves, app_hs_curves, app_gh_curves) = (
        data or _build_data(mc_ns, gh_ns)
    )

    colors_mc = plt.cm.Blues(  np.linspace(0.35, 0.90, len(mc_ns)))
    colors_hs = plt.cm.Oranges(np.linspace(0.35, 0.90, len(mc_ns)))
    colors_gh = plt.cm.Greens( np.linspace(0.35, 0.90, len(gh_ns)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)

    for ax in axes:
        ax.plot(mu_vals, app_exact, color="0.15", lw=2, zorder=10, label="exact")
        ax.set_xlabel(r"$\mu$  (mean of $x$)", fontsize=11)

    for color, n in zip(colors_mc, mc_ns):
        for i, curve in enumerate(app_mc_curves[n]):
            axes[0].plot(mu_vals, curve, color=color, alpha=0.45, lw=0.9,
                         label=f"$N={n}$" if i == 0 else None)
    for color, n in zip(colors_hs, mc_ns):
        for i, curve in enumerate(app_hs_curves[n]):
            axes[1].plot(mu_vals, curve, color=color, alpha=0.45, lw=0.9,
                         label=f"$N={n}$" if i == 0 else None)
    for color, n in zip(colors_gh, gh_ns):
        axes[2].plot(mu_vals, app_gh_curves[n], color=color, lw=1.8, label=f"$n={n}$")

    axes[0].set_title(r"Direct MC over $x$", fontsize=11)
    axes[1].set_title(r"HS trick: MC over $u$, CF in closed form", fontsize=11)
    axes[2].set_title(r"HS trick: Gauss-Hermite over $u$", fontsize=11)
    for ax in axes:
        ax.set_ylabel(r"$\mathbb{E}_x[e^{-\omega(wx)^2/2}]$", fontsize=11)
        ax.legend(fontsize=8)

    fig.suptitle(
        r"Application: $\mathbb{E}_x[e^{-\omega(wx)^2/2}]$ — direct MC vs HS trick",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    mc_ns = [10, 100, 1000]
    gh_ns = [3, 7, 15]

    data = _build_data(mc_ns, gh_ns, n_runs=5)
    plot_hs_identity(mc_ns, gh_ns, data=data)
    plot_hs_application(mc_ns, gh_ns, data=data)
