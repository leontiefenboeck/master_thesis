import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

A = 1.5  # parameter a in the identity

def lhs(h_vals): # exact
    return np.exp(-A * h_vals**2 / 2)

def rhs_mc(h_vals, n_samples, rng): # mc of right side of the identity
    u      = rng.normal(0, np.sqrt(A), n_samples)  
    phases = np.outer(u, h_vals)
    return np.exp(-1j * phases).mean(axis=0).real

def rhs_hermite(h_vals, n_nodes): # gauss-hermite quadrature of right side of the identity
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    u_nodes = np.sqrt(2 * A) * nodes               
    phases  = np.outer(u_nodes, h_vals)
    return (weights[:, None] * np.cos(phases)).sum(axis=0) / np.sqrt(np.pi)

def plot_hs_identity(mc_ns, gh_ns, n_runs=5, out_path="figures/hs_identity.pdf"):
    h_vals = np.linspace(-3, 3, 300)
    exact  = lhs(h_vals)
    rng    = np.random.default_rng(0)

    colors_mc = plt.cm.Blues(np.linspace(0.35, 0.90, len(mc_ns)))
    colors_gh = plt.cm.Reds( np.linspace(0.35, 0.90, len(gh_ns)))

    fig, (ax_fn, ax_err) = plt.subplots(1, 2, figsize=(11, 4))

    ax_fn.plot(h_vals, exact, color="0.15", lw=2, zorder=10,
               label=r"LHS: $e^{-ah^2/2}$")

    for color, n in zip(colors_mc, mc_ns):
        for run in range(n_runs):
            est = rhs_mc(h_vals, n, rng)
            ax_fn.plot(h_vals, est, color=color, alpha=0.4, lw=0.9,
                       label=rf"MC $N={n}$" if run == 0 else None)
            ax_err.plot(h_vals, np.abs(est - exact), color=color, alpha=0.4, lw=0.9,
                        label=rf"MC $N={n}$" if run == 0 else None)

    for color, n in zip(colors_gh, gh_ns):
        gh = rhs_hermite(h_vals, n)
        ax_fn.plot(h_vals, gh, color=color, lw=1.8, label=rf"GH $n={n}$")
        ax_err.plot(h_vals, np.abs(gh - exact), color=color, lw=1.8,
                    label=rf"GH $n={n}$")

    ax_fn.set_xlabel("$h$", fontsize=11)
    ax_fn.set_ylabel(r"$e^{-ah^2/2}$", fontsize=11)
    ax_fn.set_title(
        r"$e^{-ah^2/2} = \frac{1}{\sqrt{2\pi a}}\int e^{-u^2/(2a)\,-\,ihu}\,du$"
        f"  $(a={A})$",
        fontsize=11,
    )
    ax_fn.legend(fontsize=8)

    ax_err.set_xlabel("$h$", fontsize=11)
    ax_err.set_ylabel("|error|", fontsize=11)
    ax_err.set_yscale("log")
    ax_err.set_title("Absolute error vs exact LHS", fontsize=11)
    ax_err.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    plot_hs_identity(mc_ns=[10, 100, 1000], gh_ns=[3, 7, 15])
