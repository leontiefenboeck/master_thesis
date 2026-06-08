# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

- `code/` — Python research code (PyTorch). The experiments are run from `code/` directly; there is no package layout, `setup.py`, or test suite. Imports are flat (e.g. `from gmm import GMM`).
- `thesis/` — LaTeX source for the master's thesis (Karl Voit template). Build with `cd thesis && make` (uses `pdflatex` + `biber`). Output is `thesis/main.pdf`.

## Running experiments

There is no test runner. The main entry points are:

```
cd code && python demo.py        # script version
cd code && jupyter notebook      # demo.ipynb — notebook version with hyperparams in cell-4
```

`demo.py` / `demo.ipynb` fit a `GMM` + sklearn `LogisticRegression`, construct a partially observed test point, and compare all expectation estimators via `utils.evaluate`.

Figures are written to `code/figures/`.

## Core architecture

The thesis investigates computing **E[σ(wᵀx) | x_obs]** — the expected logistic-regression prediction marginalised over missing features — under a learned density p(x). Three estimators are compared:

1. **`MCBaseline`** (`baselines.py`) — vanilla Monte Carlo: draw x_mis from p(x_mis | x_obs) and average σ(wᵀx).
2. **`PCPG`** (`pcpg.py`) — full Pólya-Gamma / characteristic-function estimator. Draws ω ~ PG(1,0) and u | ω ~ N(0, ω) both via Monte Carlo, then evaluates the conditional characteristic function φ(s) of the GMM at s = −u·w_mis.
3. **`PCPGMCHermite`** (`pcpg.py`) — hybrid: ω still sampled via MC, but the inner Gaussian integral over u | ω is handled by Gauss-Hermite quadrature (deterministic given ω).
4. **`PCPGQuadrature`** (`pcpg.py`) — fully deterministic: both the outer PG integral and the inner Gaussian integral use quadrature. Zero variance by construction.

All three PCPG variants share the base `PCPG` class and its `_expectation` method.

## PCPG class hierarchy and parameters

```python
PCPG(model, n_pg, n_gauss, n_gauss_quad, seed)        # full MC
PCPGMCHermite(model, n_pg, n_gauss_quad, seed)         # MC-PG + Hermite inner
PCPGQuadrature(model, n_pg_quad, n_gauss_quad, seed)   # fully deterministic
```

Parameter naming matches `demo.ipynb` cell-4:
- `n_pg` — PG Monte Carlo samples (outer integral)
- `n_gauss` — Gaussian Monte Carlo samples (inner integral, base class only)
- `n_gauss_quad` — Gauss-Hermite nodes (inner integral, used by MCHermite and Quadrature)
- `n_pg_quad` — Gauss-PG quadrature nodes (outer integral, used by Quadrature only; max ≈ 19)

## Gauss-PG quadrature

`PCPGQuadrature` uses fixed quadrature nodes/weights for ω ~ PG(1,0), computed once at `__init__` via the Golub-Welsch algorithm in `_pg_gauss_quadrature(n)`. Key details:

- The Laplace transform of PG(1,0) is L(t) = sech(√(t/2)); moments are derived from this analytically using `mpmath` numerical differentiation.
- The entire Golub-Welsch computation (moments → Hankel matrix → Cholesky → Jacobi eigendecomposition) is done in `mpmath` at `dps=200` decimal places — float64 Hankel matrices become singular for n > ~12.
- **Maximum reliable n: 19** (n=20 works at dps=200 but is at the numerical limit).
- The computation takes a few seconds per `PCPGQuadrature(...)` construction call.

`pg_quadrature.py` is the companion script that:
- Regenerates and prints the nodes/weights for any n (useful if you want to hardcode them)
- Produces convergence plots comparing MC vs Gauss-PG quadrature: `plot_convergence_functions` and `plot_convergence_errors`

## Density model interface

Both estimators expect the density model to expose:
- `conditional_sample(x_obs, obs_mask, n)` — for MC.
- `characteristic_function(s, x_obs)` — for PCPG (evaluates the conditional CF of p(x_mis | x_obs) at frequencies s).

`GMM` (`gmm.py`) implements both: diagonal-covariance mixture trained by gradient descent on log-likelihood, parameterised with `log_vars` to keep variances positive.

Missingness convention: a partially observed input is a 1-D tensor with `NaN` in missing positions; `obs_mask = ~torch.isnan(x_partial)` is the standard idiom used everywhere.

## Dependencies

Not pinned. Required: `torch`, `numpy`, `scikit-learn`, `matplotlib`, `polyagamma`, `mpmath` (for `PCPGQuadrature`). The `ml` pyenv version has all of these — run scripts with `~/.pyenv/versions/ml/bin/python`.
