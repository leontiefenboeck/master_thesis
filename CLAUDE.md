# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

- `code/` — Python research code (PyTorch). The experiments are run from `code/` directly; there is no package layout, `setup.py`, or test suite. Imports are flat (e.g. `from gmm import GMM`).
- `thesis/` — LaTeX source for the master's thesis (Karl Voit template). Build with `cd thesis && make` (uses `pdflatex` + `biber`). Output is `thesis/main.pdf`.

## Running experiments

There is no test runner. The main entry points are:

```
cd code && jupyter notebook      # demo.ipynb — main notebook
cd code && ~/.pyenv/versions/ml/bin/python pg_demo.py   # PG identity demo
cd code && ~/.pyenv/versions/ml/bin/python hs_demo.py   # Hubbard-Stratonovich demo
```

`demo.ipynb` fits a `GMM` + sklearn `LogisticRegression`, constructs a partially observed test point, and compares all expectation estimators. The inference cell defines a local `evaluate(name, fn, n_runs)` helper that calls a zero-argument lambda.

Figures are written to `code/figures/`.

## Core architecture

The thesis investigates computing **E[σ(wᵀx) | x_obs]** — the expected logistic-regression prediction marginalised over missing features — under a learned density p(x). Four estimators are compared:

1. **`MCBaseline`** (`baselines.py`) — vanilla Monte Carlo: draw x_mis from p(x_mis | x_obs) and average σ(wᵀx).
2. **`PCPGMC`** (`pcpg.py`) — full Pólya-Gamma / characteristic-function estimator. Draws ω ~ PG(1,0) and u | ω ~ N(0, ω) both via Monte Carlo, then evaluates the conditional characteristic function φ(s) of the GMM at s = −u·w_mis.
3. **`PCPGMCHermite`** (`pcpg.py`) — hybrid: ω still sampled via MC, but the inner Gaussian integral over u | ω is handled by Gauss-Hermite quadrature (deterministic given ω).
4. **`PCPGQuadrature`** (`pcpg.py`) — fully deterministic: both the outer PG integral and the inner Gaussian integral use quadrature. Zero variance by construction.

All three PCPG variants inherit from `_PCPGBase` and share its `_expectation` method.

## PCPG class hierarchy and parameters

Sample counts are **not** stored in the constructor — they are passed at call time:

```python
# constructors — only precomputed nodes go here
PCPGMC(model, seed)
PCPGMCHermite(model, n_gauss_quad, seed)
PCPGQuadrature(model, n_pg_quad, n_gauss_quad, seed)

# call signatures
PCPGMC()(w, x_partial, n_pg, n_gauss)
PCPGMCHermite()(w, x_partial, n_pg)
PCPGQuadrature()(w, x_partial)           # fully deterministic, no runtime params
MCBaseline()(w, x_partial, n_samples)
```

Parameters:
- `n_pg` — PG Monte Carlo samples (outer integral)
- `n_gauss` — Gaussian Monte Carlo samples (inner integral, `PCPGMC` only)
- `n_gauss_quad` — Gauss-Hermite nodes (inner integral, precomputed in `__init__`)
- `n_pg_quad` — Gauss-PG quadrature nodes (outer integral, precomputed in `__init__`; max ≈ 19)

## Gauss-PG quadrature

`PCPGQuadrature` uses fixed quadrature nodes/weights for ω ~ PG(1,0), computed once at `__init__` via the Golub-Welsch algorithm in `pg_gauss_quadrature(n)` (public function in `pcpg.py`). Key details:

- The Laplace transform of PG(1,0) is L(t) = sech(√(t/2)); moments are derived from this analytically using `mpmath` numerical differentiation.
- The entire Golub-Welsch computation (moments → Hankel matrix → Cholesky → Jacobi eigendecomposition) is done in `mpmath` at `dps=200` decimal places — float64 Hankel matrices become singular for n > ~12.
- **Maximum reliable n: 19** (n=20 works at dps=200 but is at the numerical limit).
- The computation takes a few seconds per `PCPGQuadrature(...)` construction call.

`pg_demo.py` is the companion demo script with: `plot_convergence_functions`, `plot_convergence_errors`, `plot_pg_distribution`, `plot_pg_samples`.

`hs_demo.py` demonstrates the Hubbard-Stratonovich identity e^{-ωt²/2} = E_{u~N(0,ω)}[e^{iut}] standalone, without PG.

## DataManager

`DataManager` (`data.py`) exposes:
- `get_test(idx, device)` — returns the complete test point as a tensor.
- `get_test_missing(missing_rate, idx, device)` — returns a tensor with `NaN` in missing positions.

Both methods accept an optional `device` argument.

## Density model interface

Both estimators expect the density model to expose:
- `sample(n, x_obs)` — for MC.
- `characteristic_function(s, x_obs)` — for PCPG (evaluates the conditional CF of p(x_mis | x_obs) at frequencies s).

`GMM` (`gmm.py`) implements both: diagonal-covariance mixture trained by gradient descent on log-likelihood, parameterised with `log_vars` to keep variances positive.

Missingness convention: a partially observed input is a 1-D tensor with `NaN` in missing positions; `obs_mask = ~torch.isnan(x_partial)` is the standard idiom used everywhere.

## demo.py — script entry point

`code/demo.py` is the primary script version of the notebook. Top-level config variables:

```python
n_gauss_quad = 20   # Gauss-Hermite nodes (inner quadrature)
n_pg_quad    = 20   # Golub-Welsch PG nodes (outer quadrature, max 19)
n_gauss      = 100  # inner MC samples for eval_estimators()
n_pg         = 100  # outer MC samples for eval_estimators()
pcpg_ratio   = 4    # n_gauss / n_pg for PCPG-MC (see budget allocation below)
K            = 4    # GMM components
n_features   = 15
```

Four functions at the bottom, call whichever sections are needed:
```python
eval_estimators()
allocation_analysis()
results, convergence_budgets, n_gauss_quad = convergence_analysis()
variance_analysis(results, convergence_budgets, n_gauss_quad)
```

## PCPG-MC budget allocation

For a total budget `N = M * L` with desired ratio `r = L/M = pcpg_ratio`:

```
M = sqrt(N / r),   L = N // M
```

`demo.py` sets this as `n_pg_pcpg = max(1, int(np.sqrt(N / pcpg_ratio)))`.

**Recommended ratio: 4.** Reasoning:
- σ²_A (inner variance) is theoretically infinite; L must be large enough to tame the e^{1/(8γ)} blowup for small γ. L ≥ 20–50 is sufficient in practice.
- The variance formula scales the outer term as σ²_B · √ratio / √N, so large ratio hurts when σ²_B is non-negligible.
- ratio=4 gives L≈63 at N=1000 (safe), ratio=10 is comfortable but inflates the outer term unnecessarily.
- With many missing features σ²_B ≈ 0 (CF damping), so ratio matters less — any value in [4, 10] is fine.

## Same-seed pitfall with PCPG estimators

**Do not re-instantiate** `PCPGMC`, `PCPGMCHermite`, etc. inside a loop. Each constructor resets the internal RNG to `seed=42`, so every call returns the same sample — measured variance will be 0. Instantiate once before the loop:

```python
pcpg = PCPGMC(px)           # once
vals = [pcpg(w, x_partial, n_pg=1, n_gauss=1) for _ in range(n_reps)]  # varied
```

## Variance components

Estimated from single-sample runs (n_reps ≈ 1000):
- `sigma2_mc = np.var(mc1_vals)` — MC per-sample variance (≤ 0.25)
- `sigma2_B  = 4 * np.var(gh1_vals)` — outer PG variance (GH eliminates inner)
- `sigma2_A  = 4 * np.var(pcpg11_vals) - sigma2_B` — inner Gaussian variance (∞ in theory)

`print_variance_components` and `plot_variance_analysis` in `utils.py` handle display/plotting.

## Plotting conventions

Plotting functions in `utils.py` must **never** call `plt.show()` or `print()`. They save to `figures/` and call `plt.close()`. The caller decides whether to display output.

## Dependencies

Not pinned. Required: `torch`, `numpy`, `scikit-learn`, `matplotlib`, `polyagamma`, `mpmath` (for `PCPGQuadrature`). The `ml` pyenv version has all of these — run scripts with `~/.pyenv/versions/ml/bin/python`.
