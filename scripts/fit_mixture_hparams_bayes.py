"""Fully Bayesian version of the per-sample hyperparameter fit.

`fit_mixture_fourier_hparams.py` reports, per sample, the grid argmax of
``log p(y | theta)`` within each branch and picks the branch with the larger
maximum. That comparison has no Occam factor: the periodic branch has three
free hparams and the rbf_only branch one, so maximised likelihoods favour the
periodic branch by construction.

Here we integrate instead of maximise. With the data generator's prior
(uniform over each range inside a branch, Bernoulli `rbf_only_prob` over
branches — `train.sample_batch` with its defaults, i.e. no
`--n-strat-bins-*` stratification and no `--curriculum-steps`) the per-branch
evidence on a midpoint grid is

    log E_b(y) = log (1/|G_b|) sum_{theta in G_b} exp(log p(y | theta))

and the branch posterior is ``P(b | y) ∝ pi_b E_b(y)``. Within a branch the
posterior over theta is the normalised likelihood on the same grid, from which
we report marginal means and quantiles rather than a single argmax.

The grid can be much finer than the MAP script's because ``K(theta)`` does not
depend on ``y``: we factor each ``K(theta)`` once and solve against all samples
at once, instead of refactoring the whole grid per sample. That is a factor
``n_samples`` less linear algebra, which pays for the extra resolution needed
to make the evidence integral trustworthy.

Samples are read from an existing fit directory's `results.npz` so the output
is directly comparable to the MAP fit computed on the same y. The default
`--fit-dir` is what this command writes (model samples, Langevin correctors
disabled):

    uv run python scripts/fit_mixture_fourier_hparams.py \
        --sampler-num-corrections 0 --out-dir outputs/fit_mixture_hparams_ncorr0

A GT-prior control comes from the same script with `--source gt` (and its
own `--out-dir`); running this script on that directory gives the
`results_bayes.npz` that `--gt-ref` takes.

Outputs (under --out-dir):
  - results_bayes.npz : per-sample posterior summaries + population marginals.
  - results_bayes.csv : per-sample table (branch posterior, means, quantiles).
  - convergence.csv   : evidence / branch-fraction vs grid resolution.
  - marginals_bayes.png : posterior-weighted population marginals vs the MAP
                          histogram from the source fit.
  - panel_bayes.png   : sample curves overlaid with GP draws whose kernel is
                        itself drawn from the posterior (branch and theta),
                        so overlay spread shows hyperparameter uncertainty.

Example
-------
    uv run python scripts/fit_mixture_hparams_bayes.py \
        --fit-dir outputs/fit_mixture_hparams_ncorr0 \
        --out-dir outputs/fit_mixture_hparams_ncorr0_bayes
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import csv
import json
import math
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from neural_diffusion_processes import train as spt
from neural_diffusion_processes.paths import default_flow_dir, FLOW_DIR_SHOW_DEFAULT


# ---------------------------------------------------------------------------
# Likelihood over a theta grid, shared across all samples
# ---------------------------------------------------------------------------


def slack_gram(xa: jnp.ndarray, xb: jnp.ndarray, period, harmonic: float,
               width: float) -> jnp.ndarray:
    """Spectral-mixture component centred on a harmonic of the fundamental.

    ``k(r) = exp(-2 pi^2 sigma_f^2 r^2) cos(2 pi h r / period)`` has spectral
    density equal to a Gaussian pair at ``+-h/period`` with sd ``sigma_f``, so
    it puts power in a band around the ``h``-th harmonic and nowhere else.
    ``width`` is that sd measured in harmonic units, hence ``sigma_f =
    width / period``: the component tracks whatever fundamental the sample
    actually has, so an off-family residual tied to a harmonic of the
    sample's own period is captured at the same harmonic whatever that
    period is.
    """
    r = xa - xb.T
    sigma_f = width / period
    return (jnp.exp(-2.0 * (math.pi ** 2) * (sigma_f ** 2) * r ** 2)
            * jnp.cos(2.0 * math.pi * harmonic * r / period))


def make_evaluators(x: np.ndarray, variance: float, noise_var: float,
                    amplitude: str = "fixed", amp_grid: np.ndarray = None,
                    slack: dict | None = None):
    """Return (periodic, rbf_only) evaluators: theta chunk x all y -> log p.

    Each returns ``([chunk, n_samples], [chunk, n_samples])`` — the log
    likelihood and the quadratic form. The Cholesky of ``K(theta) + noise*I``
    is computed once per theta and reused across every sample via a single
    triangular solve with a matrix right-hand side.

    ``slack`` (dict with ``harmonic``, ``width``, ``c_grid``) adds a nuisance
    component to the *periodic* branch only:

        K = (1 - c) K_mix(theta) + c K_slack(period) + noise * I

    marginalised over a log-uniform grid of mixing fractions ``c``. The convex
    combination keeps the total prior variance at ``variance``, so ``c = 0``
    recovers the plain fit exactly and ``c`` reads as "fraction of variance
    that is off-family". Its purpose is to give a harmonic residual
    somewhere to go other than ``ls_per``/``ls_rbf``, which otherwise absorb it
    by projection. The rbf-only branch is left alone — it has no harmonics for
    the residual to attach to — so the periodic branch pays a genuine Occam
    cost for the extra parameter, and the branch share may move a little.

    ``amplitude``:
      - "fixed"    : covariance is ``K(theta) + noise*I`` with the generator's
                     unit variance, matching the MAP script.
      - "marginal" : an unknown signal scale ``c`` on the kernel only,
                     ``c*K(theta) + noise*I``, marginalised over a log-uniform
                     grid. The noise floor stays absolute, which is what makes
                     ``c`` identifiable.

                     Do NOT be tempted by the closed form that comes from
                     scaling the *whole* covariance (Jeffreys prior, giving
                     ``c_hat = quad/(N-2)``): with a noise floor far below the
                     signal, ``E[quad]`` is the effective rank of K above that
                     floor, which can be far below N, so the estimate
                     collapses: a GT control generated with c = 1 exactly
                     recovers a c far below 1. Hence the eigendecomposition:
                     one eigh per theta makes every c free.
    """
    xj = jnp.asarray(x, dtype=jnp.float32).reshape(-1, 1)
    n_pts = xj.shape[0]
    eye = jnp.eye(n_pts, dtype=jnp.float32)
    const = -0.5 * n_pts * math.log(2.0 * math.pi)
    c_grid = (jnp.asarray(amp_grid, dtype=jnp.float32)
              if amp_grid is not None else None)
    c_slack = (jnp.asarray(slack["c_grid"], dtype=jnp.float32)
               if slack is not None else None)

    def _log_p_fixed(K, Y):
        L = jnp.linalg.cholesky(K + noise_var * eye)
        A = jax.scipy.linalg.solve_triangular(L, Y, lower=True)  # [n_pts, N]
        quad = jnp.sum(A ** 2, axis=0)                           # [N]
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        return -0.5 * quad - 0.5 * log_det + const, jnp.zeros_like(quad)

    def _log_p_marginal(K, Y):
        mu, U = jnp.linalg.eigh(K)                  # mu: [n_pts], U: [n,n]
        mu = jnp.maximum(mu, 0.0)                   # eigh can return -1e-8
        P2 = (U.T @ Y) ** 2                         # [n_pts, N]
        lam = c_grid[:, None] * mu[None, :] + noise_var   # [n_c, n_pts]
        quad_c = (1.0 / lam) @ P2                         # [n_c, N]
        log_det_c = jnp.sum(jnp.log(lam), axis=1)         # [n_c]
        lp_c = -0.5 * quad_c - 0.5 * log_det_c[:, None] + const
        # Log-uniform prior over c: the grid is log-spaced, so a plain mean.
        lp = jax.scipy.special.logsumexp(lp_c, axis=0) - math.log(len(amp_grid))
        w_c = jax.nn.softmax(lp_c, axis=0)                # [n_c, N]
        c_hat = jnp.sum(c_grid[:, None] * w_c, axis=0)    # [N]
        return lp, c_hat

    def _log_p_slack(K, Ks, Y):
        """Marginalise the off-family mixing fraction c out of the periodic
        branch. ``lax.map`` rather than ``vmap`` over c: the per-c triangular
        solve is [n_pts, n_samples], and materialising all of them at once
        under the outer theta-vmap is what blows up memory on large sample
        sets. Sequencing costs nothing here — each c is already a full
        Cholesky."""
        def one_c(c):
            L = jnp.linalg.cholesky((1.0 - c) * K + c * Ks + noise_var * eye)
            A = jax.scipy.linalg.solve_triangular(L, Y, lower=True)
            quad = jnp.sum(A ** 2, axis=0)
            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            return -0.5 * quad - 0.5 * log_det + const

        lp_c = jax.lax.map(one_c, c_slack)                    # [n_c, N]
        lp = (jax.scipy.special.logsumexp(lp_c, axis=0)
              - math.log(len(slack["c_grid"])))
        w_c = jax.nn.softmax(lp_c, axis=0)
        c_hat = jnp.sum(c_slack[:, None] * w_c, axis=0)       # [N]
        return lp, c_hat

    _log_p = _log_p_marginal if amplitude == "marginal" else _log_p_fixed

    def one_periodic(period, ls_per, ls_rbf, Y):
        K = spt.mixture_gram(xj, xj, period, ls_per, ls_rbf,
                             jnp.bool_(False), variance)
        if slack is not None:
            Ks = variance * slack_gram(xj, xj, period, slack["harmonic"],
                                       slack["width"])
            return _log_p_slack(K, Ks, Y)
        return _log_p(K, Y)

    def one_rbf_only(ls_rbf_only, Y):
        K = spt.mixture_gram(xj, xj, jnp.float32(1.0), jnp.float32(1.0),
                             ls_rbf_only, jnp.bool_(True), variance)
        return _log_p(K, Y)

    eval_periodic = jax.jit(jax.vmap(one_periodic, in_axes=(0, 0, 0, None)))
    eval_rbf_only = jax.jit(jax.vmap(one_rbf_only, in_axes=(0, None)))
    return eval_periodic, eval_rbf_only


def midpoints(lo: float, hi: float, n: int) -> np.ndarray:
    """n equal-width cell centres spanning [lo, hi].

    Midpoints (not linspace endpoints) so that a plain mean over the grid is
    the midpoint-rule estimate of the uniform-prior expectation, with no
    edge-weight bookkeeping.
    """
    edges = np.linspace(lo, hi, n + 1)
    return (0.5 * (edges[:-1] + edges[1:])).astype(np.float32)


def log_lik_periodic_grid(eval_periodic, axes: tuple, Y: jnp.ndarray,
                          chunk: int, verbose: bool = True,
                          return_quad: bool = False):
    """log p for the full periodic tensor grid: [n_p * n_lp * n_lr, n_samples].

    ``return_quad`` additionally returns the quadratic forms (same shape); the
    convergence ladder does not need them and skipping the copy keeps its
    memory down on large sample sets.
    """
    period_ax, ls_per_ax, ls_rbf_ax = axes
    P, L1, L2 = np.meshgrid(period_ax, ls_per_ax, ls_rbf_ax, indexing="ij")
    p_flat = jnp.asarray(P.reshape(-1))
    lp_flat = jnp.asarray(L1.reshape(-1))
    lr_flat = jnp.asarray(L2.reshape(-1))
    n_grid = p_flat.shape[0]

    out = np.empty((n_grid, Y.shape[1]), dtype=np.float32)
    quad_out = (np.empty((n_grid, Y.shape[1]), dtype=np.float32)
                if return_quad else None)
    for lo in range(0, n_grid, chunk):
        hi = min(lo + chunk, n_grid)
        lp, quad = eval_periodic(p_flat[lo:hi], lp_flat[lo:hi],
                                 lr_flat[lo:hi], Y)
        out[lo:hi] = np.asarray(lp)
        if return_quad:
            quad_out[lo:hi] = np.asarray(quad)
        if verbose and (lo // chunk) % 50 == 0:
            print(f"    periodic grid {hi}/{n_grid}", flush=True)
    return (out, quad_out) if return_quad else out


def log_evidence(log_p: np.ndarray) -> np.ndarray:
    """log mean_theta exp(log_p) over axis 0 -> [n_samples], in float64."""
    return logsumexp(log_p.astype(np.float64), axis=0) - math.log(log_p.shape[0])


def branch_posterior(log_E_per: np.ndarray, log_E_rbf: np.ndarray,
                     p_rbf_only: float) -> tuple:
    log_pi_per = math.log(max(1e-12, 1.0 - p_rbf_only))
    log_pi_rbf = math.log(max(1e-12, p_rbf_only))
    a = log_pi_per + log_E_per
    b = log_pi_rbf + log_E_rbf
    norm = np.logaddexp(a, b)
    return np.exp(a - norm), np.exp(b - norm)


# ---------------------------------------------------------------------------
# Posterior summaries
# ---------------------------------------------------------------------------


def normalised_weights(log_p: np.ndarray) -> np.ndarray:
    """Column-wise softmax of [n_grid, n_samples] -> posterior over theta."""
    m = log_p.max(axis=0, keepdims=True)
    e = np.exp(log_p.astype(np.float64) - m)
    return e / e.sum(axis=0, keepdims=True)


def marginal_summaries(w: np.ndarray, axis_vals: np.ndarray,
                       qs=(0.05, 0.5, 0.95)) -> dict:
    """Mean and quantiles of a discrete marginal.

    w: [n_knots, n_samples] normalised along axis 0. Quantiles snap to knots,
    so they are only as fine as the grid — reported alongside the knot spacing.
    """
    mean = axis_vals @ w
    c = np.cumsum(w, axis=0)
    c /= c[-1:, :]
    out = {"mean": mean}
    for q in qs:
        idx = (c >= q).argmax(axis=0)
        out[f"q{int(q * 100):02d}"] = axis_vals[idx]
    # Posterior analogue of the MAP script's `at_boundary` flag: how much mass
    # sits in the outermost cells, i.e. how hard the sample is pushing against
    # the edge of the range the data generator can produce.
    out["edge_mass"] = w[0, :] + w[-1, :]
    return out


def marginalise(w_per: np.ndarray, shape: tuple) -> tuple:
    """Split a flat periodic posterior [n_grid, N] into its three marginals."""
    n_p, n_lp, n_lr = shape
    w3 = w_per.reshape(n_p, n_lp, n_lr, -1)
    return (w3.sum(axis=(1, 2)), w3.sum(axis=(0, 2)), w3.sum(axis=(0, 1)))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_panel(out_path: Path, x: np.ndarray, y_samples: np.ndarray,
               post_per: np.ndarray, post_rbf: np.ndarray, w_per: np.ndarray,
               axes: dict, summ: dict, cfg: dict, n_show: int,
               n_overlay: int, seed: int) -> None:
    """Sample curves with GP draws under kernels drawn from the posterior.

    The MAP script overlays draws under one best-fit kernel. Here each overlay
    uses its own kernel sampled from the posterior — branch from P(b | y), then
    theta from P(theta | y, b) — so the spread across overlays shows the
    hyperparameter uncertainty rather than hiding it behind a point estimate.
    """
    n_show = min(n_show, y_samples.shape[0])
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.7 * rows),
                            sharex=True)
    axs = np.atleast_2d(axs)
    variance = float(cfg["variance"])
    jitter = float(cfg["jitter"])
    y_range = tuple(cfg["y_range"])
    xj = jnp.asarray(x, dtype=jnp.float32).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_show * n_overlay)

    n_p, n_lp, n_lr = (len(axes["period"]), len(axes["ls_per"]),
                       len(axes["ls_rbf"]))
    # Cell widths, so a draw lands anywhere in its cell instead of on the knot.
    dw = {k: (float(axes[k][1] - axes[k][0]) if len(axes[k]) > 1 else 0.0)
          for k in ("period", "ls_per", "ls_rbf", "ls_rbf_only")}

    for i in range(n_show):
        ax = axs.flat[i]
        for j in range(n_overlay):
            if rng.random() < w_per[i]:
                flat = int(rng.choice(post_per.shape[0], p=post_per[:, i]))
                ip, rem = divmod(flat, n_lp * n_lr)
                ilp, ilr = divmod(rem, n_lr)
                th = dict(
                    period=axes["period"][ip] + dw["period"] * (rng.random() - .5),
                    ls_per=axes["ls_per"][ilp] + dw["ls_per"] * (rng.random() - .5),
                    ls_rbf=axes["ls_rbf"][ilr] + dw["ls_rbf"] * (rng.random() - .5),
                )
                args = (jnp.float32(th["period"]), jnp.float32(th["ls_per"]),
                        jnp.float32(th["ls_rbf"]), jnp.bool_(False))
                colour = "tab:orange"
            else:
                ir = int(rng.choice(post_rbf.shape[0], p=post_rbf[:, i]))
                v = axes["ls_rbf_only"][ir] + dw["ls_rbf_only"] * (rng.random() - .5)
                args = (jnp.float32(1.0), jnp.float32(1.0), jnp.float32(v),
                        jnp.bool_(True))
                colour = "tab:green"
            ys = spt._sample_one(keys[i * n_overlay + j], xj, *args,
                                 variance, jitter)[:, 0]
            ax.plot(x, np.asarray(ys), color=colour, lw=0.7, alpha=0.6)
        ax.plot(x, y_samples[i], color="black", lw=1.2)

        p = float(w_per[i])
        if p >= 0.5:
            hp = (f"per={summ['period']['mean'][i]:.2f} "
                  f"[{summ['period']['q05'][i]:.2f},{summ['period']['q95'][i]:.2f}]\n"
                  f"ls_p={summ['ls_per']['mean'][i]:.2f} "
                  f"ls_r={summ['ls_rbf']['mean'][i]:.2f} "
                  f"[{summ['ls_rbf']['q05'][i]:.2f},{summ['ls_rbf']['q95'][i]:.2f}]")
        else:
            hp = (f"ls_r={summ['ls_rbf_only']['mean'][i]:.2f} "
                  f"[{summ['ls_rbf_only']['q05'][i]:.2f},"
                  f"{summ['ls_rbf_only']['q95'][i]:.2f}]")
        ax.set_title(rf"$\mathbf{{\#{i}}}$  P(per|y)={p:.2f}" + f"\n{hp}",
                     fontsize=9)
        ax.set_ylim(y_range)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, alpha=0.2)
    for j in range(n_show, rows * cols):
        axs.flat[j].axis("off")
    fig.suptitle("black = model sample;  overlays = GP draws with kernel "
                 "sampled from the posterior (orange = periodic, green = rbf-only)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path}")


def _knot_edges(knots: np.ndarray) -> np.ndarray:
    """Cell edges around a set of knots, so a histogram over the source fit's
    coarse grid is binned at its own resolution instead of aliasing."""
    knots = np.asarray(knots, dtype=np.float64)
    if len(knots) == 1:
        return np.array([knots[0] - 0.5, knots[0] + 0.5])
    mid = 0.5 * (knots[:-1] + knots[1:])
    return np.concatenate([[knots[0] - (mid[0] - knots[0])], mid,
                           [knots[-1] + (knots[-1] - mid[-1])]])


def plot_marginals(out_path: Path, pop: dict, axes: dict, src: dict,
                   src_axes: dict, frac_bayes: float, frac_map: float,
                   gt_ref: dict | None = None) -> None:
    specs = [
        ("period", "period"),
        ("ls_per", r"$\ell_{per}$"),
        ("ls_rbf", r"$\ell_{rbf}$"),
        ("ls_rbf_only", r"$\ell_{rbf}$ (rbf-only branch)"),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, (key, label) in zip(axs, specs):
        knots = axes[key]
        # Both curves are densities (mass / cell width) so the fine posterior
        # grid and the coarse MAP histogram are directly comparable.
        width = float(knots[1] - knots[0]) if len(knots) > 1 else 1.0
        dens = pop[key] / pop[key].sum() / width
        ax.bar(knots, dens, width=width, color="#4C72B0", alpha=0.7,
               label="posterior (evidence-weighted)")
        map_vals = src.get(key)
        if map_vals is not None and len(map_vals):
            ax.hist(map_vals, bins=_knot_edges(src_axes[key]), density=True,
                    histtype="step", color="#C44E52", lw=2,
                    label="MAP argmax (source fit, own grid)")
        if gt_ref is not None and key in gt_ref:
            # Control: same estimator applied to draws from the actual data
            # generator. A calibrated model should sit on this curve, which
            # is flat up to the estimator's own (small) bias.
            ax.plot(knots, gt_ref[key] / gt_ref[key].sum() / width,
                    color="black", ls="--", lw=1.8,
                    label="GT prior samples (control)")
        lo, hi = float(knots[0] - width / 2), float(knots[-1] + width / 2)
        ax.axhline(1.0 / (hi - lo), color="0.45", ls=":", lw=1.4,
                   label="uniform prior")
        ax.set_title(label, fontsize=12)
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.2)
    axs[0].set_ylabel("density")
    axs[0].legend(fontsize=8)
    fig.suptitle(
        f"periodic share: posterior {frac_bayes:.3f}  vs  MAP {frac_map:.3f}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option("--fit-dir", default="outputs/fit_mixture_hparams_ncorr0",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Source of y_samples/x_grid and the MAP fit to compare to: "
                   "a fit_mixture_fourier_hparams.py --out-dir. The default is "
                   "what the --sampler-num-corrections 0 command in the module "
                   "docstring writes.")
@click.option("--run-dir", default=default_flow_dir,
              show_default=FLOW_DIR_SHOW_DEFAULT,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Checkpoint dir; only config.json is read (hparam ranges).")
@click.option("--out-dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--n-period", default=32, type=int, show_default=True)
@click.option("--n-ls-per", default=32, type=int, show_default=True)
@click.option("--n-ls-rbf", default=32, type=int, show_default=True)
@click.option("--n-ls-rbf-only", default=128, type=int, show_default=True)
@click.option("--noise-std", default=0.05, type=float, show_default=True,
              help="Must match the source fit to be comparable.")
@click.option("--amplitude", type=click.Choice(["fixed", "marginal"]),
              default="fixed", show_default=True,
              help="'fixed' holds the kernel variance at the generator's value "
                   "(as the MAP script does). 'marginal' integrates out an "
                   "signal scale c on the kernel (c*K + noise*I) over a "
                   "log-uniform grid, so the hparam posteriors reflect shape "
                   "only, not amplitude.")
@click.option("--n-amp", default=25, type=int, show_default=True,
              help="Grid points for the amplitude scale c.")
@click.option("--amp-range", default="0.2,5.0", show_default=True,
              help="Range for c, log-spaced.")
@click.option("--slack/--no-slack", default=False,
              help="Add a marginalised spectral-mixture nuisance component at "
                   "a harmonic of each sample's own fundamental, so the "
                   "off-family residual stops being absorbed by ls_per/ls_rbf. "
                   "Periodic branch only.")
@click.option("--slack-harmonic", default=3.0, type=float, show_default=True,
              help="Harmonic the slack component sits on.")
@click.option("--slack-width", default=0.85, type=float, show_default=True,
              help="Slack bandwidth, in harmonic units (spectral sd).")
@click.option("--n-slack", default=8, type=int, show_default=True,
              help="Grid points for the slack mixing fraction c.")
@click.option("--slack-range", default="0.001,0.3", show_default=True,
              help="Range for the slack mixing fraction, log-spaced. The "
                   "lower end should be small enough to read as 'no slack'; "
                   "posterior mass piling there means none was needed.")
@click.option("--max-samples", default=0, type=int, show_default=True,
              help="Cap on samples read from the fit dir (0 = all). Use to "
                   "keep the slack fit affordable on large control sets.")
@click.option("--theta-chunk", default=64, type=int, show_default=True,
              help="Number of grid points factored per XLA call.")
@click.option("--resolutions", default="8,11,16,24,32", show_default=True,
              help="Per-axis periodic-grid sizes for the convergence check.")
@click.option("--store-grids/--no-store-grids", default=False,
              help="Persist the full log-likelihood grids (large).")
@click.option("--n-panel", default=16, type=int, show_default=True,
              help="Samples shown in panel_bayes.png (0 disables the panel).")
@click.option("--panel-overlay", default=3, type=int, show_default=True,
              help="Posterior kernel draws overlaid per sample.")
@click.option("--panel-seed", default=7, type=int, show_default=True)
@click.option("--gt-ref", type=click.Path(exists=True, dir_okay=False,
                                          path_type=Path), default=None,
              help="results_bayes.npz from a GT-prior control run, overlaid on "
                   "the marginals plot. Must share the same grid axes.")
def main(fit_dir: Path, run_dir: Path, out_dir: Path,
         n_period: int, n_ls_per: int, n_ls_rbf: int, n_ls_rbf_only: int,
         noise_std: float, amplitude: str, n_amp: int, amp_range: str,
         slack: bool, slack_harmonic: float, slack_width: float,
         n_slack: int, slack_range: str, max_samples: int,
         theta_chunk: int, resolutions: str,
         store_grids: bool, n_panel: int, panel_overlay: int,
         panel_seed: int, gt_ref: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run_dir / "config.json").read_text())
    _src = np.load(fit_dir / "results.npz")
    src_files = list(_src.files)
    src = {k: np.asarray(_src[k]) for k in src_files}

    if max_samples and len(src["y_samples"]) > max_samples:
        # Deterministic prefix, not a random subset: the source fit draws
        # samples i.i.d., so the prefix is already an unbiased subsample and
        # taking it keeps every run on the same y for paired comparison.
        n_full = len(src["branch"])
        keep = [k for k in src_files
                if k not in ("x_grid",) and not k.endswith(("_axis", "_range"))
                and src[k].shape[:1] == (n_full,)]
        for k in keep:
            src[k] = src[k][:max_samples]
        print(f"capped to first {max_samples} of {n_full} samples")

    x_grid = src["x_grid"]
    y_samples = src["y_samples"]                      # [N, n_pts]
    n_samples = y_samples.shape[0]
    Y = jnp.asarray(y_samples.T, dtype=jnp.float32)   # [n_pts, N]

    variance = float(cfg["variance"])
    noise_var = float(noise_std) ** 2
    p_rbf_only = float(cfg["rbf_only_prob"])
    ranges = {k: tuple(cfg[f"{k}_range"])
              for k in ("period", "ls_per", "ls_rbf", "ls_rbf_only")}

    print(f"source fit: {fit_dir}  N={n_samples}  n_pts={x_grid.shape[0]}")
    print(f"  variance={variance}  noise_var={noise_var}  "
          f"rbf_only_prob={p_rbf_only}")
    print(f"  ranges: {ranges}")

    amp_grid = None
    if amplitude == "marginal":
        a_lo, a_hi = (float(v) for v in amp_range.split(","))
        amp_grid = np.geomspace(a_lo, a_hi, n_amp).astype(np.float32)
    print(f"  amplitude: {amplitude}"
          + (f"  grid {n_amp} pts over [{amp_range}] (log-uniform)"
             if amp_grid is not None else ""))

    slack_cfg = None
    if slack:
        if amplitude == "marginal":
            raise click.UsageError(
                "--slack and --amplitude marginal are not supported together: "
                "the slack path needs a Cholesky per (theta, c) and cannot "
                "reuse the eigendecomposition the amplitude path relies on.")
        s_lo, s_hi = (float(v) for v in slack_range.split(","))
        slack_cfg = dict(harmonic=float(slack_harmonic),
                         width=float(slack_width),
                         c_grid=np.geomspace(s_lo, s_hi, n_slack).astype(np.float32))
        print(f"  slack: harmonic {slack_harmonic} width {slack_width} "
              f"(harmonic units), c grid {n_slack} pts over [{slack_range}] "
              f"(log-uniform), periodic branch only")
    eval_periodic, eval_rbf_only = make_evaluators(x_grid, variance, noise_var,
                                                   amplitude, amp_grid,
                                                   slack_cfg)

    # ---- convergence in grid resolution -----------------------------------
    # The evidence is an integral of a sharply peaked function; the MAP script
    # used 11 knots per axis, which is where we expect the largest error.
    res_list = [int(r) for r in resolutions.split(",") if r.strip()]
    # Only fold the final grid into the ladder when it is actually a cube;
    # with an anisotropic grid (e.g. a finely resolved period axis) appending
    # n_period would build an n_period^3 grid nobody asked for.
    if n_period == n_ls_per == n_ls_rbf and n_period not in res_list:
        res_list.append(n_period)
    res_list = sorted(set(res_list))

    conv_rows = []
    log_E_per_by_res = {}
    for r in res_list:
        axes_r = (midpoints(*ranges["period"], r),
                  midpoints(*ranges["ls_per"], r),
                  midpoints(*ranges["ls_rbf"], r))
        print(f"  evidence at {r}^3 = {r ** 3} grid points ...", flush=True)
        lp = log_lik_periodic_grid(eval_periodic, axes_r, Y, theta_chunk,
                                   verbose=False)
        log_E_per_by_res[r] = log_evidence(lp)
        del lp

    n_rbf_ref = max(n_ls_rbf_only, 4 * max(res_list))
    ls_rbf_only_ref = midpoints(*ranges["ls_rbf_only"], n_rbf_ref)
    lp_rbf_ref = np.asarray(eval_rbf_only(jnp.asarray(ls_rbf_only_ref), Y)[0])
    log_E_rbf_ref = log_evidence(lp_rbf_ref)
    del lp_rbf_ref

    ref_res = max(res_list)
    for r in res_list:
        d = log_E_per_by_res[r] - log_E_per_by_res[ref_res]
        w_per_r, _ = branch_posterior(log_E_per_by_res[r], log_E_rbf_ref,
                                      p_rbf_only)
        conv_rows.append(dict(
            n_per_axis=r, n_grid=r ** 3,
            mean_log_E_per=float(log_E_per_by_res[r].mean()),
            mean_abs_err_vs_ref=float(np.abs(d).mean()),
            max_abs_err_vs_ref=float(np.abs(d).max()),
            periodic_share=float((w_per_r > 0.5).mean()),
            mean_w_per=float(w_per_r.mean()),
        ))

    # ---- reference fit at the requested resolution ------------------------
    axes = dict(
        period=midpoints(*ranges["period"], n_period),
        ls_per=midpoints(*ranges["ls_per"], n_ls_per),
        ls_rbf=midpoints(*ranges["ls_rbf"], n_ls_rbf),
        ls_rbf_only=midpoints(*ranges["ls_rbf_only"], n_ls_rbf_only),
    )
    print(f"\nfinal grid: periodic {n_period}x{n_ls_per}x{n_ls_rbf} = "
          f"{n_period * n_ls_per * n_ls_rbf}  rbf_only {n_ls_rbf_only}",
          flush=True)
    log_p_per, amp_per = log_lik_periodic_grid(
        eval_periodic, (axes["period"], axes["ls_per"], axes["ls_rbf"]),
        Y, theta_chunk, return_quad=True)
    lp_rbf, a_rbf = eval_rbf_only(jnp.asarray(axes["ls_rbf_only"]), Y)
    log_p_rbf = np.asarray(lp_rbf)
    amp_rbf = np.asarray(a_rbf)

    log_E_per = log_evidence(log_p_per)
    log_E_rbf = log_evidence(log_p_rbf)
    w_per, w_rbf = branch_posterior(log_E_per, log_E_rbf, p_rbf_only)

    post_per = normalised_weights(log_p_per)   # [n_grid, N]
    post_rbf = normalised_weights(log_p_rbf)   # [n_knots, N]
    m_period, m_ls_per, m_ls_rbf = marginalise(
        post_per, (n_period, n_ls_per, n_ls_rbf))

    # Posterior amplitude, averaged over the theta-posterior within each
    # branch and then over the branch posterior: the factor by which a
    # sample's signal variance exceeds the generator's unit variance. Only
    # defined when c was actually part of the model.
    if amplitude == "marginal":
        c_per = (post_per * amp_per).sum(axis=0)   # [N]
        c_rbf = (post_rbf * amp_rbf).sum(axis=0)   # [N]
        amp_mean = w_per * c_per + w_rbf * c_rbf
    else:
        amp_mean = np.full(n_samples, np.nan)

    # Posterior slack fraction, averaged over the theta-posterior. Conditional
    # on the periodic branch, since that is the only branch it is defined on.
    if slack_cfg is not None:
        slack_mean = (post_per * amp_per).sum(axis=0)   # [N]
    else:
        slack_mean = np.full(n_samples, np.nan)
    del amp_per, amp_rbf

    summ = {
        "period": marginal_summaries(m_period, axes["period"]),
        "ls_per": marginal_summaries(m_ls_per, axes["ls_per"]),
        "ls_rbf": marginal_summaries(m_ls_rbf, axes["ls_rbf"]),
        "ls_rbf_only": marginal_summaries(post_rbf, axes["ls_rbf_only"]),
    }

    # MAP on the same fine grid, for a like-for-like comparison with the
    # source fit (which used 11/11/11/31 knots).
    i_per = log_p_per.argmax(axis=0)
    i_rbf = log_p_rbf.argmax(axis=0)
    P, L1, L2 = np.meshgrid(axes["period"], axes["ls_per"], axes["ls_rbf"],
                            indexing="ij")
    map_fine = dict(
        period=P.reshape(-1)[i_per],
        ls_per=L1.reshape(-1)[i_per],
        ls_rbf=L2.reshape(-1)[i_per],
        ls_rbf_only=axes["ls_rbf_only"][i_rbf],
        gap=log_p_per.max(axis=0).astype(np.float64)
            - log_p_rbf.max(axis=0).astype(np.float64),
    )

    branch_bayes = np.where(w_per > 0.5, "periodic", "rbf_only")
    branch_map_fine = np.where(map_fine["gap"] >= 0.0, "periodic", "rbf_only")
    branch_src = np.asarray(src["branch"])

    # Legacy evidence: exactly what the MAP script's marginals plot used, read
    # straight from the source npz (11-knot linspace grid, endpoints included).
    legacy = None
    if "log_p_periodic_grid" in src_files:
        lg_per = log_evidence(np.asarray(src["log_p_periodic_grid"]).T)
        lg_rbf = log_evidence(np.asarray(src["log_p_rbf_only_grid"]).T)
        lg_w_per, _ = branch_posterior(lg_per, lg_rbf, p_rbf_only)
        legacy = dict(log_E_per=lg_per, log_E_rbf=lg_rbf, w_per=lg_w_per)

    # ---- report ------------------------------------------------------------
    print("\nresolution convergence (periodic branch evidence):")
    print(f"{'n/axis':>7} {'n_grid':>8} {'mean log E':>12} "
          f"{'mean|err|':>10} {'max|err|':>9} {'periodic share':>15}")
    for row in conv_rows:
        print(f"{row['n_per_axis']:>7} {row['n_grid']:>8} "
              f"{row['mean_log_E_per']:>12.3f} "
              f"{row['mean_abs_err_vs_ref']:>10.3f} "
              f"{row['max_abs_err_vs_ref']:>9.3f} "
              f"{row['periodic_share']:>15.3f}")

    frac_bayes = float((branch_bayes == "periodic").mean())
    frac_map_fine = float((branch_map_fine == "periodic").mean())
    frac_src = float((branch_src == "periodic").mean())
    flip = float((branch_bayes != branch_src).mean())
    print(f"\nperiodic share  source MAP (11^3): {frac_src:.3f}")
    print(f"periodic share  MAP on fine grid : {frac_map_fine:.3f}")
    print(f"periodic share  branch posterior : {frac_bayes:.3f}")
    print(f"disagreement posterior vs source MAP: {flip:.3f}")
    n_ambig = float(((w_per > 0.05) & (w_per < 0.95)).mean())
    print(f"fraction with ambiguous branch (0.05 < P < 0.95): {n_ambig:.3f}")
    if slack_cfg is not None:
        cg = slack_cfg["c_grid"]
        is_per = branch_bayes == "periodic"
        s = slack_mean[is_per]
        lo_frac = float((s <= cg[0] * 1.5).mean())
        hi_frac = float((s >= cg[-1] * 0.9).mean())
        print(f"\nslack fraction c (periodic branch, n={is_per.sum()}): "
              f"mean {s.mean():.4f} median {np.median(s):.4f}  "
              f"[{np.quantile(s, 0.05):.4f}, {np.quantile(s, 0.95):.4f}]")
        print(f"  at grid floor (c ~ 0, no slack needed): {lo_frac:.3f}   "
              f"at grid ceiling (slack range too narrow): {hi_frac:.3f}")
    if amplitude == "marginal":
        at_edge = float(((amp_mean <= amp_grid[0] * 1.02)
                         | (amp_mean >= amp_grid[-1] * 0.98)).mean())
        print(f"\nposterior amplitude (generator = 1.0): "
              f"mean {amp_mean.mean():.4f} median {np.median(amp_mean):.4f}  "
              f"[{np.quantile(amp_mean, 0.05):.3f}, "
              f"{np.quantile(amp_mean, 0.95):.3f}]  "
              f"at c-grid edge {at_edge:.3f}  "
              f"raw sample variance {y_samples.var():.4f}")
    if legacy is not None:
        d = log_E_per - legacy["log_E_per"]
        print(f"\nlegacy 11^3 linspace evidence vs {ref_res}^3 midpoint: "
              f"mean err {d.mean():+.3f} nats, mean|err| {np.abs(d).mean():.3f}, "
              f"max|err| {np.abs(d).max():.3f}")
        print(f"legacy periodic share: "
              f"{float((legacy['w_per'] > 0.5).mean()):.3f}")

    # ---- write -------------------------------------------------------------
    conv_path = out_dir / "convergence.csv"
    with conv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(conv_rows[0].keys()))
        w.writeheader()
        w.writerows(conv_rows)
    print(f"\nwrote {conv_path}")

    csv_path = out_dir / "results_bayes.csv"
    fields = ["sample", "branch_posterior", "P_periodic", "log_E_periodic",
              "log_E_rbf_only", "branch_map_fine", "branch_map_source",
              "amplitude", "slack_c"]
    for k in ("period", "ls_per", "ls_rbf", "ls_rbf_only"):
        fields += [f"{k}_mean", f"{k}_q05", f"{k}_q50", f"{k}_q95",
                   f"{k}_map", f"{k}_edge_mass"]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i in range(n_samples):
            row = [i, branch_bayes[i], f"{w_per[i]:.6g}", f"{log_E_per[i]:.6g}",
                   f"{log_E_rbf[i]:.6g}", branch_map_fine[i], branch_src[i],
                   f"{amp_mean[i]:.6g}", f"{slack_mean[i]:.6g}"]
            for k in ("period", "ls_per", "ls_rbf", "ls_rbf_only"):
                row += [f"{summ[k]['mean'][i]:.6g}", f"{summ[k]['q05'][i]:.6g}",
                        f"{summ[k]['q50'][i]:.6g}", f"{summ[k]['q95'][i]:.6g}",
                        f"{map_fine[k][i]:.6g}",
                        f"{summ[k]['edge_mass'][i]:.6g}"]
            w.writerow(row)
    print(f"wrote {csv_path}")

    pop = dict(
        period=(w_per * m_period).mean(axis=1),
        ls_per=(w_per * m_ls_per).mean(axis=1),
        ls_rbf=(w_per * m_ls_rbf).mean(axis=1),
        ls_rbf_only=(w_rbf * post_rbf).mean(axis=1),
    )

    npz = dict(
        w_per=w_per, w_rbf=w_rbf,
        log_E_periodic=log_E_per, log_E_rbf_only=log_E_rbf,
        branch_posterior=branch_bayes, branch_map_fine=branch_map_fine,
        branch_map_source=branch_src,
        period_axis=axes["period"], ls_per_axis=axes["ls_per"],
        ls_rbf_axis=axes["ls_rbf"], ls_rbf_only_axis=axes["ls_rbf_only"],
        pop_period=pop["period"], pop_ls_per=pop["ls_per"],
        pop_ls_rbf=pop["ls_rbf"], pop_ls_rbf_only=pop["ls_rbf_only"],
        conv_n_axis=np.array([r["n_per_axis"] for r in conv_rows]),
        conv_mean_log_E=np.array([r["mean_log_E_per"] for r in conv_rows]),
        conv_periodic_share=np.array([r["periodic_share"] for r in conv_rows]),
        noise_std=np.float64(noise_std), p_rbf_only=np.float64(p_rbf_only),
        amplitude=amp_mean, amplitude_mode=np.str_(amplitude),
        slack_c=slack_mean,
        slack_on=np.bool_(slack_cfg is not None),
        slack_harmonic=np.float64(slack_harmonic),
        slack_width=np.float64(slack_width),
        slack_c_grid=(slack_cfg["c_grid"] if slack_cfg is not None
                      else np.zeros(0, dtype=np.float32)),
    )
    for k in ("period", "ls_per", "ls_rbf", "ls_rbf_only"):
        for stat in ("mean", "q05", "q50", "q95", "edge_mass"):
            npz[f"{k}_{stat}"] = summ[k][stat]
        npz[f"{k}_map_fine"] = map_fine[k]
    if legacy is not None:
        npz["legacy_log_E_periodic"] = legacy["log_E_per"]
        npz["legacy_log_E_rbf_only"] = legacy["log_E_rbf"]
        npz["legacy_w_per"] = legacy["w_per"]
    if store_grids:
        npz["log_p_periodic_grid"] = log_p_per.astype(np.float32).T
        npz["log_p_rbf_only_grid"] = log_p_rbf.astype(np.float32).T
    npz_path = out_dir / "results_bayes.npz"
    np.savez_compressed(npz_path, **npz)
    print(f"wrote {npz_path}")

    src_map = {k: np.asarray(src[k]) for k in
               ("period", "ls_per", "ls_rbf", "ls_rbf_only")}
    is_per_src = branch_src == "periodic"
    if n_panel > 0:
        plot_panel(out_dir / "panel_bayes.png", x_grid, y_samples,
                   post_per, post_rbf, w_per, axes, summ, cfg,
                   n_panel, panel_overlay, panel_seed)

    src_axes = {k: np.asarray(src[f"{k}_axis"]) for k in
                ("period", "ls_per", "ls_rbf", "ls_rbf_only")}

    gt_pop = None
    if gt_ref is not None:
        g = np.load(gt_ref)
        mismatch = [k for k in ("period", "ls_per", "ls_rbf", "ls_rbf_only")
                    if len(g[f"{k}_axis"]) != len(axes[k])]
        if mismatch:
            print(f"WARNING: --gt-ref grid differs on {mismatch}; "
                  f"skipping the control overlay")
        else:
            gt_pop = {k: g[f"pop_{k}"] for k in
                      ("period", "ls_per", "ls_rbf", "ls_rbf_only")}
            print(f"overlaying GT control from {gt_ref}")

    plot_marginals(
        out_dir / "marginals_bayes.png", pop, axes,
        dict(period=src_map["period"][is_per_src],
             ls_per=src_map["ls_per"][is_per_src],
             ls_rbf=src_map["ls_rbf"][is_per_src],
             ls_rbf_only=src_map["ls_rbf_only"][~is_per_src]),
        src_axes, frac_bayes, frac_src, gt_pop,
    )


if __name__ == "__main__":
    main()
