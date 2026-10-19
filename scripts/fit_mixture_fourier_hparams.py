"""Per-sample data-generating-prior hyperparameter fit.

Draw unconditional samples from the flow checkpoint at `--run-dir` and,
for each sample y on the training x-grid, find the kernel hyperparameters
inside the data generator's training prior that maximize the GP marginal
log-likelihood ``log p(y | N(0, K(theta) + noise_std^2 * I))`` (``--noise-std``,
default 0.05). Two branches are
searched, mirroring the data generator's mixture (see
`train.mixture_gram`):

  - periodic × RBF: theta = (period, ls_per, ls_rbf), ranges from config.json
  - pure RBF:       theta = (ls_rbf_only,),         range from config.json

For each sample we report the optimum within each branch, the winning branch
(the one with higher log p), and a `at_boundary` flag for diagnostic when the
optimum lands on the edge of any range — a hint the model is producing a
sample the data generator can't.

Outputs (under --out-dir):
  - results.npz : per-sample arrays plus the search grids.
  - results.csv : per-sample table for quick inspection.
  - panel.png   : sample curves overlaid with 3 GP prior draws under the
                  best-fit kernel for visual confirmation.
  - sanity.png  : (unless --no-sanity) ground-truth GP draws with known hparams,
                  showing the fitter recovers the true values within grid
                  resolution before we trust it on model samples.

Example
-------
    uv run python scripts/fit_mixture_fourier_hparams.py \
        --n-samples 16 --sanity
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import contextlib
import csv
import io
import json
import math
from pathlib import Path

import click
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from neural_diffusion_processes import train as spt
from neural_diffusion_processes.paths import default_flow_dir, FLOW_DIR_SHOW_DEFAULT
from neural_diffusion_processes.core import XPoints
from neural_diffusion_processes.core import FlowProcess


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_checkpoint(run_dir: Path):
    cfg = json.loads((run_dir / "config.json").read_text())
    key0 = jax.random.PRNGKey(int(cfg["seed"]))
    _, net_key = jax.random.split(key0)
    with contextlib.redirect_stdout(io.StringIO()):
        net_template = spt.build_network(
            key=net_key,
            hidden_dim=int(cfg["hidden_dim"]),
            n_layers=int(cfg["n_layers"]),
            num_heads=int(cfg["num_heads"]),
            intermediate_size=int(cfg["intermediate_size"]),
            xy_embedding_dim=int(cfg["xy_embedding_dim"]),
            t_sin_features=int(cfg["t_sin_features"]),
            t_mlp_hidden_dim=int(cfg["t_mlp_hidden_dim"]),
            t_embedding_dim=int(cfg["t_embedding_dim"]),
            dropout_rate=float(cfg["dropout_rate"]),
            attention_dropout_rate=float(cfg["attention_dropout_rate"]),
            use_rope=bool(cfg["use_rope"]),
            fourier_x_features=int(cfg["fourier_x_features"]),
            fourier_x_base=float(cfg["fourier_x_base"]),
        )
    optim = optax.chain(
        optax.clip_by_global_norm(float(cfg["grad_clip"])),
        optax.adamw(float(cfg["lr"]), eps=1e-6),
    )
    opt_state_template = optim.init(eqx.filter(net_template, eqx.is_array))
    template = (net_template, net_template, opt_state_template, jax.random.PRNGKey(0))
    resume = run_dir / "resume.eqx"
    if resume.exists():
        _, smooth_network, _, _ = eqx.tree_deserialise_leaves(str(resume), template)
        return cfg, smooth_network
    # `resume.eqx` is the training state (net, smooth_net, opt_state, key). A
    # checkpoint directory may hold only the inference `run.eqx`, a different
    # tree, so deserialising it against the template above would fail. Load it
    # with `pipeline.load_ndp_flow_network_local` instead -- same weights, same
    # EMA.
    from neural_diffusion_processes import pipeline as ndp  # heavy; only needed here
    print(f"[load] no resume.eqx; using {run_dir}/run.eqx via "
          f"pipeline.load_ndp_flow_network_local")
    return cfg, ndp.load_ndp_flow_network_local(
        run_dir, gamma_scaling_distributional=1.0, condition_ndp_part=True)


# ---------------------------------------------------------------------------
# GP marginal log-likelihood under the data-generator's mixture kernel
# ---------------------------------------------------------------------------


def gp_log_marg_lik(
    y: jnp.ndarray,
    x: jnp.ndarray,
    period: jnp.ndarray,
    ls_per: jnp.ndarray,
    ls_rbf: jnp.ndarray,
    is_rbf_only: jnp.ndarray,
    variance: float,
    noise_var: float,
) -> jnp.ndarray:
    """log p(y | N(0, K(theta) + noise_var * I)) on the supplied x-grid.

    y: [N], x: [N, 1]. ``noise_var`` plays a Gaussian-likelihood role: it lets
    the fitter ignore sub-σ fluctuations (e.g., diffusion-sampler artifacts)
    instead of treating them as load-bearing high-frequency structure. Pass
    ``jitter`` (~1e-5) for a near-noiseless density; pass a larger value
    (e.g., 2.5e-3 = 0.05²) to recover macroscale lengthscale fits.
    """
    K = spt.mixture_gram(x, x, period, ls_per, ls_rbf, is_rbf_only, variance)
    L = jnp.linalg.cholesky(K + noise_var * jnp.eye(x.shape[0]))
    alpha = jax.scipy.linalg.solve_triangular(L, y, lower=True)
    quad = jnp.sum(alpha ** 2)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    n = y.shape[0]
    return -0.5 * quad - 0.5 * log_det - 0.5 * n * math.log(2.0 * math.pi)


def make_branch_evaluators(x: jnp.ndarray, variance: float, noise_var: float):
    """Return two JIT-compiled fns: periodic-grid eval, rbf-only-grid eval.

    Each takes a single y vector and a flat hparam grid, vmaps the
    log-likelihood across the grid, and returns shape [grid_size]. We jit
    inside the closure so x/variance/jitter bake in.
    """
    x = jnp.asarray(x, dtype=jnp.float32).reshape(-1, 1)

    def per_eval_periodic(period, ls_per, ls_rbf, y):
        return gp_log_marg_lik(
            y, x, period, ls_per, ls_rbf,
            jnp.bool_(False), variance, noise_var,
        )

    def per_eval_rbf_only(ls_rbf_only, y):
        return gp_log_marg_lik(
            y, x,
            jnp.float32(1.0), jnp.float32(1.0), ls_rbf_only,
            jnp.bool_(True), variance, noise_var,
        )

    eval_periodic = jax.jit(
        jax.vmap(per_eval_periodic, in_axes=(0, 0, 0, None))
    )
    eval_rbf_only = jax.jit(jax.vmap(per_eval_rbf_only, in_axes=(0, None)))
    return eval_periodic, eval_rbf_only


def fit_one(
    y: jnp.ndarray,
    eval_periodic,
    eval_rbf_only,
    grids: dict,
    store_log_p_grids: bool = False,
) -> dict:
    """Fit both branches for a single sample y. Returns a dict of optima.

    If ``store_log_p_grids`` is True the full log_p arrays (one per branch) are
    attached to the returned dict — needed for density-weighted population
    marginals. They are float64 numpy arrays sized
    ``[n_period * n_ls_per * n_ls_rbf]`` and ``[n_ls_rbf_only]``.
    """
    log_p_per = eval_periodic(grids["period_flat"], grids["ls_per_flat"],
                              grids["ls_rbf_flat"], y)
    log_p_rbf = eval_rbf_only(grids["ls_rbf_only"], y)

    log_p_per_np = np.asarray(log_p_per)
    log_p_rbf_np = np.asarray(log_p_rbf)

    i_per = int(np.argmax(log_p_per_np))
    i_rbf = int(np.argmax(log_p_rbf_np))
    best_per = float(log_p_per_np[i_per])
    best_rbf = float(log_p_rbf_np[i_rbf])

    period_b = float(grids["period_flat_np"][i_per])
    ls_per_b = float(grids["ls_per_flat_np"][i_per])
    ls_rbf_b = float(grids["ls_rbf_flat_np"][i_per])
    ls_rbf_only_b = float(grids["ls_rbf_only_np"][i_rbf])

    branch = "periodic" if best_per >= best_rbf else "rbf_only"

    def _at_boundary(v: float, lo: float, hi: float, n: int) -> bool:
        # On a discrete grid of n points spanning [lo, hi], a "boundary"
        # is the first or last knot. Compare with a tiny tolerance.
        eps = 1e-6 * max(1.0, hi - lo)
        return abs(v - lo) < eps or abs(v - hi) < eps

    if branch == "periodic":
        at_boundary = (
            _at_boundary(period_b, *grids["period_range"], grids["n_period"])
            or _at_boundary(ls_per_b, *grids["ls_per_range"], grids["n_ls_per"])
            or _at_boundary(ls_rbf_b, *grids["ls_rbf_range"], grids["n_ls_rbf"])
        )
    else:
        at_boundary = _at_boundary(
            ls_rbf_only_b, *grids["ls_rbf_only_range"], grids["n_ls_rbf_only"]
        )

    out = dict(
        branch=branch,
        log_p_periodic=best_per,
        log_p_rbf_only=best_rbf,
        gap=best_per - best_rbf,
        period=period_b,
        ls_per=ls_per_b,
        ls_rbf=ls_rbf_b,
        ls_rbf_only=ls_rbf_only_b,
        at_boundary=bool(at_boundary),
    )
    if store_log_p_grids:
        out["log_p_periodic_grid"] = log_p_per_np
        out["log_p_rbf_only_grid"] = log_p_rbf_np
    return out


def build_grids(cfg: dict, n_period: int, n_ls_per: int,
                n_ls_rbf: int, n_ls_rbf_only: int) -> dict:
    period_range = tuple(cfg["period_range"])
    ls_per_range = tuple(cfg["ls_per_range"])
    ls_rbf_range = tuple(cfg["ls_rbf_range"])
    ls_rbf_only_range = tuple(cfg["ls_rbf_only_range"])

    period = np.linspace(*period_range, n_period, dtype=np.float32)
    ls_per = np.linspace(*ls_per_range, n_ls_per, dtype=np.float32)
    ls_rbf = np.linspace(*ls_rbf_range, n_ls_rbf, dtype=np.float32)
    ls_rbf_only = np.linspace(*ls_rbf_only_range, n_ls_rbf_only, dtype=np.float32)

    P, L1, L2 = np.meshgrid(period, ls_per, ls_rbf, indexing="ij")
    period_flat_np = P.reshape(-1)
    ls_per_flat_np = L1.reshape(-1)
    ls_rbf_flat_np = L2.reshape(-1)

    return dict(
        period_range=period_range,
        ls_per_range=ls_per_range,
        ls_rbf_range=ls_rbf_range,
        ls_rbf_only_range=ls_rbf_only_range,
        n_period=n_period,
        n_ls_per=n_ls_per,
        n_ls_rbf=n_ls_rbf,
        n_ls_rbf_only=n_ls_rbf_only,
        period_axis=period,
        ls_per_axis=ls_per,
        ls_rbf_axis=ls_rbf,
        ls_rbf_only=jnp.asarray(ls_rbf_only),
        ls_rbf_only_np=ls_rbf_only,
        period_flat=jnp.asarray(period_flat_np),
        ls_per_flat=jnp.asarray(ls_per_flat_np),
        ls_rbf_flat=jnp.asarray(ls_rbf_flat_np),
        period_flat_np=period_flat_np,
        ls_per_flat_np=ls_per_flat_np,
        ls_rbf_flat_np=ls_rbf_flat_np,
    )


# ---------------------------------------------------------------------------
# Sanity: fit ground-truth GP draws with known hparams. If the fitter doesn't
# recover the true values within grid resolution, the model-sample fits aren't
# trustworthy.
# ---------------------------------------------------------------------------


def run_sanity(
    x: jnp.ndarray,
    cfg: dict,
    grids: dict,
    eval_periodic,
    eval_rbf_only,
    out_path: Path,
    seed: int = 1234,
) -> None:
    variance = float(cfg["variance"])
    jitter = float(cfg["jitter"])
    cases = [
        # (label, period, ls_per, ls_rbf, is_rbf_only)
        ("periodic mid", 1.0, 0.75, 2.5, False),
        ("periodic short-period", 0.8, 0.6, 1.5, False),
        ("periodic long-rbf", 1.2, 0.9, 3.5, False),
        ("rbf short", 1.0, 1.0, 0.2, True),
        ("rbf mid", 1.0, 1.0, 0.5, True),
        ("rbf long", 1.0, 1.0, 0.9, True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    rows = []
    for ax, (label, period, ls_per, ls_rbf, is_rbf_only) in zip(
        axes.flat, cases
    ):
        y = spt._sample_one(
            jax.random.PRNGKey(seed + hash(label) % 10000),
            x.reshape(-1, 1),
            jnp.float32(period), jnp.float32(ls_per), jnp.float32(ls_rbf),
            jnp.bool_(is_rbf_only),
            variance, jitter,
        )[:, 0]
        fit = fit_one(y, eval_periodic, eval_rbf_only, grids)
        rows.append((label, period, ls_per, ls_rbf, is_rbf_only, fit))
        ax.plot(np.asarray(x), np.asarray(y), color="black", lw=1.0)
        true_kind = "rbf_only" if is_rbf_only else "periodic"
        ok = fit["branch"] == true_kind
        if true_kind == "periodic":
            true_str = (f"true: per={period:.2f} ls_per={ls_per:.2f} "
                        f"ls_rbf={ls_rbf:.2f}")
            fit_str = (f"fit: per={fit['period']:.2f} ls_per={fit['ls_per']:.2f} "
                       f"ls_rbf={fit['ls_rbf']:.2f}")
        else:
            true_str = f"true: ls={ls_rbf:.2f}"
            fit_str = f"fit: ls={fit['ls_rbf_only']:.2f}"
        ax.set_title(
            f"{label}  ({'OK' if ok else 'MISS'} branch: {fit['branch']})\n"
            f"{true_str}\n{fit_str}",
            fontsize=8,
        )
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")
    for r in rows:
        print(f"  {r[0]:>22s} -> branch={r[5]['branch']:<8s} "
              f"log_p_per={r[5]['log_p_periodic']:>9.2f} "
              f"log_p_rbf={r[5]['log_p_rbf_only']:>9.2f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def draw_under_kernel(
    x: jnp.ndarray, fit: dict, variance: float, jitter: float,
    n: int, key: jax.Array,
) -> np.ndarray:
    """Draw n GP samples under the best-fit kernel for visual comparison."""
    if fit["branch"] == "periodic":
        period = jnp.float32(fit["period"])
        ls_per = jnp.float32(fit["ls_per"])
        ls_rbf = jnp.float32(fit["ls_rbf"])
        is_rbf_only = jnp.bool_(False)
    else:
        period = jnp.float32(1.0)
        ls_per = jnp.float32(1.0)
        ls_rbf = jnp.float32(fit["ls_rbf_only"])
        is_rbf_only = jnp.bool_(True)
    keys = jax.random.split(key, n)
    ys = jax.vmap(
        lambda k: spt._sample_one(
            k, x.reshape(-1, 1), period, ls_per, ls_rbf, is_rbf_only,
            variance, jitter,
        )[:, 0]
    )(keys)
    return np.asarray(ys)


def plot_panel(
    out_path: Path,
    x: np.ndarray,
    y_samples: np.ndarray,
    fits: list[dict],
    cfg: dict,
    n_show: int = 8,
    n_overlay: int = 3,
    seed: int = 7,
) -> None:
    n_show = min(n_show, len(fits))
    cols = 4
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.6 * rows),
                             sharex=True)
    axes = np.atleast_2d(axes)
    variance = float(cfg["variance"])
    jitter = float(cfg["jitter"])
    y_range = tuple(cfg["y_range"])
    keys = jax.random.split(jax.random.PRNGKey(seed), n_show)
    for i in range(n_show):
        ax = axes.flat[i]
        fit = fits[i]
        ys = draw_under_kernel(
            jnp.asarray(x), fit, variance, jitter, n_overlay, keys[i],
        )
        for j in range(n_overlay):
            ax.plot(x, ys[j], color="tab:orange", lw=0.7, alpha=0.55)
        ax.plot(x, y_samples[i], color="black", lw=1.2)
        if fit["branch"] == "periodic":
            branch_label = "periodic"
            hp = (f"per={fit['period']:.2f}  ls_p={fit['ls_per']:.2f}  "
                  f"ls_r={fit['ls_rbf']:.2f}")
        else:
            branch_label = "rbf"
            hp = f"ls_rbf={fit['ls_rbf_only']:.2f}"
        ax.set_title(
            rf"$\mathbf{{\#{i}\ {branch_label}}}$" + f"\n{hp}",
            fontsize=11,
        )
        ax.set_ylim(y_range)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.2)
    for j in range(n_show, rows * cols):
        axes.flat[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path}")


def _soft_population_marginals(
    log_p_per_all: np.ndarray,  # [N, n_period * n_ls_per * n_ls_rbf]
    log_p_rbf_all: np.ndarray,  # [N, n_ls_rbf_only]
    grids: dict,
    p_rbf_only: float,
) -> dict:
    """Density-weighted population marginals over each kernel hparam.

    For each sample i we compute:
      - branch posterior  P(branch | y_i)  via the per-branch
        evidence  log E_i = log mean_θ exp(log p(y_i | θ))  with a uniform
        prior over θ inside the training range, combined with the Bernoulli
        branch prior `rbf_only_prob`.
      - within-branch posterior  P(θ | y_i, branch)  ∝ exp(log p(y_i | θ)),
        which behaves like a soft tent around the MAP so high-confidence
        samples contribute a sharp spike and uncertain ones a wide bump.

    The population marginal is `(1/N) Σ_i P(branch | y_i) · P(θ | y_i, branch)`,
    summed over the marginalized hparams. Returns probability masses on the
    same grid axes used by the fit, so visualizing as a step / bar over the
    knots is exact (no rebinning).
    """
    from scipy.special import logsumexp

    n_period = grids["n_period"]
    n_ls_per = grids["n_ls_per"]
    n_ls_rbf = grids["n_ls_rbf"]
    n_ls_rbf_only = grids["n_ls_rbf_only"]
    n_per_grid = log_p_per_all.shape[1]
    n_rbf_grid = log_p_rbf_all.shape[1]
    n_samples = log_p_per_all.shape[0]

    # Per-branch log-evidence per sample (uniform prior over θ in the grid):
    # log E = logmeanexp(log_p) = logsumexp(log_p) - log(grid_size)
    log_E_per = logsumexp(log_p_per_all, axis=1) - math.log(n_per_grid)
    log_E_rbf = logsumexp(log_p_rbf_all, axis=1) - math.log(n_rbf_grid)

    # Branch posterior per sample: log p(branch | y) ∝ log_prior + log E.
    log_pi_per = math.log(max(1e-12, 1.0 - p_rbf_only))
    log_pi_rbf = math.log(max(1e-12, p_rbf_only))
    log_post_per = log_pi_per + log_E_per
    log_post_rbf = log_pi_rbf + log_E_rbf
    log_norm = np.logaddexp(log_post_per, log_post_rbf)
    w_per = np.exp(log_post_per - log_norm)  # [N]
    w_rbf = np.exp(log_post_rbf - log_norm)  # [N]

    # Within-branch soft posterior over θ (subtract per-row max for stability,
    # then exponentiate and normalize to sum to 1 along θ).
    def _row_softmax(log_p: np.ndarray) -> np.ndarray:
        m = log_p.max(axis=1, keepdims=True)
        e = np.exp(log_p - m)
        return e / e.sum(axis=1, keepdims=True)

    p_theta_per = _row_softmax(log_p_per_all)  # [N, n_per_grid]
    p_theta_rbf = _row_softmax(log_p_rbf_all)  # [N, n_rbf_grid]

    # Population marginals — sum across samples weighted by branch posterior.
    pop_per = (w_per[:, None] * p_theta_per).mean(axis=0)  # [n_per_grid]
    pop_rbf = (w_rbf[:, None] * p_theta_rbf).mean(axis=0)  # [n_rbf_grid]

    # Reshape periodic grid back to [n_period, n_ls_per, n_ls_rbf] (in the same
    # 'ij' order used by build_grids' meshgrid) and marginalize.
    pop_per_3d = pop_per.reshape(n_period, n_ls_per, n_ls_rbf)
    pop_period = pop_per_3d.sum(axis=(1, 2))    # marginalize ls_per, ls_rbf
    pop_ls_per = pop_per_3d.sum(axis=(0, 2))    # marginalize period, ls_rbf
    pop_ls_rbf = pop_per_3d.sum(axis=(0, 1))    # marginalize period, ls_per

    # The three periodic marginals share the same total mass (mean of w_per
    # across samples). The rbf_only marginal uses w_rbf. Both sum to numbers
    # in [0, 1] reflecting the population branch share.
    return dict(
        w_per=w_per, w_rbf=w_rbf,
        pop_period=pop_period,
        pop_ls_per=pop_ls_per,
        pop_ls_rbf=pop_ls_rbf,
        pop_ls_rbf_only=pop_rbf,
    )


def plot_marginals(out_path: Path, fits: list[dict], cfg: dict,
                   grids: dict, run_label: str) -> None:
    """Per-hparam population marginals vs the uniform training prior.

    Two overlapping views per panel:
      - Bars at each grid knot, density of MAP fits per branch (counts /
        n_branch / step), bin-aligned to the fit's discrete grid so no
        bin is artificially empty.
      - Red curve: density-weighted population marginal — soft posterior
        over θ per sample, weighted by per-sample branch posterior, summed
        across samples. High-confidence samples sharpen a knot, uncertain
        ones spread mass (so the curve never drops to zero on a knot we
        actually have evidence for).
      - Dashed: uniform-prior pdf 1/(hi-lo). Dotted verticals: prior bounds.

    The branch panel additionally shows the density-weighted soft branch
    counts (Σ_i w_per, Σ_i w_rbf) alongside the hard MAP counts.
    """
    log_p_per_all = np.asarray(
        [f["log_p_periodic_grid"] for f in fits], dtype=np.float64,
    )
    log_p_rbf_all = np.asarray(
        [f["log_p_rbf_only_grid"] for f in fits], dtype=np.float64,
    )
    p_rbf_only = float(cfg["rbf_only_prob"])
    soft = _soft_population_marginals(
        log_p_per_all, log_p_rbf_all, grids, p_rbf_only,
    )

    fig, axes = plt.subplots(1, 5, figsize=(23, 4.4))

    # ----- branch: hard counts + soft (density-weighted) counts -----
    ax = axes[0]
    n_per = sum(1 for f in fits if f["branch"] == "periodic")
    n_rbf = sum(1 for f in fits if f["branch"] == "rbf_only")
    n_total = n_per + n_rbf
    soft_n_per = float(soft["w_per"].sum())
    soft_n_rbf = float(soft["w_rbf"].sum())
    bar_pos = np.arange(2)
    bw = 0.38
    ax.bar(bar_pos - bw / 2, [n_per, n_rbf], width=bw,
           color=["tab:blue", "tab:orange"], alpha=0.7,
           edgecolor="black", linewidth=0.4, label="MAP")
    ax.bar(bar_pos + bw / 2, [soft_n_per, soft_n_rbf], width=bw,
           color=["tab:blue", "tab:orange"], alpha=0.4, hatch="//",
           edgecolor="black", linewidth=0.4, label="density-weighted")
    ax.axhline((1.0 - p_rbf_only) * n_total, ls="--", color="tab:blue",
               lw=1.2, label=f"prior periodic = {(1.0 - p_rbf_only):.0%}")
    ax.axhline(p_rbf_only * n_total, ls="--", color="tab:orange",
               lw=1.2, label=f"prior rbf_only = {p_rbf_only:.0%}")
    ax.set_xticks(bar_pos)
    ax.set_xticklabels(["periodic", "rbf_only"])
    ax.set_ylabel("count")
    ax.set_title(f"branch (N = {n_total})")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)

    per_fits = [f for f in fits if f["branch"] == "periodic"]
    rbf_fits = [f for f in fits if f["branch"] == "rbf_only"]

    def _knot_panel(ax, vals, axis, rng, name, color, n_branch, soft_density):
        """vals: list of MAP knot values. axis: 1d array of grid knots.
        soft_density: probability mass over each knot summing to the branch
        share (∈ [0, 1]); we divide by step to put it on the same density
        axis as the bar / prior pdf.
        """
        n_knots = len(axis)
        step = float(axis[1] - axis[0]) if n_knots > 1 else 1.0
        lo, hi = float(rng[0]), float(rng[1])
        # MAP density at each knot.
        if vals:
            counts = np.zeros(n_knots, dtype=np.float64)
            for v in vals:
                idx = int(np.argmin(np.abs(axis - v)))
                counts[idx] += 1
            map_density = counts / (n_branch * step)
            ax.bar(axis, map_density, width=0.85 * step,
                   color=color, alpha=0.6, edgecolor="black", linewidth=0.4,
                   label=f"MAP (n={n_branch})")
        else:
            ax.text(0.5, 0.55, f"no {name} fits", transform=ax.transAxes,
                    ha="center", va="center")
        # Density-weighted population marginal. Soft mass per knot sums to
        # the branch share; divide by branch share to plot a normalized
        # density comparable to the MAP histogram and the uniform prior.
        soft_share = float(soft_density.sum())
        if soft_share > 0.0:
            soft_pdf = (soft_density / soft_share) / step
            ax.plot(axis, soft_pdf, color="tab:red", lw=1.8,
                    marker="o", markersize=4, label="density-weighted")
        ax.axhline(1.0 / (hi - lo), ls="--", color="black", lw=1.0,
                   label="uniform prior pdf")
        ax.axvline(lo, color="gray", ls=":", lw=0.8)
        ax.axvline(hi, color="gray", ls=":", lw=0.8)
        # Pad x to show knots cleanly.
        x_pad = 0.5 * step
        ax.set_xlim(axis[0] - x_pad, axis[-1] + x_pad)
        ax.set_xlabel(name)
        ax.set_ylabel("density")
        ax.set_title(f"{name}  (n = {n_branch})")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

    _knot_panel(
        axes[1], [f["period"] for f in per_fits],
        grids["period_axis"], cfg["period_range"], "period",
        "tab:blue", len(per_fits), soft["pop_period"],
    )
    _knot_panel(
        axes[2], [f["ls_per"] for f in per_fits],
        grids["ls_per_axis"], cfg["ls_per_range"], "ls_per",
        "tab:blue", len(per_fits), soft["pop_ls_per"],
    )
    _knot_panel(
        axes[3], [f["ls_rbf"] for f in per_fits],
        grids["ls_rbf_axis"], cfg["ls_rbf_range"], "ls_rbf  (periodic branch)",
        "tab:blue", len(per_fits), soft["pop_ls_rbf"],
    )
    _knot_panel(
        axes[4], [f["ls_rbf_only"] for f in rbf_fits],
        grids["ls_rbf_only_np"], cfg["ls_rbf_only_range"], "ls_rbf_only",
        "tab:orange", len(rbf_fits), soft["pop_ls_rbf_only"],
    )

    fig.suptitle(run_label, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Batched unconditional sampling — vmap over keys for speed
# ---------------------------------------------------------------------------


def batched_gt_samples(
    cfg: dict,
    x_grid: jnp.ndarray,
    n_samples: int,
    seed: int,
    chunk_size: int,
) -> tuple[np.ndarray, dict]:
    """Draw `n_samples` ground-truth GP samples from the training prior on the
    fixed `x_grid`, returning (y_samples [N, len(x_grid)], true_hparams dict).

    Uses ``_spt._sample_one`` per draw, vmapped in chunks so the cholesky
    cost stays bounded. Hparams are sampled from the training ranges in
    config.json (linear-uniform), with branch chosen by Bernoulli on
    ``rbf_only_prob``. Mirrors the data-generating distribution exactly so a
    fit on these samples is the unbiased reference.
    """
    period_range = tuple(cfg["period_range"])
    ls_per_range = tuple(cfg["ls_per_range"])
    ls_rbf_range = tuple(cfg["ls_rbf_range"])
    ls_rbf_only_range = tuple(cfg["ls_rbf_only_range"])
    rbf_only_prob = float(cfg["rbf_only_prob"])
    variance = float(cfg["variance"])
    jitter = float(cfg["jitter"])

    key = jax.random.PRNGKey(seed)
    k_per, k_lp, k_lr, k_lro, k_rbo, k_y = jax.random.split(key, 6)
    period = jax.random.uniform(
        k_per, (n_samples,), minval=period_range[0], maxval=period_range[1],
    )
    ls_per = jax.random.uniform(
        k_lp, (n_samples,), minval=ls_per_range[0], maxval=ls_per_range[1],
    )
    ls_rbf_periodic = jax.random.uniform(
        k_lr, (n_samples,), minval=ls_rbf_range[0], maxval=ls_rbf_range[1],
    )
    ls_rbf_only_val = jax.random.uniform(
        k_lro, (n_samples,),
        minval=ls_rbf_only_range[0], maxval=ls_rbf_only_range[1],
    )
    is_rbf_only = jax.random.bernoulli(
        k_rbo, p=rbf_only_prob, shape=(n_samples,),
    )
    ls_rbf = jnp.where(is_rbf_only, ls_rbf_only_val, ls_rbf_periodic)
    sample_keys = jax.random.split(k_y, n_samples)

    x_2d = x_grid.reshape(-1, 1)

    @eqx.filter_jit
    def _chunk(per_, lsp_, lsr_, irb_, sk_):
        def _one(k, p, lp, lr, ir):
            return spt._sample_one(
                k, x_2d, p, lp, lr, ir, variance, jitter,
            )[:, 0]
        return jax.vmap(_one)(sk_, per_, lsp_, lsr_, irb_)

    n_grid = x_grid.shape[0]
    out = np.empty((n_samples, n_grid), dtype=np.float32)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, n_samples)
        y_batch = _chunk(period[lo:hi], ls_per[lo:hi], ls_rbf[lo:hi],
                         is_rbf_only[lo:hi], sample_keys[lo:hi])
        out[lo:hi] = np.asarray(y_batch)
        if (ci + 1) % max(1, n_chunks // 20) == 0 or ci == n_chunks - 1:
            print(f"  sampled {hi}/{n_samples}")

    true_hparams = dict(
        period=np.asarray(period),
        ls_per=np.asarray(ls_per),
        ls_rbf=np.asarray(ls_rbf),
        is_rbf_only=np.asarray(is_rbf_only),
        ls_rbf_periodic=np.asarray(ls_rbf_periodic),
        ls_rbf_only=np.asarray(ls_rbf_only_val),
    )
    return out, true_hparams


def batched_unconditional_samples(
    smooth_network,
    process: FlowProcess,
    target: XPoints,
    n_samples: int,
    seed: int,
    chunk_size: int,
) -> np.ndarray:
    """Draw `n_samples` curves on `target.x` by vmapping process.sample over
    keys in chunks. Returns shape [n_samples, len(target.x)]."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n_samples)

    @eqx.filter_jit
    def _chunk(net, ks):
        def _one(k):
            y_pts, _ = process.sample(net, target, k)
            return y_pts.y[:, 0]
        return jax.vmap(_one)(ks)

    n_grid = target.x.shape[0]
    out = np.empty((n_samples, n_grid), dtype=np.float32)
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, n_samples)
        chunk_keys = keys[lo:hi]
        y_batch = _chunk(smooth_network, chunk_keys)
        out[lo:hi] = np.asarray(y_batch)
        if (ci + 1) % max(1, n_chunks // 20) == 0 or ci == n_chunks - 1:
            print(f"  sampled {hi}/{n_samples}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option("--run-dir", default=default_flow_dir,
              show_default=FLOW_DIR_SHOW_DEFAULT,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", default="outputs/fit_mixture_hparams",
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--n-samples", default=16, type=int,
              help="Number of unconditional model samples to draw.")
@click.option("--n-grid", default=256, type=int,
              help="x-grid size; default matches training num_points.")
@click.option("--seed", default=0, type=int)
@click.option("--n-period", default=11, type=int)
@click.option("--n-ls-per", default=11, type=int)
@click.option("--n-ls-rbf", default=11, type=int)
@click.option("--n-ls-rbf-only", default=31, type=int)
@click.option("--noise-std", default=0.05, type=float, show_default=True,
              help="Likelihood noise stddev added in K + sigma^2 * I. "
                   "Default 0.05 ignores sub-sigma high-frequency content so "
                   "the fitter recovers macroscale lengthscale fits. Set to "
                   "sqrt(jitter) (~3e-3) for near-noiseless fits.")
@click.option("--sample-chunk-size", default=32, type=int,
              help="Vmap chunk size for unconditional sampling.")
@click.option("--source", "source", default="model",
              type=click.Choice(["model", "gt"]),
              help="model = sample from the trained NDP checkpoint (default). "
                   "gt = sample from the data-generating GP prior with the same "
                   "hparam ranges/branch prior. Use 'gt' to baseline the fitter.")
@click.option("--sanity/--no-sanity", default=True,
              help="Run a fitter-recovery check on GT GP draws first.")
@click.option("--sampler-n-steps", type=int, default=None,
              help="Override VelocitySampler.n_steps (default = sampler "
                   "default in neural_diffusion_processes/core.py).")
@click.option("--sampler-num-corrections", type=int, default=None,
              help="Override VelocitySampler.num_corrections. Pass 0 to "
                   "disable Langevin correctors (pure flow ODE).")
@click.option("--sampler-tau", type=float, default=None,
              help="Override VelocitySampler.tau (LMC step-size scale).")
def main(
    run_dir: Path, out_dir: Path, n_samples: int, n_grid: int, seed: int,
    n_period: int, n_ls_per: int, n_ls_rbf: int, n_ls_rbf_only: int,
    noise_std: float | None, sample_chunk_size: int,
    source: str, sanity: bool,
    sampler_n_steps: int | None,
    sampler_num_corrections: int | None,
    sampler_tau: float | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if source == "model":
        print(f"loading checkpoint from {run_dir}")
        cfg, smooth_network = load_checkpoint(run_dir)
    else:
        print(f"source=gt: skipping checkpoint, loading config only from {run_dir}")
        cfg = json.loads((run_dir / "config.json").read_text())
        smooth_network = None
    x_range = tuple(cfg["x_range"])
    y_range = tuple(cfg["y_range"])
    variance = float(cfg["variance"])
    jitter = float(cfg["jitter"])
    print(f"  x_range={x_range}  y_range={y_range}  variance={variance}  "
          f"jitter={jitter}")

    process = FlowProcess(x_range=x_range, y_range=y_range)
    # Optional sampler override. Monkey-patches FlowProcess._get_sampler so
    # batched_unconditional_samples uses the new config without rewriting it.
    if (sampler_n_steps is not None or sampler_num_corrections is not None
            or sampler_tau is not None):
        from neural_diffusion_processes.core import VelocitySampler
        from neural_diffusion_processes.core import DEFAULT_STEPS, DEFAULT_CORRECTIONS, DEFAULT_TAU
        ns = sampler_n_steps if sampler_n_steps is not None else DEFAULT_STEPS
        nc = (sampler_num_corrections if sampler_num_corrections is not None
              else DEFAULT_CORRECTIONS)
        tau_v = sampler_tau if sampler_tau is not None else DEFAULT_TAU
        print(f"  sampler override: n_steps={ns}  num_corrections={nc}  tau={tau_v}")
        def _patched_get_sampler(self):
            return VelocitySampler(
                start_time=0.1, n_steps=ns, num_corrections=nc, tau=tau_v,
            )
        FlowProcess._get_sampler = _patched_get_sampler
    x_grid = jnp.linspace(x_range[0], x_range[1], n_grid).reshape(-1, 1)
    target = XPoints(x=x_grid, mask=jnp.zeros(n_grid, dtype=bool)) if source == "model" else None

    grids = build_grids(cfg, n_period, n_ls_per, n_ls_rbf, n_ls_rbf_only)
    print(f"hparam grid sizes: periodic={n_period * n_ls_per * n_ls_rbf}  "
          f"rbf_only={n_ls_rbf_only}")

    # Likelihood noise: if --noise-std is set, dominate jitter so the fitter
    # ignores sub-sigma fluctuations. Otherwise stay near-noiseless (jitter).
    noise_var = jitter if noise_std is None else float(noise_std) ** 2
    print(f"likelihood noise variance = {noise_var:.3g}  "
          f"(noise_std = {math.sqrt(noise_var):.4g})")

    eval_periodic, eval_rbf_only = make_branch_evaluators(
        x_grid[:, 0], variance, noise_var,
    )

    if sanity:
        print("\nrunning fitter sanity check on GT GP draws ...")
        run_sanity(
            x_grid[:, 0], cfg, grids, eval_periodic, eval_rbf_only,
            out_dir / "sanity.png",
        )

    # Batched vmap-over-keys sampling. process.sample is filter_jit'd, so
    # vmapping the closure compiles once per chunk size and reuses for the
    # remaining chunks. Lower --sample-chunk-size if a chunk runs out of GPU
    # memory. The 'gt' branch instead draws from the training prior directly.
    print(f"\ndrawing {n_samples} unconditional samples on N={n_grid} grid "
          f"(source={source}, chunk_size={sample_chunk_size}) ...")
    true_hparams: dict | None = None
    if source == "model":
        y_samples = batched_unconditional_samples(
            smooth_network, process, target,
            n_samples, seed, sample_chunk_size,
        )
    else:
        y_samples, true_hparams = batched_gt_samples(
            cfg, x_grid[:, 0], n_samples, seed, sample_chunk_size,
        )

    # Fit each sample. For large N we print only the first/last few and a
    # progress counter; the full per-sample table lands in results.csv.
    print("\nfitting hparams per sample ...")
    fits: list[dict] = []
    print_each = n_samples <= 32
    for i in range(n_samples):
        fit = fit_one(
            jnp.asarray(y_samples[i]),
            eval_periodic, eval_rbf_only, grids,
            store_log_p_grids=True,
        )
        fits.append(fit)
        if print_each:
            if fit["branch"] == "periodic":
                hp = (f"per={fit['period']:.2f} ls_per={fit['ls_per']:.2f} "
                      f"ls_rbf={fit['ls_rbf']:.2f}")
            else:
                hp = f"ls_rbf_only={fit['ls_rbf_only']:.2f}"
            print(
                f"  #{i:02d} branch={fit['branch']:<8s}  {hp}   "
                f"log_p_per={fit['log_p_periodic']:>9.2f}  "
                f"log_p_rbf={fit['log_p_rbf_only']:>9.2f}  "
                f"gap={fit['gap']:+.2f}  "
                f"{'(boundary)' if fit['at_boundary'] else ''}"
            )
        elif (i + 1) % max(1, n_samples // 20) == 0 or i + 1 == n_samples:
            print(f"  fitted {i + 1}/{n_samples}")

    n_per = sum(1 for f in fits if f["branch"] == "periodic")
    n_rbf = sum(1 for f in fits if f["branch"] == "rbf_only")
    n_boundary = sum(1 for f in fits if f["at_boundary"])
    print(f"\nbranch counts: periodic={n_per}  rbf_only={n_rbf}  "
          f"(prior rbf_only_prob={cfg['rbf_only_prob']})")
    print(f"boundary hits: {n_boundary} / {n_samples}  "
          f"(model produced samples outside the data-generator's prior range)")

    # Persist results.
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sample", "branch", "log_p_periodic", "log_p_rbf_only", "gap",
            "period", "ls_per", "ls_rbf", "ls_rbf_only", "at_boundary",
        ])
        for i, fit in enumerate(fits):
            w.writerow([
                i, fit["branch"],
                f"{fit['log_p_periodic']:.6g}",
                f"{fit['log_p_rbf_only']:.6g}",
                f"{fit['gap']:.6g}",
                f"{fit['period']:.6g}", f"{fit['ls_per']:.6g}",
                f"{fit['ls_rbf']:.6g}", f"{fit['ls_rbf_only']:.6g}",
                int(fit["at_boundary"]),
            ])
    print(f"wrote {csv_path}")

    npz_path = out_dir / "results.npz"
    np.savez(
        npz_path,
        x_grid=np.asarray(x_grid)[:, 0],
        y_samples=y_samples,
        branch=np.array([f["branch"] for f in fits]),
        log_p_periodic=np.array([f["log_p_periodic"] for f in fits]),
        log_p_rbf_only=np.array([f["log_p_rbf_only"] for f in fits]),
        gap=np.array([f["gap"] for f in fits]),
        period=np.array([f["period"] for f in fits]),
        ls_per=np.array([f["ls_per"] for f in fits]),
        ls_rbf=np.array([f["ls_rbf"] for f in fits]),
        ls_rbf_only=np.array([f["ls_rbf_only"] for f in fits]),
        at_boundary=np.array([f["at_boundary"] for f in fits]),
        # Full per-sample log_p grids — let downstream code redo soft-marginal
        # weighting with different priors / branch combinations without
        # re-running the diffusion sampler.
        log_p_periodic_grid=np.asarray(
            [f["log_p_periodic_grid"] for f in fits], dtype=np.float64,
        ),
        log_p_rbf_only_grid=np.asarray(
            [f["log_p_rbf_only_grid"] for f in fits], dtype=np.float64,
        ),
        period_axis=grids["period_axis"],
        ls_per_axis=grids["ls_per_axis"],
        ls_rbf_axis=grids["ls_rbf_axis"],
        ls_rbf_only_axis=grids["ls_rbf_only_np"],
        period_range=np.array(grids["period_range"]),
        ls_per_range=np.array(grids["ls_per_range"]),
        ls_rbf_range=np.array(grids["ls_rbf_range"]),
        ls_rbf_only_range=np.array(grids["ls_rbf_only_range"]),
    )
    print(f"wrote {npz_path}")

    plot_panel(
        out_dir / "panel.png",
        np.asarray(x_grid)[:, 0],
        y_samples, fits, cfg,
        n_show=min(n_samples, 12),
        seed=seed + 1,
    )

    plot_marginals(
        out_dir / "marginals.png",
        fits, cfg, grids,
        run_label=(f"{run_dir.name}  |  source = {source}  "
                   f"|  N = {n_samples}  "
                   f"|  noise_std = {math.sqrt(noise_var):.4g}"),
    )

    if true_hparams is not None:
        # Stash the true hparams alongside results.npz so downstream code can
        # plot fitted-vs-true scatter / build calibration curves.
        true_npz = out_dir / "true_hparams.npz"
        np.savez(true_npz, **true_hparams)
        print(f"wrote {true_npz}")


if __name__ == "__main__":
    main()
