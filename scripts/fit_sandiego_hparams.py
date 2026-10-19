"""Posterior hyperparameter marginals for the sandiego NDP and LLM-FP samples.

Applies the same evidence-based machinery as `fit_mixture_hparams_bayes.py` to
the conditional samples in `<preset><suffix>_samples.npz` caches (as written
by `sample_for_hparams.py`, or any `run.cached_sample_for_preset` call), so the
question "what kernel does each method's output look like?" is answered on the
same footing as the unconditional analysis.

Coordinates. The samples live in the demo's own units (months on x, inches on
y). The kernel prior ranges are defined in the *training* coordinate system, so
each sample is mapped through the same normalisation the network sees:

    x_fit = ((x - x0) / (x1 - x0) - 0.5) * 4          -> [-2, 2]
    y_fit = ((y - y0) / (y1 - y0) - 0.5) * 2 * 3.5    -> training y scale

The trailing 3.5 undoes the training y_range, so a unit-variance draw from the
data generator maps to unit variance here and `variance=1.0` in the kernel is
the right scale. A 12-month season becomes a normalised period of 0.96, near
the middle of the generator's [0.75, 1.25] range.

Two variants are fitted:

  raw      : each sample treated as a zero-mean GP draw, exactly as the
             unconditional analysis does.
  centred  : the across-sample ensemble mean removed first. These are
             *conditional* samples — LLM-FP in particular shares a strong
             seasonal mean across draws, which a zero-mean fit books as
             signal. The centred fit asks the narrower question of what
             kernel describes the sample-to-sample spread.

Both are reported because they answer different questions and can disagree,
LLM-FP in particular.

`--gt-ref` overlays the unconditional training-prior marginals. It is produced
in two steps -- GT-prior draws fitted by MAP, then the Bayesian fit on those
draws at this script's grid size (the overlay is skipped if the axes differ):

    uv run python scripts/fit_mixture_fourier_hparams.py \
        --source gt --out-dir outputs/fit_mixture_hparams_gt
    uv run python scripts/fit_mixture_hparams_bayes.py \
        --fit-dir outputs/fit_mixture_hparams_gt \
        --out-dir outputs/fit_mixture_hparams_gt_bayes --n-period 128

Outputs (under --out-dir):
  - marginals_<preset>.png : population marginals, NDP vs LLM-FP, per variant.
  - summary.csv            : branch share and marginal means per fit.

Example (reads the `_ncorr0` caches written by the `sample_for_hparams.py`
example)
-------
    uv run python scripts/fit_sandiego_hparams.py \
        --out-dir outputs/sandiego_ncorr0_hparams
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import csv
import json
from pathlib import Path

import click
from neural_diffusion_processes.paths import default_flow_dir, FLOW_DIR_SHOW_DEFAULT
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fit_mixture_hparams_bayes import (
    branch_posterior, log_evidence, log_lik_periodic_grid, make_evaluators,
    marginalise, midpoints, normalised_weights,
)

# (label, key in the samples npz)
METHODS = (("NDP", "y_ndp"), ("LLM-FP", "y_ndp_llm"))
COLOURS = {"NDP": "#C44E52", "LLM-FP": "#4C72B0"}
KEYS = ("period", "ls_per", "ls_rbf", "ls_rbf_only")
LABELS = {"period": "period", "ls_per": r"$\ell_{per}$",
          "ls_rbf": r"$\ell_{rbf}$",
          "ls_rbf_only": r"$\ell_{rbf}$ (rbf-only branch)"}


def _default_samples_dir() -> Path:
    """`run.OUTPUTS_DIR`, imported only when the default is actually used."""
    from neural_diffusion_processes.run import OUTPUTS_DIR
    return OUTPUTS_DIR


def normalise(x: np.ndarray, y: np.ndarray, x_range, y_range,
              train_y_half: float):
    """Demo units -> the training coordinate system (see module docstring)."""
    x0, x1 = x_range
    y0, y1 = y_range
    x_fit = ((x - x0) / (x1 - x0) - 0.5) * 4.0
    y_fit = ((y - y0) / (y1 - y0) - 0.5) * 2.0 * train_y_half
    return x_fit.astype(np.float64), y_fit.astype(np.float64)


def fit_one(x_fit, y_fit, cfg, axes, noise_std, theta_chunk):
    """Evidence fit -> (branch share, population marginals, per-sample w_per)."""
    variance = float(cfg["variance"])
    p_rbf_only = float(cfg["rbf_only_prob"])
    Y = jnp.asarray(y_fit.T, dtype=jnp.float32)          # [n_pts, N]
    eval_per, eval_rbf = make_evaluators(
        x_fit, variance, float(noise_std) ** 2)

    log_p_per = log_lik_periodic_grid(
        eval_per, (axes["period"], axes["ls_per"], axes["ls_rbf"]),
        Y, theta_chunk, verbose=False)
    log_p_rbf = np.asarray(
        eval_rbf(jnp.asarray(axes["ls_rbf_only"]), Y)[0])

    w_per, w_rbf = branch_posterior(
        log_evidence(log_p_per), log_evidence(log_p_rbf), p_rbf_only)
    post_per = normalised_weights(log_p_per)
    post_rbf = normalised_weights(log_p_rbf)
    m_period, m_ls_per, m_ls_rbf = marginalise(
        post_per, (len(axes["period"]), len(axes["ls_per"]),
                   len(axes["ls_rbf"])))
    pop = dict(
        period=(w_per * m_period).mean(axis=1),
        ls_per=(w_per * m_ls_per).mean(axis=1),
        ls_rbf=(w_per * m_ls_rbf).mean(axis=1),
        ls_rbf_only=(w_rbf * post_rbf).mean(axis=1),
    )
    return float((w_per > 0.5).mean()), pop, w_per


def plot_preset(out_path, preset, axes, results, gt_pop, period_months, label):
    variants = list(results)
    fig, axs = plt.subplots(len(variants), 4,
                            figsize=(21, 4.4 * len(variants)), squeeze=False)
    for r, variant in enumerate(variants):
        for c, key in enumerate(KEYS):
            ax = axs[r][c]
            knots = axes[key]
            width = float(knots[1] - knots[0])
            if gt_pop is not None and key in gt_pop:
                g = gt_pop[key]
                ax.plot(knots, g / g.sum() / width, color="black", ls="--",
                        lw=1.6, label="training prior (unconditional GT)")
            for method, _ in METHODS:
                share, pop, _ = results[variant][method]
                p = pop[key]
                ax.plot(knots, p / p.sum() / width, color=COLOURS[method],
                        lw=1.9, label=f"{method} (periodic {share:.2f})")
            lo, hi = knots[0] - width / 2, knots[-1] + width / 2
            ax.axhline(1.0 / (hi - lo), color="0.7", ls=":", lw=1.2)
            if key == "period" and period_months is not None:
                ax.axvline(period_months, color="#55A868", lw=1.6, alpha=0.8,
                           label="12-month season")
            ax.set_title(f"{LABELS[key]}  [{variant}]", fontsize=11)
            ax.set_xlabel(LABELS[key])
            ax.grid(True, alpha=0.2)
        axs[r][0].set_ylabel("density")
        axs[r][0].legend(fontsize=8)
    fig.suptitle(f"{preset} [{label}]: hyperparameter posteriors of the "
                 f"generated samples", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out_path}")


@click.command()
@click.option("--samples-dir", default=_default_samples_dir,
              show_default="$NDP_OUTPUTS_DIR, else outputs/",
              type=click.Path(file_okay=False, path_type=Path),
              help="Directory holding the <preset><suffix>_samples.npz caches.")
@click.option("--suffix", default="_ncorr0", show_default=True,
              help="Sample-cache suffix, e.g. the --cache-suffix given to "
                   "sample_for_hparams.py. Empty string reads "
                   "<preset>_samples.npz.")
@click.option("--label", default=None,
              help="Text identifying the sampler setting in plot titles. "
                   "Defaults to the suffix, or 'original' when it is empty.")
@click.option("--presets", default="sandiego,sandiego_long", show_default=True)
@click.option("--run-dir", default=default_flow_dir,
              show_default=FLOW_DIR_SHOW_DEFAULT,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--gt-ref", default="outputs/fit_mixture_hparams_gt_bayes/results_bayes.npz",
              type=click.Path(path_type=Path),
              help="Unconditional training-prior marginals, drawn as reference "
                   "(see the module docstring for the commands that produce "
                   "it). Must share the grid axes; skipped if missing.")
@click.option("--n-period", default=128, type=int, show_default=True)
@click.option("--n-ls-per", default=32, type=int, show_default=True)
@click.option("--n-ls-rbf", default=32, type=int, show_default=True)
@click.option("--n-ls-rbf-only", default=128, type=int, show_default=True)
@click.option("--noise-std", default=0.05, type=float, show_default=True)
@click.option("--theta-chunk", default=64, type=int, show_default=True)
def main(samples_dir, suffix, label, presets, run_dir, out_dir, gt_ref,
         n_period, n_ls_per, n_ls_rbf, n_ls_rbf_only, noise_std, theta_chunk):
    from neural_diffusion_processes import pipeline as ndp

    label = label or (suffix.lstrip("_") or "original")

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run_dir / "config.json").read_text())
    train_y_half = float(cfg["y_range"][1])
    ranges = {k: tuple(cfg[f"{k}_range"])
              for k in ("period", "ls_per", "ls_rbf", "ls_rbf_only")}
    axes = dict(
        period=midpoints(*ranges["period"], n_period),
        ls_per=midpoints(*ranges["ls_per"], n_ls_per),
        ls_rbf=midpoints(*ranges["ls_rbf"], n_ls_rbf),
        ls_rbf_only=midpoints(*ranges["ls_rbf_only"], n_ls_rbf_only),
    )

    gt_pop = None
    if Path(gt_ref).exists():
        g = np.load(gt_ref, allow_pickle=True)
        if all(len(g[f"{k}_axis"]) == len(axes[k]) for k in KEYS):
            gt_pop = {k: g[f"pop_{k}"] for k in KEYS}
            print(f"overlaying training-prior reference from {gt_ref}")
        else:
            print(f"WARNING: --gt-ref grid differs; skipping the reference")

    rows = []
    for preset_name in [p.strip() for p in presets.split(",") if p.strip()]:
        cache = samples_dir / f"{preset_name}{suffix}_samples.npz"
        if not cache.exists():
            print(f"WARNING: {cache} missing; skipping {preset_name}")
            continue
        d = np.load(cache)
        preset = ndp.PROMPT_PRESETS[preset_name]
        x_span = preset.x_range[1] - preset.x_range[0]
        # A 12-month season in normalised x units, for the plot marker.
        period_months = 12.0 / x_span * 4.0
        print(f"\n=== {preset_name}: x_range={preset.x_range} "
              f"y_range={preset.y_range}  12-month season -> "
              f"normalised period {period_months:.3f} ===")

        results = {"raw": {}, "centred": {}}
        for method, key in METHODS:
            y_raw = np.asarray(d[key], dtype=np.float64)     # [N, n_pts]
            x_fit, y_fit = normalise(np.asarray(d["x_grid"], dtype=np.float64),
                                     y_raw, preset.x_range, preset.y_range,
                                     train_y_half)
            for variant in ("raw", "centred"):
                yv = y_fit - y_fit.mean(axis=0, keepdims=True) if variant == "centred" else y_fit
                share, pop, w_per = fit_one(x_fit, yv, cfg, axes, noise_std,
                                            theta_chunk)
                results[variant][method] = (share, pop, w_per)
                means = {k: float(axes[k] @ (pop[k] / pop[k].sum())) for k in KEYS}
                print(f"  {method:7s} [{variant:7s}] N={len(yv)}  "
                      f"periodic share {share:.3f}  "
                      f"period {means['period']:.3f} "
                      f"({means['period'] / 4.0 * x_span:.1f} months)  "
                      f"ls_per {means['ls_per']:.3f}  "
                      f"ls_rbf {means['ls_rbf']:.3f}  "
                      f"ls_rbf_only {means['ls_rbf_only']:.3f}")
                rows.append(dict(
                    preset=preset_name, method=method, variant=variant,
                    n_samples=len(yv), periodic_share=share,
                    period_norm=means["period"],
                    period_months=means["period"] / 4.0 * x_span,
                    ls_per=means["ls_per"], ls_rbf=means["ls_rbf"],
                    ls_rbf_only=means["ls_rbf_only"]))

        plot_preset(out_dir / f"marginals_{preset_name}.png", preset_name,
                    axes, results, gt_pop, period_months, label)

    if rows:
        p = out_dir / "summary.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
