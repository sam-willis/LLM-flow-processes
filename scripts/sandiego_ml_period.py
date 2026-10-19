"""Maximum-likelihood period per San Diego sample, from the GP evidence grid.

`fit_sandiego_hparams.py` reports the *population marginal* period: the
posterior over the period axis, averaged across samples and weighted by each
sample's periodic-branch probability. That is a distributional summary and can
sit at a value no individual sample prefers.

This reports the per-sample argmax instead -- for each sample, the (period,
ls_per, ls_rbf) triple maximising the exact GP log marginal likelihood -- and
then two summaries of that population:

  median ML period (months)
  fraction whose ML period is within +/-tol of a 12-month season

It is the GP-side counterpart of the periodogram numbers in
`sandiego_period_accuracy.py`: same question, different model class, so the two
disagreeing is informative rather than a bug.

Grids, coordinates, noise and the periodic kernel are taken from
fit_sandiego_hparams.py so the likelihood surface is the identical one its
share is computed from.

Reads `<preset><suffix>_samples.npz` from --samples-dir ($NDP_OUTPUTS_DIR,
default `outputs/`) and writes one CSV row per (preset, method) to --out.

  uv run --no-sync python scripts/sandiego_ml_period.py \\
      --suffix _ncorr0 --out outputs/sandiego_hparams/ml_period.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import click
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from neural_diffusion_processes import pipeline as ndp  # noqa: E402
from neural_diffusion_processes.paths import default_flow_dir, FLOW_DIR_SHOW_DEFAULT
from fit_mixture_hparams_bayes import (  # noqa: E402
    branch_posterior, log_evidence, log_lik_periodic_grid,
    make_evaluators, midpoints)
from fit_sandiego_hparams import METHODS, normalise  # noqa: E402


def _default_samples_dir() -> Path:
    """`run.OUTPUTS_DIR`, imported only when the default is actually used."""
    from neural_diffusion_processes.run import OUTPUTS_DIR
    return OUTPUTS_DIR


@click.command()
@click.option("--samples-dir", default=_default_samples_dir,
              show_default="$NDP_OUTPUTS_DIR, else outputs/",
              help="Directory holding the <preset><suffix>_samples.npz caches.")
@click.option("--suffix", default="", show_default=True)
@click.option("--presets", default="sandiego,sandiego_long", show_default=True)
@click.option("--run-dir", default=default_flow_dir,
              show_default=FLOW_DIR_SHOW_DEFAULT)
@click.option("--variant", default="raw", type=click.Choice(["raw", "centred"]),
              show_default=True)
@click.option("--n-period", default=128, type=int, show_default=True)
@click.option("--n-ls-per", default=32, type=int, show_default=True)
@click.option("--n-ls-rbf", default=32, type=int, show_default=True)
@click.option("--n-ls-rbf-only", default=128, type=int, show_default=True)
@click.option("--noise-std", default=0.05, type=float, show_default=True)
@click.option("--tol", default=0.10, type=float, show_default=True,
              help="Fractional tolerance around 12 months.")
@click.option("--theta-chunk", default=64, type=int, show_default=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(samples_dir, suffix, presets, run_dir, variant, n_period, n_ls_per,
         n_ls_rbf, n_ls_rbf_only, noise_std, tol, theta_chunk, out):
    samples_dir = Path(samples_dir)
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    train_y_half = float(cfg["y_range"][1])
    variance = float(cfg["variance"])
    per_axis = midpoints(*cfg["period_range"], n_period)
    lp_axis = midpoints(*cfg["ls_per_range"], n_ls_per)
    lr_axis = midpoints(*cfg["ls_rbf_range"], n_ls_rbf)
    lro_axis = midpoints(*cfg["ls_rbf_only_range"], n_ls_rbf_only)
    p_rbf_only = float(cfg["rbf_only_prob"])

    rows = []
    for preset_name in [p.strip() for p in presets.split(",") if p.strip()]:
        cache = samples_dir / f"{preset_name}{suffix}_samples.npz"
        if not cache.exists():
            print(f"WARNING: {cache} missing; skipping")
            continue
        d = np.load(cache)
        preset = ndp.PROMPT_PRESETS[preset_name]
        x_span = preset.x_range[1] - preset.x_range[0]
        # normalised period <-> months: x is squashed onto [-2, 2], i.e. a
        # factor 4 / x_span, so a 12-month season is period 12/x_span*4.
        to_months = x_span / 4.0

        for method, key in METHODS:
            y_raw = np.asarray(d[key], dtype=np.float64)
            x_fit, y_fit = normalise(np.asarray(d["x_grid"], dtype=np.float64),
                                     y_raw, preset.x_range, preset.y_range,
                                     train_y_half)
            yv = (y_fit - y_fit.mean(axis=0, keepdims=True)
                  if variant == "centred" else y_fit)
            Y = jnp.asarray(yv.T, dtype=jnp.float32)          # [n_pts, N]
            eval_per, eval_rbf = make_evaluators(x_fit, variance, noise_std ** 2)

            # Reuse the shared grid helper rather than re-implementing the
            # chunked evaluation: it returns [n_period*n_ls_per*n_ls_rbf, N]
            # built with indexing="ij", so period is the OUTERMOST axis and the
            # flat argmax maps back with // (n_ls_per * n_ls_rbf).
            log_p = log_lik_periodic_grid(
                eval_per, (per_axis, lp_axis, lr_axis), Y, theta_chunk,
                verbose=False)
            flat = log_p.argmax(axis=0)                       # [N]
            best_per = per_axis[flat // (len(lp_axis) * len(lr_axis))]
            months = best_per * to_months

            # Same likelihood surface, so "periodic" and "right period" are two
            # readings of one fit rather than two different estimators.
            log_p_rbf = np.asarray(eval_rbf(jnp.asarray(lro_axis), Y)[0])
            w_per, _ = branch_posterior(log_evidence(log_p),
                                        log_evidence(log_p_rbf), p_rbf_only)
            is_per = w_per > 0.5
            in_band = np.abs(months - 12.0) <= tol * 12.0
            rows.append({
                "preset": preset_name, "method": method, "variant": variant,
                "n_samples": int(len(months)),
                "frac_periodic": float(is_per.mean()),
                "frac_in_band": float(in_band.mean()),
                "frac_both": float((is_per & in_band).mean()),
                "ml_period_median_months": float(np.median(months)),
                "ml_period_median_months_periodic": (
                    float(np.median(months[is_per])) if is_per.any() else float("nan")),
            })
            r = rows[-1]
            print(f"  {preset_name:14s} {method:7s} N={len(months)}  "
                  f"periodic {r['frac_periodic']:.3f}  in-band {r['frac_in_band']:.3f}  "
                  f"both {r['frac_both']:.3f}  median ML period "
                  f"{r['ml_period_median_months']:6.2f} mo", flush=True)

    if not rows:
        raise click.ClickException(
            f"no <preset>{suffix}_samples.npz caches found in {samples_dir}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
