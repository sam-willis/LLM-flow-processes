"""How many sandiego samples are periodic *at the right frequency*?

`fit_sandiego_hparams.py` reports two numbers per method: the share of samples
the evidence assigns to the periodic branch, and the population-mean period.
Neither answers "what fraction of individual samples are both periodic and
close to a 12-month season" — a population mean near 12.5 months is equally
consistent with every sample sitting at 12.5 and with half the samples at 10
and half at 15.

Two independent estimators are reported, because each is wrong in a different
way:

  bayes     Reuses the evidence fit. Per sample it has P(periodic | y) and a
            posterior over the period, so the joint quantity is

                P(periodic and |f/f0 - 1| <= tol | y)
                    = P(periodic | y) * P(period in band | y, periodic)

            summed over samples. Caveat: the generator's period prior is
            [0.75, 1.25] normalised = [9.4, 15.6] months, and the fit cannot
            look outside it. A sample with a 20-month cycle is squeezed to the
            edge rather than rejected, so this estimator is optimistic. The
            band itself covers 39% of that prior, which is the number to
            compare against — not zero.

  spectral  Model-free and unconstrained. Per sample, fit
            ``y = a + b*x + A cos(2 pi f x) + B sin(2 pi f x)`` by least
            squares on a dense grid of f, take the maximising f as the sample's
            frequency and the variance it explains (over and above the linear
            trend) as R^2. A sample counts when R^2 exceeds a threshold *and*
            f lands in the band. Frequency resolution is not grid-limited, but
            the 50-month window holds only ~4 cycles, so 10% discrimination is
            near the information limit of the data; the R^2 sweep shows how
            much the answer leans on the threshold.

"Correct frequency" is a 12-month season and the tolerance is on *frequency*:
f in [0.9, 1.1] / 12 per month, i.e. a period in [10.91, 13.33] months. The
period-tolerance reading ([10.8, 13.2]) is printed alongside since the two
differ by less than the grid.

Outputs (under --out-dir):
  - period_accuracy.csv  : one row per (preset, method, variant, estimator).
  - period_hist.png      : per-sample period/frequency histograms with the band.

Example
-------
    uv run python scripts/sandiego_period_accuracy.py \
        --out-dir outputs/sandiego_ncorr0_period_accuracy
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TARGET_MONTHS = 12.0
METHODS = (("NDP", "y_ndp"), ("LLM-FP", "y_ndp_llm"), ("iLLMP", "y_illmp"))
COLOURS = {"NDP": "#C44E52", "LLM-FP": "#4C72B0", "iLLMP": "#55A868",
           "LLM-direct": "#8172B2"}


def _default_samples_dir() -> Path:
    """`run.OUTPUTS_DIR`, imported only when the default is actually used."""
    from neural_diffusion_processes.run import OUTPUTS_DIR
    return OUTPUTS_DIR


# ---------------------------------------------------------------------------
# Band
# ---------------------------------------------------------------------------


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion.

    Reported because the sample counts are not comparable: the npz caches hold
    `run.DEFAULTS["n_samples"]` draws per method but LLM-direct is typically
    run with far fewer, so a bare percentage invites reading a difference that
    the smaller set cannot support.
    """
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1.0 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (c - h) / d, (c + h) / d


def freq_band(target: float, tol: float) -> tuple[float, float]:
    """Period band, in the same units as `target`, for a frequency tolerance.

    |f/f0 - 1| <= tol  <=>  T in [T0/(1+tol), T0/(1-tol)]. Asymmetric in period
    by construction — that asymmetry is the point of quoting the tolerance on
    frequency rather than on period.
    """
    return target / (1.0 + tol), target / (1.0 - tol)


# ---------------------------------------------------------------------------
# Estimator 1: model-free sinusoid fit over a dense frequency grid
# ---------------------------------------------------------------------------


def sinusoid_scan(x: np.ndarray, Y: np.ndarray, freqs: np.ndarray,
                  chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Best-fitting frequency over `freqs` and its R^2, per sample.

    Y is [N, n_pts]. For every f the design is ``[1, x, cos, sin]``; the
    constant and linear columns are always present so a trend cannot masquerade
    as a very low frequency. R^2 is measured against the *detrended* signal,
    so it reads as "fraction of the non-trend variance this sinusoid explains".

    Solved by projecting onto an orthonormalised design rather than lstsq per
    (f, sample): the trend basis is shared, so detrend once and then only the
    2-column sinusoid block varies with f. Called twice — once over the whole
    scan range, once restricted to the ±tol band — which is what separates
    "the dominant cycle is seasonal" from "a seasonal cycle is present".
    """
    n_pts = x.shape[0]
    # Orthonormal basis for the trend subspace, shared across all f.
    T = np.stack([np.ones(n_pts), (x - x.mean()) / np.ptp(x)], axis=1)
    Qt, _ = np.linalg.qr(T)                                  # [n_pts, 2]
    Yd = Y - (Y @ Qt) @ Qt.T                                 # detrended [N, n]
    ss_tot = np.einsum("ij,ij->i", Yd, Yd)                   # [N]
    ss_tot = np.maximum(ss_tot, 1e-30)

    best_r2 = np.zeros(Y.shape[0])
    best_f = np.full(Y.shape[0], np.nan)
    for lo in range(0, len(freqs), chunk):
        f = freqs[lo:lo + chunk][:, None]                    # [c, 1]
        ang = 2.0 * np.pi * f * x[None, :]                   # [c, n_pts]
        B = np.stack([np.cos(ang), np.sin(ang)], axis=2)     # [c, n_pts, 2]
        # Remove the trend from the sinusoid columns too, so the projection is
        # onto the part of the sinusoid orthogonal to the trend.
        B = B - np.einsum("ij,cjk->cik", Qt, np.einsum("ij,cjk->cik", Qt.T, B))
        G = np.einsum("cij,cik->cjk", B, B)                  # [c, 2, 2]
        C = np.einsum("cij,nj->cni", B.transpose(0, 2, 1), Yd)  # [c, N, 2]
        # 2x2 solve in closed form; G is symmetric positive semi-definite.
        det = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
        det = np.where(np.abs(det) < 1e-12, np.nan, det)
        a = (G[:, 1, 1, None] * C[:, :, 0] - G[:, 0, 1, None] * C[:, :, 1]) / det[:, None]
        b = (-G[:, 1, 0, None] * C[:, :, 0] + G[:, 0, 0, None] * C[:, :, 1]) / det[:, None]
        ss_exp = a * C[:, :, 0] + b * C[:, :, 1]             # [c, N]
        r2 = np.nan_to_num(ss_exp / ss_tot[None, :], nan=0.0)
        j = r2.argmax(axis=0)
        take = r2[j, np.arange(Y.shape[0])] > best_r2
        best_r2 = np.where(take, r2[j, np.arange(Y.shape[0])], best_r2)
        best_f = np.where(take, freqs[lo:lo + chunk][j], best_f)
    return best_f, best_r2


# ---------------------------------------------------------------------------
# Estimator 2: the evidence fit's own posterior
# ---------------------------------------------------------------------------


def normalise(x: np.ndarray, y: np.ndarray, x_range, y_range,
              train_y_half: float):
    """Demo units -> training coordinates (identical to fit_sandiego_hparams)."""
    x0, x1 = x_range
    y0, y1 = y_range
    x_fit = ((x - x0) / (x1 - x0) - 0.5) * 4.0
    y_fit = ((y - y0) / (y1 - y0) - 0.5) * 2.0 * train_y_half
    return x_fit.astype(np.float64), y_fit.astype(np.float64)


def bayes_period_posterior(x_fit, y_fit, cfg, axes, noise_std, theta_chunk):
    """Per-sample (P(periodic), period marginal [n_knots, N])."""
    import jax.numpy as jnp
    from fit_mixture_hparams_bayes import (
        branch_posterior, log_evidence, log_lik_periodic_grid, make_evaluators,
        marginalise, normalised_weights,
    )

    variance = float(cfg["variance"])
    p_rbf_only = float(cfg["rbf_only_prob"])
    Y = jnp.asarray(y_fit.T, dtype=jnp.float32)
    eval_per, eval_rbf = make_evaluators(x_fit, variance, float(noise_std) ** 2)
    log_p_per = log_lik_periodic_grid(
        eval_per, (axes["period"], axes["ls_per"], axes["ls_rbf"]),
        Y, theta_chunk, verbose=False)
    log_p_rbf = np.asarray(eval_rbf(jnp.asarray(axes["ls_rbf_only"]), Y)[0])
    w_per, _ = branch_posterior(log_evidence(log_p_per),
                                log_evidence(log_p_rbf), p_rbf_only)
    m_period, _, _ = marginalise(
        normalised_weights(log_p_per),
        (len(axes["period"]), len(axes["ls_per"]), len(axes["ls_rbf"])))
    return w_per, m_period


def midpoints(lo: float, hi: float, n: int) -> np.ndarray:
    edges = np.linspace(lo, hi, n + 1)
    return (0.5 * (edges[:-1] + edges[1:])).astype(np.float32)


def band_mass(knots: np.ndarray, w: np.ndarray, lo: float, hi: float):
    """Posterior mass in [lo, hi], counting partial cells at the edges.

    The knots are cell midpoints of a uniform grid; a knot's cell contributes
    the fraction of its width that overlaps the band. At 128 knots over
    [0.75, 1.25] the correction is under one cell, but it removes the
    grid-alignment jitter that would otherwise show up when comparing methods.
    """
    width = float(knots[1] - knots[0])
    left = knots.astype(np.float64) - width / 2.0
    right = left + width
    frac = np.clip((np.minimum(right, hi) - np.maximum(left, lo)) / width, 0, 1)
    return frac @ w                                          # [N]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_hist(out_path: Path, panels, lo_m, hi_m, r2_min):
    # Explicit rc_context so the font sizes suit a grid this wide regardless
    # of any matplotlib style the caller has active.
    rc = {"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
          "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
          "figure.titlesize": 12}
    with plt.rc_context(rc):
        n = len(panels)
        fig, axs = plt.subplots(1, n, figsize=(6.0 * n, 4.2), squeeze=False)
        for ax, (title, per_method) in zip(axs[0], panels):
            for label, (period_months, r2) in per_method.items():
                keep = r2 >= r2_min
                if not keep.any():
                    ax.plot([], [], color=COLOURS[label], lw=1.8,
                            label=f"{label} (R²≥{r2_min}: 0.00, none shown)")
                    continue
                ax.hist(np.clip(period_months[keep], 4, 40),
                        bins=np.arange(4, 40.5, 0.5), histtype="step", lw=1.8,
                        color=COLOURS[label], density=True,
                        label=f"{label} (R²≥{r2_min}: {keep.mean():.2f})")
            ax.axvspan(lo_m, hi_m, color="#55A868", alpha=0.18, zorder=0)
            ax.axvline(TARGET_MONTHS, color="black", ls="--", lw=1.4)
            ax.set_title(title)
            ax.set_xlabel("best-fit period (months, clipped to [4, 40];\n"
                          "the spike at 40 is the low-frequency scan edge)")
            ax.grid(True, alpha=0.2)
            ax.legend()
        axs[0][0].set_ylabel("density")
        fig.suptitle("Model-free dominant period per sample; shaded = ±10% in "
                     "frequency around a 12-month season")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out_path, dpi=140)
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@click.command()
@click.option("--samples-dir", default=_default_samples_dir,
              show_default="$NDP_OUTPUTS_DIR, else outputs/",
              type=click.Path(file_okay=False, path_type=Path),
              help="Directory holding the <preset><suffix>_samples.npz caches "
                   "and the <preset>_direct.pkl LLM-direct pickles.")
@click.option("--suffix", default="_ncorr0", show_default=True,
              help="Sample-cache suffix, e.g. the --cache-suffix given to "
                   "sample_for_hparams.py. Empty string reads "
                   "<preset>_samples.npz.")
@click.option("--presets", default="sandiego,sandiego_long", show_default=True)
@click.option("--run-dir", default=default_flow_dir,
              show_default=FLOW_DIR_SHOW_DEFAULT,
              type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True,
              type=click.Path(file_okay=False, path_type=Path))
@click.option("--tol", default=0.10, show_default=True,
              help="Fractional tolerance on frequency.")
@click.option("--r2-min", default=0.5, show_default=True,
              help="Headline R^2 threshold for 'periodic' in the spectral "
                   "estimator. The CSV carries a sweep either way.")
@click.option("--r2-sweep", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8",
              show_default=True)
@click.option("--f-min", default=1.0 / 40.0, show_default=True,
              help="Lowest frequency scanned, cycles per month.")
@click.option("--f-max", default=1.0 / 4.0, show_default=True)
@click.option("--n-freq", default=4000, type=int, show_default=True)
@click.option("--skip-bayes", is_flag=True,
              help="Spectral estimator only (seconds instead of minutes).")
@click.option("--include-direct/--no-include-direct", default=True,
              show_default=True,
              help="Also read <preset>_direct.pkl (direct LLM sampling). It is "
                   "a separate artifact on its own integer-month grid, with "
                   "no _ncorr0 variant, since it never runs the corrector.")
@click.option("--n-period", default=128, type=int, show_default=True)
@click.option("--n-ls-per", default=32, type=int, show_default=True)
@click.option("--n-ls-rbf", default=32, type=int, show_default=True)
@click.option("--n-ls-rbf-only", default=128, type=int, show_default=True)
@click.option("--noise-std", default=0.05, type=float, show_default=True)
@click.option("--theta-chunk", default=64, type=int, show_default=True)
def main(samples_dir, suffix, presets, run_dir, out_dir, tol, r2_min, r2_sweep,
         f_min, f_max, n_freq, skip_bayes, include_direct, n_period, n_ls_per,
         n_ls_rbf, n_ls_rbf_only, noise_std, theta_chunk):
    from neural_diffusion_processes import pipeline as ndp

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run_dir / "config.json").read_text())
    train_y_half = float(cfg["y_range"][1])
    axes = dict(
        period=midpoints(*cfg["period_range"], n_period),
        ls_per=midpoints(*cfg["ls_per_range"], n_ls_per),
        ls_rbf=midpoints(*cfg["ls_rbf_range"], n_ls_rbf),
        ls_rbf_only=midpoints(*cfg["ls_rbf_only_range"], n_ls_rbf_only),
    )
    thresholds = [float(t) for t in r2_sweep.split(",") if t.strip()]
    freqs = np.linspace(f_min, f_max, n_freq)

    lo_m, hi_m = freq_band(TARGET_MONTHS, tol)
    print(f"target {TARGET_MONTHS:.0f} months, frequency tolerance ±{tol:.0%}")
    print(f"  -> period band [{lo_m:.3f}, {hi_m:.3f}] months "
          f"(period-tolerance reading would be "
          f"[{TARGET_MONTHS * (1 - tol):.3f}, {TARGET_MONTHS * (1 + tol):.3f}])")

    rows, panels = [], []
    for preset_name in [p.strip() for p in presets.split(",") if p.strip()]:
        cache = samples_dir / f"{preset_name}{suffix}_samples.npz"
        if not cache.exists():
            print(f"WARNING: {cache} missing; skipping {preset_name}")
            continue
        d = np.load(cache)
        preset = ndp.PROMPT_PRESETS[preset_name]
        x_span = preset.x_range[1] - preset.x_range[0]
        x = np.asarray(d["x_grid"], dtype=np.float64)
        band_freqs = np.linspace((1.0 - tol) / TARGET_MONTHS,
                                 (1.0 + tol) / TARGET_MONTHS, 200)
        # Same band, expressed in the fit's normalised period units.
        to_norm = 4.0 / x_span
        lo_n, hi_n = lo_m * to_norm, hi_m * to_norm
        p_lo, p_hi = cfg["period_range"]
        prior_share = (min(hi_n, p_hi) - max(lo_n, p_lo)) / (p_hi - p_lo)
        print(f"\n=== {preset_name} (x span {x_span:.0f} months) ===")
        print(f"  band in normalised period units: [{lo_n:.4f}, {hi_n:.4f}]; "
              f"prior [{p_lo}, {p_hi}] -> band covers {prior_share:.1%} of the "
              f"prior (chance level for the bayes estimator)")

        series = [(label, x, np.asarray(d[key], dtype=np.float64))
                  for label, key in METHODS if key in d.files]

        # LLM-direct lives in its own pickle on its own grid (integer months,
        # 51 points) and has no _ncorr0 variant — direct sampling never runs
        # the flow corrector, so the suffix that defines this comparison does
        # not apply to it. Same 50-month window, so the frequency resolution is
        # the same; only N differs, which the CSV carries per row.
        if include_direct:
            pkl = samples_dir / f"{preset_name}_direct.pkl"
            if pkl.exists():
                import pickle
                dd = pickle.loads(pkl.read_bytes())
                series.append(("LLM-direct",
                               np.asarray(dd["x_test"], dtype=np.float64),
                               np.asarray(dd["samples"], dtype=np.float64)))
                print(f"  + LLM-direct from {pkl.name}: "
                      f"{np.asarray(dd['samples']).shape} on "
                      f"{len(dd['x_test'])} grid points")
            else:
                print(f"  WARNING: {pkl} missing; skipping LLM-direct")

        per_method = {}
        for label, x_m, y_raw in series:
            n = len(y_raw)

            # -- spectral -------------------------------------------------
            f_hat, r2 = sinusoid_scan(x_m, y_raw, freqs)
            # Best sinusoid *within* the band. A sample whose largest component
            # is a slow drift can still carry a real 12-month cycle underneath;
            # the global argmax cannot see it, this can.
            _, r2_band = sinusoid_scan(x_m, y_raw, band_freqs)
            period_months = 1.0 / f_hat
            in_band = (f_hat >= (1.0 - tol) / TARGET_MONTHS) & \
                      (f_hat <= (1.0 + tol) / TARGET_MONTHS)
            per_method[label] = (period_months, r2)
            at_edge = float(((f_hat <= freqs[1]) | (f_hat >= freqs[-2])).mean())
            print(f"  {label:7s} spectral  N={n}  median R² {np.median(r2):.3f}"
                  f"  median in-band R² {np.median(r2_band):.3f}"
                  f"  median period {np.median(period_months):.2f} mo"
                  f"  f at scan edge {at_edge:.3f}")
            for t in thresholds:
                for crit, ok in (("dominant", (r2 >= t) & in_band),
                                 ("present", r2_band >= t)):
                    ci_lo, ci_hi = wilson(int(ok.sum()), n)
                    rows.append(dict(
                        preset=preset_name, method=label, variant="raw",
                        estimator="spectral",
                        criterion=f"{crit} R2>={t:g}",
                        n_samples=n, frac_periodic=float((r2 >= t).mean()),
                        frac_in_band=float(in_band.mean()),
                        frac_both=float(ok.mean()), n_both=int(ok.sum()),
                        ci95_lo=ci_lo, ci95_hi=ci_hi,
                        frac_in_band_given_periodic=(
                            float(in_band[r2 >= t].mean()) if (r2 >= t).any()
                            else float("nan")),
                        median_period_months=float(np.median(period_months)),
                        chance_level=float("nan")))
                dom = (r2 >= t) & in_band
                pres = r2_band >= t
                print(f"    R²≥{t:<4g} periodic {(r2 >= t).mean():.3f}  "
                      f"dominant-and-in-band {dom.mean():.3f} "
                      f"({dom.sum()}/{n})  "
                      f"seasonal-present {pres.mean():.3f} ({pres.sum()}/{n})")

            # -- bayes ----------------------------------------------------
            if skip_bayes:
                continue
            x_fit, y_fit = normalise(x_m, y_raw, preset.x_range,
                                     preset.y_range, train_y_half)
            for variant in ("raw", "centred"):
                yv = (y_fit - y_fit.mean(axis=0, keepdims=True)
                      if variant == "centred" else y_fit)
                w_per, m_period = bayes_period_posterior(
                    x_fit, yv, cfg, axes, noise_std, theta_chunk)
                mass = band_mass(axes["period"], m_period, lo_n, hi_n)
                joint = w_per * mass
                hard_per = w_per > 0.5
                mean_period = axes["period"].astype(np.float64) @ m_period
                hard_both = hard_per & (mean_period >= lo_n) & (mean_period <= hi_n)
                rows.append(dict(
                    preset=preset_name, method=label, variant=variant,
                    estimator="bayes", criterion="posterior mass",
                    n_samples=n, frac_periodic=float(w_per.mean()),
                    frac_in_band=float(mass.mean()),
                    frac_both=float(joint.mean()),
                    n_both=float(joint.sum()),
                    # An expected count, not a Bernoulli one — a Wilson
                    # interval would misrepresent it, so it is left blank.
                    ci95_lo=float("nan"), ci95_hi=float("nan"),
                    frac_in_band_given_periodic=float(
                        (w_per * mass).sum() / max(w_per.sum(), 1e-12)),
                    median_period_months=float(
                        np.median(mean_period) / to_norm),
                    chance_level=float(prior_share)))
                rows.append(dict(
                    preset=preset_name, method=label, variant=variant,
                    estimator="bayes", criterion="hard P>0.5 & mean in band",
                    n_samples=n, frac_periodic=float(hard_per.mean()),
                    frac_in_band=float(((mean_period >= lo_n)
                                        & (mean_period <= hi_n)).mean()),
                    frac_both=float(hard_both.mean()),
                    n_both=int(hard_both.sum()),
                    **dict(zip(("ci95_lo", "ci95_hi"),
                               wilson(int(hard_both.sum()), n))),
                    frac_in_band_given_periodic=(
                        float(hard_both.sum() / hard_per.sum())
                        if hard_per.any() else float("nan")),
                    median_period_months=float(np.median(mean_period) / to_norm),
                    chance_level=float(prior_share)))
                print(f"  {label:7s} bayes [{variant:7s}]  "
                      f"E[periodic] {w_per.mean():.3f}  "
                      f"E[in band] {mass.mean():.3f}  "
                      f"E[both] {joint.mean():.3f} ({joint.sum():.1f}/{n})  |  "
                      f"hard both {hard_both.mean():.3f} "
                      f"({hard_both.sum()}/{n})")

        panels.append((f"{preset_name} [{suffix.lstrip('_') or 'original'}]",
                       per_method))

    if panels:
        plot_hist(out_dir / "period_hist.png", panels, lo_m, hi_m, r2_min)
    if rows:
        p = out_dir / "period_accuracy.csv"
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
