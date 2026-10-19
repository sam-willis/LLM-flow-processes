"""Multi-source finetune: flow-matching NDP on ALL Mauna Loa trace-gas records.

Warm-starts the pretrained ``FlowMatchingTransformer`` and continues training on
~5-year slices drawn from the full NOAA Mauna Loa records for FOUR sources
(CO2 excludes the monthly file -- weekly is used instead):

    source      file                                           cutoff      held-out window
    co2_weekly  data/co2_weekly_mlo.txt                        2019-01-06  2019-01 .. 2023-11
    ch4         data/ch4_mlo_surface-flask_1_ccgg_month.txt    2009-03     2009-03 .. 2014-02
    n2o         data/n2o_mlo_surface-flask_1_ccgg_month.txt    2020-01     2020-01 .. 2024-12
    sf6         data/sf6_mlo_surface-flask_1_ccgg_month.txt    2020-01     2020-01 .. 2024-12

Each source's cutoff is the START of its evaluation window, so the entire
evaluation window (context AND target) is unseen during training. Training
slices are sampled only from the pre-cutoff pool.

To unify heterogeneous units (ppm / ppb / ppt) and sampling rates (weekly vs
monthly), every slice is resampled to ``NUM_POINTS`` points across a ~5-year
window and range-normalised per-slice: x -> [-2, 2], y -> [-1, 1] via a band
``pad`` x the slice's own min-max. Sources are mixed round-robin within each
batch. A random 50-100%% subset of each slice is the observed context.

Example
-------
    uv run python scripts/finetune_mauna_loa_multi.py --n-steps 3000
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import click
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

jax.config.update("jax_compilation_cache_dir", "./.jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

from neural_diffusion_processes.core import XPoints, XYPoints
from neural_diffusion_processes.paths import default_flow_dir, FLOW_DIR_SHOW_DEFAULT
from neural_diffusion_processes.core import FlowMatchingTransformer
from neural_diffusion_processes.core import batch_loss
from neural_diffusion_processes.core import FlowProcess

REPO = Path(__file__).resolve().parent.parent
# The flow checkpoint to finetune. Follows NDP_FLOW_DIR, the same knob the
# rest of the codebase uses, and --base-ckpt overrides it per run. No
# checkpoint ships with the repo: train one first (see the README).
BASE_CKPT = default_flow_dir()
NUM_POINTS = 256

# Per-source config. span_years ~ the inference window (256 weeks / 60 months).
SOURCES = [
    dict(name="co2_weekly", file="data/co2_weekly_mlo.txt", parser="co2_weekly",
         url="https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_weekly_mlo.txt",
         cutoff=2019.0151, span_years=255 * 7 / 365.25, unit="CO2 (ppm)"),
    dict(name="ch4", file="data/ch4_mlo_surface-flask_1_ccgg_month.txt", parser="gas_month",
         url="https://gml.noaa.gov/aftp/data/trace_gases/ch4/flask/surface/txt/ch4_mlo_surface-flask_1_ccgg_month.txt",
         cutoff=2009.0 + 2 / 12, span_years=59 / 12, unit="CH4 (ppb)"),
    dict(name="n2o", file="data/n2o_mlo_surface-flask_1_ccgg_month.txt", parser="gas_month",
         url="https://gml.noaa.gov/aftp/data/trace_gases/n2o/flask/surface/txt/n2o_mlo_surface-flask_1_ccgg_month.txt",
         cutoff=2020.0, span_years=59 / 12, unit="N2O (ppb)"),
    dict(name="sf6", file="data/sf6_mlo_surface-flask_1_ccgg_month.txt", parser="gas_month",
         url="https://gml.noaa.gov/aftp/data/trace_gases/sf6/flask/surface/txt/sf6_mlo_surface-flask_1_ccgg_month.txt",
         cutoff=2020.0, span_years=59 / 12, unit="SF6 (ppt)"),
]

# Normalised x positions (shared): fill [-2, 2] exactly.
_X_N = (np.arange(NUM_POINTS, dtype=np.float32) / (NUM_POINTS - 1)) * 4.0 - 2.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _parse(path: Path, parser: str) -> tuple[np.ndarray, np.ndarray]:
    dec, val = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if parser == "co2_weekly":       # yr mon day decimal co2 ...
            d, v = float(p[3]), float(p[4])
        else:                             # site year month value
            d, v = int(p[1]) + (int(p[2]) - 0.5) / 12.0, float(p[3])
        if v <= -999.0 or v <= 0.0:
            continue
        dec.append(d)
        val.append(v)
    order = np.argsort(dec)
    return np.asarray(dec)[order], np.asarray(val)[order]


def load_sources() -> list[dict]:
    out = []
    for s in SOURCES:
        path = REPO / s["file"]
        if not path.exists():
            print(f"[data] downloading {s['name']} <- {s['url']}", flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(s["url"], timeout=60) as r:
                path.write_bytes(r.read())
        dec, val = _parse(path, s["parser"])
        keep = dec < s["cutoff"]
        pool_dec, pool_val = dec[keep], val[keep]
        lo, hi = float(pool_dec.min()), s["cutoff"] - s["span_years"]
        if hi <= lo:
            raise click.UsageError(f"{s['name']}: pre-cutoff pool shorter than one window.")
        out.append({**s, "dec": dec, "val": val,
                    "pool_dec": pool_dec, "pool_val": pool_val,
                    "start_lo": lo, "start_hi": hi})
        print(f"[data] {s['name']:11s} {len(dec)} pts, pool={len(pool_dec)} "
              f"(< {s['cutoff']:.3f}), window={s['span_years']:.2f}yr", flush=True)
    return out


def make_batch(rng: np.random.Generator, sources: list[dict], batch_size: int,
               pad: float) -> XYPoints:
    xs, ys, masks = [], [], []
    for i in range(batch_size):
        s = sources[i % len(sources)]
        start = rng.uniform(s["start_lo"], s["start_hi"])
        xg = np.linspace(start, start + s["span_years"], NUM_POINTS)
        y = np.interp(xg, s["pool_dec"], s["pool_val"]).astype(np.float32)
        ymid = 0.5 * (y.max() + y.min())
        half = 0.5 * (y.max() - y.min()) * pad
        half = max(half, 1e-6)
        y_n = (y - ymid) / half
        x_n = _X_N.copy()
        perm = rng.permutation(NUM_POINTS)
        x_n, y_n = x_n[perm], y_n[perm]
        n_on = int(rng.integers(NUM_POINTS // 2, NUM_POINTS + 1))
        masks.append(np.arange(NUM_POINTS) >= n_on)
        xs.append(x_n)
        ys.append(y_n)
    return XYPoints(
        x=jnp.asarray(np.stack(xs)[..., None]),
        y=jnp.asarray(np.stack(ys)[..., None]),
        mask=jnp.asarray(np.stack(masks)),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _build(key, cfg):
    return FlowMatchingTransformer(
        hidden_dim=cfg["hidden_dim"], n_layers=cfg["n_layers"], num_heads=cfg["num_heads"],
        input_dim=cfg.get("input_dim", 1), output_dim=cfg.get("output_dim", 1),
        intermediate_size=cfg["intermediate_size"],
        dropout_rate=cfg.get("dropout_rate", 0.0),
        attention_dropout_rate=cfg.get("attention_dropout_rate", 0.0),
        t_embedding_dim=cfg["t_embedding_dim"], t_sin_features=cfg["t_sin_features"],
        t_mlp_hidden_dim=cfg["t_mlp_hidden_dim"], xy_embedding_dim=cfg["xy_embedding_dim"],
        use_rope=cfg.get("use_rope", False), fourier_x_features=cfg.get("fourier_x_features", 0),
        fourier_x_base=cfg.get("fourier_x_base", 2.0), key=key,
    )


def load_flow(ckpt_dir: Path, key):
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    tmpl = _build(key, cfg)
    for name in ("run.eqx", "flow.eqx"):
        if (ckpt_dir / name).exists():
            return eqx.tree_deserialise_leaves(str(ckpt_dir / name), tmpl), cfg
    raise FileNotFoundError(f"no run.eqx/flow.eqx in {ckpt_dir}")


def _ema(smooth, net, decay):
    sP, _ = eqx.partition(smooth, eqx.is_array)
    nP, nS = eqx.partition(net, eqx.is_array)
    return eqx.combine(jax.tree.map(lambda s, n: decay * s + (1 - decay) * n, sP, nP), nS)


# ---------------------------------------------------------------------------
# Held-out NDP forecast (sanity eval, no LLM)
# ---------------------------------------------------------------------------
def forecast_window(net, s: dict, pad: float, n_samples: int, seed: int):
    """Conditional NDP forecast on the held-out post-cutoff window of source s."""
    start = s["cutoff"]
    xg = np.linspace(start, start + s["span_years"], NUM_POINTS)
    truth = np.interp(xg, s["dec"], s["val"])
    ymid = 0.5 * (truth.max() + truth.min())
    half = max(0.5 * (truth.max() - truth.min()) * pad, 1e-6)
    x_range = (float(xg[0]), float(xg[-1]))
    y_range = (ymid - half, ymid + half)
    # 20 context points in the first 55% of the window (rest is forecast).
    ctx_idx = np.linspace(0, int(0.55 * NUM_POINTS) - 1, 20).round().astype(int)
    process = FlowProcess(x_range=x_range, y_range=y_range, gamma_scaling_ndp=50.0)
    context = XYPoints(
        x=jnp.asarray(xg[ctx_idx, None], dtype=jnp.float32),
        y=jnp.asarray(truth[ctx_idx, None], dtype=jnp.float32),
        mask=jnp.zeros(len(ctx_idx), dtype=jnp.bool_))
    target = XPoints(x=jnp.asarray(xg[:, None], dtype=jnp.float32),
                     mask=jnp.zeros(NUM_POINTS, dtype=jnp.bool_))
    pred, _, _ = process.sample_n_conditionals(
        net, target, context, key=jax.random.PRNGKey(seed), n_samples=n_samples,
        likelihood_noise_std=0.001)
    return xg, truth, xg[ctx_idx], truth[ctx_idx], np.asarray(pred.y[..., 0])


def eval_plot(base, ft, sources, pad, n_samples, seed, out_png):
    fig, axes = plt.subplots(len(sources), 2, figsize=(13, 3.2 * len(sources)))
    rmse = {}
    fcut = int(0.55 * NUM_POINTS)  # forecast region starts here
    for r, s in enumerate(sources):
        for c, (tag, net) in enumerate((("base", base), ("finetuned", ft))):
            xg, truth, cx, cy, samp = forecast_window(net, s, pad, n_samples, seed + c)
            mean = samp.mean(0)
            lo, hi = np.percentile(samp, 5, 0), np.percentile(samp, 95, 0)
            e = float(np.sqrt(np.mean((mean[fcut:] - truth[fcut:]) ** 2)))
            rmse[(s["name"], tag)] = e
            ax = axes[r, c]
            ax.fill_between(xg, lo, hi, alpha=0.22, color="C0")
            ax.plot(xg, mean, color="C0", lw=1.4)
            ax.plot(xg, truth, color="k", lw=1.0)
            ax.scatter(cx, cy, color="C3", s=14, zorder=5)
            ax.axvline(xg[fcut], color="grey", ls=":", lw=0.8)
            ax.set_title(f"{s['name']} - {tag}  (fcst RMSE {e:.3g} {s['unit'].split()[-1]})",
                         fontsize=10)
            if c == 0:
                ax.set_ylabel(s["unit"])
    axes[-1, 0].set_xlabel("year"); axes[-1, 1].set_xlabel("year")
    fig.suptitle("Held-out (post-cutoff) NDP forecast per source: base vs multi-source finetune",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return rmse


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--n-steps", type=int, default=3000, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--ema-decay", type=float, default=0.999, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--pad", type=float, default=1.4, show_default=True,
              help="Per-slice y-band = pad x (max-min); >1 leaves headroom like the presets.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--n-eval-samples", type=int, default=50, show_default=True)
@click.option("--out-dir", type=click.Path(path_type=Path),
              default=REPO / "outputs" / "finetune_mauna_loa_multi", show_default=True)
@click.option("--base-ckpt", type=click.Path(path_type=Path, exists=True), default=BASE_CKPT)
@click.option("--log-every", type=int, default=200, show_default=True)
@click.option("--skip-eval", is_flag=True, default=False)
def main(n_steps, batch_size, lr, ema_decay, grad_clip, pad, seed, n_eval_samples,
         out_dir, base_ckpt, log_every, skip_eval):
    print(f"jax devices: {jax.devices()}", flush=True)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base_ckpt = Path(base_ckpt)

    sources = load_sources()
    key = jax.random.PRNGKey(seed)
    k_build, k_train = jax.random.split(key)
    net, cfg = load_flow(base_ckpt, k_build)
    smooth = net
    print(f"[model] warm-started from {base_ckpt}/run.eqx", flush=True)

    optim = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adamw(lr, eps=1e-6))
    opt_state = optim.init(eqx.filter(net, eqx.is_array))

    @eqx.filter_jit
    def step(net, smooth, opt_state, pts, k):
        params = eqx.filter(net, eqx.is_array)
        loss, grads = eqx.filter_value_and_grad(batch_loss)(net, pts, k)
        updates, opt_state = optim.update(grads, opt_state, params)
        net = eqx.apply_updates(net, updates)
        smooth = _ema(smooth, net, ema_decay)
        return net, smooth, opt_state, loss

    rng = np.random.default_rng(seed)
    losses = np.empty(n_steps, dtype=np.float32)
    pbar = tqdm(range(n_steps), desc="finetune-multi")
    for i in pbar:
        pts = make_batch(rng, sources, batch_size, pad)
        k_train, sk = jax.random.split(k_train)
        net, smooth, opt_state, loss = step(net, smooth, opt_state, pts, sk)
        losses[i] = float(loss)
        if i % log_every == 0 or i == n_steps - 1:
            pbar.set_postfix(loss=f"{losses[i]:.4f}",
                             ema50=f"{losses[max(0,i-49):i+1].mean():.4f}")

    (out_dir / "config.json").write_text(json.dumps(
        {**cfg, "finetuned_from": str(base_ckpt),
         "finetune_multi": {"sources": [s["name"] for s in SOURCES],
                            "cutoffs": {s["name"]: s["cutoff"] for s in SOURCES},
                            "n_steps": n_steps, "batch_size": batch_size, "lr": lr,
                            "pad": pad, "seed": seed}}, indent=2, sort_keys=True))
    eqx.tree_serialise_leaves(str(out_dir / "run.eqx"), smooth)
    np.save(out_dir / "loss_history.npy", losses)
    print(f"[save] {out_dir}/run.eqx  (final ema50 {losses[-50:].mean():.4f})", flush=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, alpha=0.3, lw=0.6)
    if n_steps >= 50:
        ax.plot(np.arange(49, n_steps), np.convolve(losses, np.ones(50) / 50, "valid"), lw=1.5)
    ax.set_xlabel("step"); ax.set_ylabel("flow-matching loss")
    ax.set_title("Multi-source Mauna Loa finetune loss")
    fig.tight_layout(); fig.savefig(out_dir / "loss.png", dpi=140); plt.close(fig)

    if skip_eval:
        return
    print("[eval] held-out NDP forecasts per source (base vs finetuned)...", flush=True)
    base_net, _ = load_flow(base_ckpt, k_build)
    rmse = eval_plot(base_net, smooth, sources, pad, n_eval_samples, seed + 1,
                     out_dir / "eval_forecast_per_source.png")
    print(f"\n[eval] held-out forecast RMSE (native units):")
    print(f"{'source':>12} {'base':>10} {'finetuned':>10}")
    for s in sources:
        print(f"{s['name']:>12} {rmse[(s['name'],'base')]:>10.4g} "
              f"{rmse[(s['name'],'finetuned')]:>10.4g}")
    print(f"[eval] wrote {out_dir/'eval_forecast_per_source.png'}", flush=True)


if __name__ == "__main__":
    main()
