"""Flow-sample a list of presets into `_samples.npz` caches, nothing else.

For each preset, reads the LLM score pickle
`$NDP_OUTPUTS_DIR/<preset><pkl-suffix>.pkl` and writes the flow-sample cache
`$NDP_OUTPUTS_DIR/<preset><cache-suffix>_samples.npz` (NDP_OUTPUTS_DIR defaults
to `outputs/`) via `run.cached_sample_for_preset`. The hyperparameter analysis
(`scripts/fit_sandiego_hparams.py`) reads nothing but `y_ndp` / `y_ndp_llm` out
of the npz, so this is just the sampling call -- which is what a
prompt-ablation arm needs, since a new prompt gets a new score pickle but no
Direct run.

Sampler settings come from `neural_diffusion_processes.run.DEFAULTS`, so a cache
written here is interchangeable with one written by any other
`run.cached_sample_for_preset` caller. `NDP_N_CORR` (read by `run` at import)
selects the corrector count; pass `--cache-suffix` to keep caches from
different settings apart, exactly as `run` warns. The suffixes are free-form
tags: `--pkl-suffix` must match the name the score pickle was saved under, and
`--cache-suffix` is whatever name `fit_sandiego_hparams.py --suffix` will read.

Example
-------
    NDP_N_CORR=0 uv run python scripts/sample_for_hparams.py \
        --presets sandiego,sandiego_long \
        --cache-suffix _ncorr0
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pathlib import Path

import click

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@click.command()
@click.option("--presets", required=True,
              help="Comma-separated preset names, e.g. 'sandiego,sandiego_long'.")
@click.option("--pkl-suffix", default="", show_default=True,
              help="Score pickle is $NDP_OUTPUTS_DIR (default outputs/)"
                   "/<preset><pkl-suffix>.pkl.")
@click.option("--cache-suffix", default="", show_default=True,
              help="Cache is $NDP_OUTPUTS_DIR (default outputs/)"
                   "/<preset><cache-suffix>_samples.npz.")
@click.option("--resample", is_flag=True, default=False,
              help="Re-sample even when the cache exists.")
def main(presets: str, pkl_suffix: str, cache_suffix: str, resample: bool):
    from neural_diffusion_processes.run import (  # noqa: E402
        DEFAULTS, OUTPUTS_DIR, ndp, cached_sample_for_preset, load_extra_fn,
        load_network,
    )

    names = [p.strip() for p in presets.split(",") if p.strip()]
    unknown = [n for n in names if n not in ndp.PROMPT_PRESETS]
    if unknown:
        raise click.UsageError(f"unknown preset(s): {unknown}")
    missing = [
        p for p in (OUTPUTS_DIR / f"{n}{pkl_suffix}.pkl" for n in names)
        if not p.exists()
    ]
    if missing:
        raise click.UsageError(
            "missing score pickle(s) — compute them first with "
            "`uv run python neural_diffusion_processes/pipeline.py "
            "--prompt-name <preset> --save-path <path below>`:\n  "
            + "\n  ".join(str(p) for p in missing)
        )

    print(f"[sample] n_corr={ndp._FLOW_SAMPLER_NUM_CORRECTIONS} "
          f"n_samples={DEFAULTS['n_samples']} presets={names}", flush=True)

    network = None
    for name in names:
        cache = OUTPUTS_DIR / f"{name}{cache_suffix}_samples.npz"
        if network is None and (resample or not cache.exists()):
            network = load_network()
        cached_sample_for_preset(
            name,
            extra_fn=load_extra_fn(name, suffix=pkl_suffix),
            cache_path=cache, force_resample=resample, network=network,
        )
        print(f"[sample] {name} -> {cache}", flush=True)


if __name__ == "__main__":
    main()
