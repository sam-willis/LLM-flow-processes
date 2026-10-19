"""Refresh one preset's NDP sample cache, reusing its existing LLM artifacts.

Reads the score pickle `<preset><suffix>.pkl` and force-rewrites the sample
cache `<preset><suffix>_samples.npz`, both in $NDP_OUTPUTS_DIR (default
`outputs/`), via `run.cached_sample_for_preset`. Consumers of a cache never
re-sample, so changing a sampler setting needs the cache rebuilt first. This
does exactly that and nothing else: no Qwen, no plotting.

Everything that matters is read from the environment, the same contract as
`neural_diffusion_processes.run` -- NDP_OUTPUTS_DIR for the artifacts,
NDP_N_CORR for the sampler. The npz does NOT record the sampler setting, so
write caches for a non-default n_corr into their own directory (with the score
pickle copied or linked in) or they become indistinguishable from
default-setting ones. `--suffix` is whatever suffix the
score pickle was saved under (empty for `<preset>.pkl`).

  NDP_N_CORR=0 NDP_OUTPUTS_DIR=outputs/ncorr0 \\
  uv run --no-sync python scripts/resample_samples.py --preset changepoint
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from neural_diffusion_processes import run as fc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--flow-dir", type=Path, default=None,
                    help="Checkpoint to sample from (default: NDP_FLOW_DIR).")
    args = ap.parse_args()

    n_corr = fc.ndp._FLOW_SAMPLER_NUM_CORRECTIONS
    cache = fc.OUTPUTS_DIR / f"{args.preset}{args.suffix}_samples.npz"
    print(f"[resample] {args.preset}{args.suffix} n_corr={n_corr} -> {cache}",
          flush=True)

    extra_fn = fc.load_extra_fn(args.preset, suffix=args.suffix)
    flow_dir = args.flow_dir or fc.FLOW_DIR
    network = fc.ndp.load_ndp_flow_network_local(
        flow_dir,
        gamma_scaling_distributional=fc.DEFAULTS["gamma_scaling_distributional"],
        condition_ndp_part=fc.DEFAULTS["condition_ndp_part"])

    fc.cached_sample_for_preset(
        args.preset, extra_fn=extra_fn, network=network, cache_path=cache,
        force_resample=True, suffix=args.suffix)
    print(f"[resample] wrote {cache}", flush=True)


if __name__ == "__main__":
    main()
