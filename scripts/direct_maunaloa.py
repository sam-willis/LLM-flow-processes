"""LLM-Direct baseline for the Mauna Loa weekly CO2 context-size sweep.

`llm_direct.py` looks its
preset up in `PROMPT_PRESETS`, which is a plain dict, so a context-subsampled
copy can be injected under a generated name and passed to
`run_direct_for_preset` without touching the registered preset.

Context subsets come from `maunaloa_ctx.context_subset`, shared with
`llm_score_maunaloa_ctx.py` -- by default evenly-spaced indices into the
preset's own 20 weeks, always retaining index 0 and index 19 so week 128 stays
the context endpoint and the forecast region is identical across n_ctx.

This uses the preset's prompt scaffolding unchanged: Mauna Loa really is a
time series, so the default framing is correct.

Writes `<out-dir>/maunaloa_20rand_ctx<N>[_d<seed>]_direct.pkl` (the LLM-Direct
samples) and a matching `_direct.png` panel per n_ctx.

Usage:
  uv run python scripts/direct_maunaloa.py --n-ctx 3 5 10 15 20
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from neural_diffusion_processes import pipeline as ndp  # noqa: E402
from neural_diffusion_processes import llm_direct as direct  # noqa: E402
from maunaloa_ctx import context_subset, stem  # noqa: E402

# Must agree with maunaloa_ctx.PRESET: the score cache and the LLM-Direct
# samples have to describe the SAME task. Importing it rather than
# repeating the literal is what keeps them in step -- a second hardcoded
# copy here meant NDP_CTX_PRESET reached the scorer but not this script.
from maunaloa_ctx import PRESET  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-ctx", type=int, nargs="+", default=[3, 5, 10, 15, 20])
    p.add_argument("--n-samples", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=2000)
    p.add_argument("--max-thinking-tokens", type=int, default=4096)
    p.add_argument("--enforcer", type=str, default="lmfe")
    p.add_argument("--llm-path", type=str, default=None)
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--ctx-seed", type=int, default=None,
                   help="Draw the context at random instead of the default "
                        "evenly-spaced subsample. Artifacts get a _d<seed> tag.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs",
                   help="Where the pickles and panel PNGs land. Point this at "
                        "a scratch directory to re-run at a different "
                        "--n-samples without overwriting existing results.")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = ndp.PROMPT_PRESETS[PRESET]
    xc_all = np.asarray(base.x_context, dtype=np.float64)
    yc_all = np.asarray(base.y_context, dtype=np.float64)
    ctx_end = float(xc_all.max())

    for n in args.n_ctx:
        xc, yc = context_subset(xc_all, yc_all, n, args.ctx_seed)
        assert xc.max() == ctx_end, "context endpoint moved"

        name = stem(n, args.ctx_seed)
        ndp.PROMPT_PRESETS[name] = dataclasses.replace(
            base,
            x_context=tuple(float(v) for v in xc),
            y_context=tuple(float(v) for v in yc),
        )

        save = args.out_dir / f"{name}_direct.pkl"
        plot = args.out_dir / f"{name}_direct.png"
        if save.exists() and not args.recompute:
            print(f"[direct] {save.name} exists — skipping n_ctx={n}", flush=True)
            continue

        print(f"=== LLM-Direct n_ctx={n} (weeks "
              f"{', '.join(f'{v:.0f}' for v in xc[:6])}"
              f"{'...' if len(xc) > 6 else ''}) -> {save.name} ===", flush=True)
        direct.run_direct_for_preset(
            preset_name=name,
            n_samples=args.n_samples,
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=args.max_new_tokens,
            max_thinking_tokens=args.max_thinking_tokens,
            llm_path=args.llm_path or ndp.DEFAULT_LLM_PATH,
            save_path=save,
            plot_path=plot,
            recompute=args.recompute,
            batch_size=args.batch_size,
            enable_thinking=True,
            enforcer=args.enforcer,
        )


if __name__ == "__main__":
    main()
