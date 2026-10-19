"""LLM score grid for one (n_ctx, draw) of the Mauna Loa weekly sweep.

Computes only the LLM score grid, one (n_ctx, draw) cell per invocation, so
the expensive Qwen pass can be run as independent jobs and the flow sampling
can happen later. The context subset comes from `maunaloa_ctx.context_subset`
(shared with `direct_maunaloa.py`), and the scoring settings are the preset's
own.

Writes `<out-dir>/maunaloa_20rand_ctx<N>[_d<seed>]_llm.pkl`, a score pickle
loadable with `pipeline.LLMExtraScoreFn.from_file`.

  uv run --no-sync python scripts/llm_score_maunaloa_ctx.py \\
      --n-ctx 10 --ctx-seed 2 --out-dir outputs/ctx_draws
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from neural_diffusion_processes import pipeline as ndp  # noqa: E402
from maunaloa_ctx import PRESET, context_subset, stem  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-ctx", type=int, required=True)
    p.add_argument("--ctx-seed", type=int, default=None,
                   help="Omit for the default evenly-spaced subsample.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    p.add_argument("--max-thinking-tokens", type=int, default=4096)
    p.add_argument("--llm-path", type=str, default=None)
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    preset = ndp.PROMPT_PRESETS[PRESET]
    xc, yc = context_subset(preset.x_context, preset.y_context,
                            args.n_ctx, args.ctx_seed)
    x_grid = np.linspace(preset.x_range[0], preset.x_range[1], preset.n_grid)

    out = args.out_dir / f"{stem(args.n_ctx, args.ctx_seed)}_llm.pkl"
    if out.exists() and not args.recompute:
        print(f"[llm] {out.name} exists — skipping", flush=True)
        return

    print(f"[llm] n_ctx={args.n_ctx} seed={args.ctx_seed} "
          f"weeks={np.round(xc).astype(int).tolist()} -> {out.name}", flush=True)
    ndp.compute_and_save_llm_extra_score_fn(
        x_context=xc, y_context=yc, x_grid=x_grid,
        language_context=preset.text, save_path=out,
        x_range=preset.x_range, y_range=preset.y_range,
        x_unit=preset.x_unit, y_unit=preset.y_unit,
        enable_thinking=True,
        max_thinking_tokens=args.max_thinking_tokens,
        llm_path=args.llm_path,
        prompt_range=preset.prompt_range,
        llm_x_strategy=preset.llm_x_strategy,
        llm_n_targets=preset.llm_n_targets,
        llm_x_decimals=preset.llm_x_decimals,
        llm_y_decimals=preset.llm_y_decimals,
        llm_x_range=preset.llm_x_range,
        x_formatter=preset.x_formatter,
        x_constraint_text=preset.x_constraint_text,
        strip_context=preset.strip_context,
        # No prompt-scaffolding overrides: this IS a time series, so the
        # default framing is correct here.
    )
    print(f"[llm] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
