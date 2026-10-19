"""Text-conditioned NDP flow via an LLM: sampling, score caches, plots.

Also the package's import surface -- it re-exports the presets, the LLM score
machinery and the plotting helpers, so consumers do
``from neural_diffusion_processes import pipeline as ndp`` rather than
reaching into individual modules.

This module provides:

  1. ``compute_and_save_llm_extra_score_fn(...)``: scores each grid x against
     a y-discretization with a Qwen model (or any HF causal LM) using a
     per-prompt KV cache, then pickles the resulting ``LLMExtraScoreFn``.
  2. ``LLMExtraScoreFn.from_file(save_path)``: reloads that pickle; the
     object provides the additive score term
     ``extra_score(pred_output, pred_output_var) -> log p``, which is
     ``torch.autograd.grad``-able.
  3. NDP-flow samplers (``sample_ndp_flow_no_llm``, ``sample_ndp_flow_with_llm``,
     ``sample_llm_marginal``) and a multi-panel comparison plot.

Usage:

    uv run python neural_diffusion_processes/pipeline.py \
        --save-path ./outputs/llm_probs_demo.pkl \
        --plot-path ./outputs/pipeline_demo.png
"""

from __future__ import annotations

import os

# JAX (for the NDP flow) and PyTorch (for the LLM) share one GPU. By default
# XLA preallocates up to XLA_PYTHON_CLIENT_MEM_FRACTION of *total* VRAM when
# JAX first touches the device, which competes with a large HF model. Prefer
# on-demand allocation and a modest fraction unless set in the environment.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
# Reduce CUDA allocator fragmentation in PyTorch. The Qwen scoring path
# creates many short-lived KV-cache tensors of varying sizes; the default
# allocator's binning can wedge a few GiB into unusable holes. With
# expandable segments PyTorch can grow/shrink contiguous blocks instead.
# (Quoted from the OOM message itself: "If reserved but unallocated memory
# is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True".)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import decimal
import json
import math
import pickle
import textwrap
from dataclasses import dataclass, replace as _dc_replace
from pathlib import Path
from typing import Callable

import click
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from neural_diffusion_processes.paths import default_flow_dir, FLOW_DIR_SHOW_DEFAULT
from neural_diffusion_processes.config import (
    DiffusionModelConfig,
    LLMFlowConfig2,
    TransformerConfig,
    load_model,
)
from neural_diffusion_processes.core import ConditionalFlow, ConditioningDistributionalConvolved1D, ConditioningDistributionalConvolved1DAndX, ConditioningX, sample_n_conditional
from neural_diffusion_processes.core import XPoints, XYPoints
from neural_diffusion_processes.core import FlowProcess
from neural_diffusion_processes.core import VelocitySampler

_FLOW_SAMPLER_NUM_CORRECTIONS = 4
_FLOW_SAMPLER_N_STEPS = 100
_FLOW_SAMPLER_TAU = float(os.environ.get("NDP_TAU", "0.1"))
_FLOW_SAMPLER_START_TIME = 0.1


def _patched_get_sampler(self):
    return VelocitySampler(
        start_time=_FLOW_SAMPLER_START_TIME,
        n_steps=_FLOW_SAMPLER_N_STEPS,
        num_corrections=_FLOW_SAMPLER_NUM_CORRECTIONS,
        tau=_FLOW_SAMPLER_TAU,
    )


FlowProcess._get_sampler = _patched_get_sampler

from neural_diffusion_processes import train as _spt

# Re-exported so this module stays the single import surface for consumers.
from neural_diffusion_processes.plotting import (  # noqa: F401,E402
    _midpoint_edges,
    _plot_basic_panels,
    _plot_llm_only,
    _plot_panels,
    _scored_columns,
)
# PROMPT_PRESETS is re-exported, not copied: scripts/direct_maunaloa.py injects
# entries via `ndp.PROMPT_PRESETS[name] = ...` and llm_direct reads through it
# at call time, so every module must see the same dict object.
from neural_diffusion_processes.presets import (  # noqa: F401,E402
    DEFAULT_PROMPT_NAME,
    PROMPT_PRESETS,
    PromptPreset,
    _default_plot_path,
    _default_save_path,
)
from neural_diffusion_processes.llm_score import (  # noqa: F401,E402
    LLM_LOG_PROBS_BATCH_SIZE,
    LLMExtraScoreFn,
    _build_prob_xys,
    _build_per_target_prompt_after_shared_thinking,
    _compute_shared_thinking_state,
    _generate_thinking,
    _score_y_grid_for_one_target,
    compute_and_save_llm_extra_score_fn,
    llm_extra_log_prob,
)
from neural_diffusion_processes.prompts import (  # noqa: F401,E402
    BASE_SYSTEM_PROMPT,
    DEFAULT_FORMAT_NOUN,
    DEFAULT_LLM_PATH,
    DEFAULT_LLM_TYPE,
    DEFAULT_QUERY_VERB,
    DEFAULT_QWEN_VERSION,
    DEFAULT_REASONING_PRIME,
    DEFAULT_TASK_NOUN,
    GENERIC_BASE_SYSTEM_PROMPT,
    GENERIC_FORMAT_NOUN,
    GENERIC_QUERY_VERB,
    GENERIC_REASONING_PRIME,
    GENERIC_TASK_NOUN,
    GENERIC_THINKING_SYSTEM_PROMPT,
    KEY_POINTS_PROMPT_ADDENDUM,
    LLM_BREAK_STR,
    LLM_X_PRECISION,
    LLM_X_PREFIX,
    LLM_Y_PRECISION,
    LLM_Y_PREFIX,
    QWEN_VERSION_PATHS,
    THINKING_SYSTEM_PROMPT,
    _build_chat_prompt,
    _build_x_llm_grid,
    _build_y_grid_step,
    _compose_user_message,
    _compose_user_message_general,
    _float_to_str,
    _format_x_constraint_line,
    _format_x_value,
    _LLM_DECIMAL_CTX,
)
from neural_diffusion_processes.llm_backend import (  # noqa: F401,E402
    _compute_prefix_kv,
    _expand_kv_cache,
    _load_causal_lm,
    _patch_qwen3_5_gated_delta_net,
    _render_preserving_thinking,
    _score_suffixes_with_kv,
)
from neural_diffusion_processes.flow import (  # noqa: F401,E402
    _make_points_from_np,
    load_ndp_flow_network_local,
    sample_llm_marginal,
    sample_ndp_flow_no_llm,
    sample_ndp_flow_with_llm,
)
from neural_diffusion_processes.metrics import (  # noqa: F401,E402
    empirical_crps,
    evaluate_against_true_fn,
    evaluate_panels_against_true_fn,
    gaussian_nlpd,
    mean_absolute_error,
)




# ---------------------------------------------------------------------------
# Block 1: LLM compute
# ---------------------------------------------------------------------------
# The LLM compute that *builds* an LLMExtraScoreFn pickle -- model loading
# and KV-cache helpers (`llm_backend`), prompt construction (`prompts`),
# per-prompt scoring and the driver `compute_and_save_llm_extra_score_fn`
# (`llm_score`) -- lives in those modules and is re-exported above. Once a
# pickle is on disk, the diffusion side only needs to load it. This block
# holds the ground-truth sidecar writer.
# ---------------------------------------------------------------------------


def write_truth_json(
    path,
    *,
    preset,
    x_grid,
    x_llm,
    preset_text: str | None = None,
    x_unit: str | None = None,
    y_unit: str | None = None,
    log_prefix: str = "[truth]",
) -> bool:
    """Write the ground-truth sidecar for ``preset``; return whether it wrote.

    One schema for every producer: ``preset_text``, ``x_unit``, ``y_unit``,
    ``x_grid``/``true_y_grid`` (the dense NDP grid) and ``x_llm``/``true_y_llm``
    (the LLM target grid). ``llm_direct`` previously wrote a second
    schema keyed ``x_test``/``true_y_test``; every producer now writes this
    one.
    """
    if preset.true_fn is None:
        return False
    import numpy as _np
    tf = preset.true_fn
    xg = _np.asarray(x_grid, dtype=_np.float64)
    xl = _np.asarray(x_llm, dtype=_np.float64)
    truth_doc = {
        "preset_text": preset.text if preset_text is None else preset_text,
        "x_unit": preset.x_unit if x_unit is None else x_unit,
        "y_unit": preset.y_unit if y_unit is None else y_unit,
        "x_grid": [float(v) for v in xg],
        "true_y_grid": [float(tf(float(v))) for v in xg],
        "x_llm": [float(v) for v in xl],
        "true_y_llm": [float(tf(float(v))) for v in xl],
    }
    Path(path).write_text(json.dumps(truth_doc) + "\n")
    print(f"{log_prefix} wrote {path}", flush=True)
    return True


# ---------------------------------------------------------------------------
# Block 2: Diffusion (SWAPPABLE without recomputing the LLM score function)
# ---------------------------------------------------------------------------
# The LLM extra-score log-prob plus the picklable `LLMExtraScoreFn` wrapper
# (both in `llm_score`, re-exported above).
# The downstream NDP-flow sampler consumes `LLMExtraScoreFn.prob_xys` and
# can be swapped freely without rerunning Block 1 or regenerating the
# pickle, as long as it still feeds the score function the marginal
# variance of `y_t - E[y_t | y_0]` in raw coordinates.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Block 2b: NDP flow adapter.
#
# `flow.py` (re-exported above) loads a trained NDP flow network (Equinox /
# JAX) and samples from it conditionally using the precomputed LLM
# `prob_xys` as the distributional guidance term, so no LLM is loaded at
# sampling time -- the torch path in Block 1 has already produced the score
# term we need.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Block 3: Glue - plotting, CLI entry point, demo orchestration.
# ---------------------------------------------------------------------------


def _run_for_prompt(
    prompt_name: str,
    save_path: Path | None,
    plot_path: Path | None,
    llm_path: str,
    llm_type: str | None,
    n_samples: int,
    n_steps: int,
    gamma_llm: float,
    likelihood_variance_llm: float,
    conditioning_time_fraction: float,
    debug_log_every: int,
    recompute_llm: bool,
    enable_thinking: bool,
    max_thinking_tokens: int,
    device: str,
    flow_local_dir: Path,
    gamma_scaling_distributional: float,
    condition_ndp_part: bool,
    gamma_scaling_ndp: float,
    flow_likelihood_variance: float,
    ndp_likelihood_noise_std: float,
    n_ctx: int | None,
    llm_sparse: bool,
    show_llm_panels: bool,
    plot_llm_only: bool,
    truth_only: bool = False,
    key_points_prompt: bool = False,
    phase: str = "both",  # "compute" | "sample" | "both"
) -> None:
    # `phase` lets --all-prompts split the pipeline into a torch-only LLM
    # pass over every preset followed by a jax-only sampling pass, so the
    # two frameworks never compete for GPU memory in the same iteration.
    if phase not in ("compute", "sample", "both"):
        raise ValueError(f"phase must be one of compute|sample|both, got {phase!r}")
    # Resolve preset prompt + per-prompt default paths + numerical setup.
    prompt_name = prompt_name.lower()
    preset = PROMPT_PRESETS[prompt_name]
    language_context = preset.text
    x_unit = preset.x_unit
    y_unit = preset.y_unit
    x_range = preset.x_range
    y_range = preset.y_range
    if n_ctx is not None:
        if preset.context_x_range is None or preset.true_fn is None:
            raise click.UsageError(
                f"--n-ctx requires preset {prompt_name!r} to define "
                f"`context_x_range` and `true_fn`."
            )
        ctx_xs = np.linspace(
            preset.context_x_range[0], preset.context_x_range[1], int(n_ctx),
        )
        x_context = ctx_xs.astype(np.float64)
        y_context = np.asarray(
            [preset.true_fn(float(xv)) for xv in ctx_xs], dtype=np.float64,
        )
    else:
        x_context = np.asarray(preset.x_context, dtype=np.float64)
        y_context = np.asarray(preset.y_context, dtype=np.float64)
    x_grid = np.linspace(x_range[0], x_range[1], preset.n_grid, dtype=np.float64)
    try:
        _x_llm_preview, _ = _build_x_llm_grid(
            x_range=x_range,
            n_grid=preset.n_grid,
            llm_x_strategy=preset.llm_x_strategy,
            llm_n_targets=preset.llm_n_targets,
            llm_x_decimals=preset.llm_x_decimals,
            llm_x_range=preset.llm_x_range,
            x_grid=x_grid,
        )
    except ValueError as exc:  # preset errors surface as CLI usage errors
        raise click.UsageError(f"Preset {prompt_name!r}: {exc}") from exc
    n_grid_llm = len(_x_llm_preview)
    # Metrics are scored at the LLM target x's across the whole x range,
    # context region included: the same evaluation grid for every method.
    eval_mask = np.zeros(len(x_grid), dtype=bool)
    eval_mask[np.abs(x_grid[:, None] - _x_llm_preview[None, :]).argmin(axis=0)] = True
    if save_path is None:
        save_path = _default_save_path(
            prompt_name, enable_thinking, n_ctx,
        )
    if plot_path is None:
        plot_path = _default_plot_path(
            prompt_name, enable_thinking, n_ctx,
        )
    print(
        f"[pipeline] prompt={prompt_name!r} x_unit={x_unit!r} "
        f"y_unit={y_unit!r} x_range={x_range} y_range={y_range} "
        f"n_grid={preset.n_grid} llm_x_strategy={preset.llm_x_strategy!r} "
        f"n_grid_llm={n_grid_llm} llm_sparse={llm_sparse} "
        f"save_path={save_path} plot_path={plot_path}",
        flush=True,
    )

    # --- Early exit: --truth-only just refreshes the sidecar JSON ---------
    if truth_only:
        if preset.true_fn is None:
            raise click.UsageError(
                f"--truth-only requires preset {prompt_name!r} to have a `true_fn`."
            )
        if not Path(save_path).exists():
            raise click.UsageError(
                f"--truth-only requires the LLM pickle to already exist at {save_path}."
            )
        print(f"[truth-only] loading {save_path}", flush=True)
        extra_fn = LLMExtraScoreFn.from_file(
            save_path,
            likelihood_variance=likelihood_variance_llm,
            gamma=gamma_llm,
            sparse=llm_sparse,
        )
        write_truth_json(
            Path(save_path).with_suffix(".truth.json"),
            preset=preset,
            x_grid=extra_fn.x_grid,
            x_llm=extra_fn.x_llm,
            preset_text=extra_fn.language_context,
            x_unit=extra_fn.x_unit,
            y_unit=extra_fn.y_unit,
            log_prefix="[truth-only]",
        )
        return

    # --- Stage 1: build / load the LLM extra-score function -----------------
    # Skipped entirely when `--no-llm-panels` is set: no Qwen load, no LLM
    # scoring, and `extra_fn` stays None so Stage 2's LLM-side samplers
    # short-circuit too.
    extra_fn: LLMExtraScoreFn | None = None
    if show_llm_panels and phase in ("compute", "both"):
        if not save_path.exists() or recompute_llm:
            if save_path.exists():
                print(
                    f"[pipeline] --recompute-llm set; overwriting {save_path}",
                    flush=True,
                )
            compute_and_save_llm_extra_score_fn(
                x_context=x_context,
                y_context=y_context,
                x_grid=x_grid,
                language_context=language_context,
                save_path=save_path,
                x_range=x_range,
                y_range=y_range,
                x_unit=x_unit,
                y_unit=y_unit,
                enable_thinking=enable_thinking,
                max_thinking_tokens=max_thinking_tokens,
                llm_path=llm_path,
                llm_type=llm_type,
                prompt_range=preset.prompt_range,
                llm_x_strategy=preset.llm_x_strategy,
                llm_n_targets=preset.llm_n_targets,
                llm_x_decimals=preset.llm_x_decimals,
                llm_y_decimals=preset.llm_y_decimals,
                llm_x_range=preset.llm_x_range,
                key_points_prompt=key_points_prompt,
                x_formatter=preset.x_formatter,
                x_constraint_text=preset.x_constraint_text,
                strip_context=preset.strip_context,
            )

    # In compute-only mode we exit here: the pickle is on disk; the JAX-side
    # sampling will happen in a later pass.
    if phase == "compute":
        return

    if show_llm_panels:
        print(f"[pipeline] loading {save_path}", flush=True)
        extra_fn = LLMExtraScoreFn.from_file(
            save_path,
            likelihood_variance=likelihood_variance_llm,
            gamma=gamma_llm,
            sparse=llm_sparse,
        )
        # Sidecar truth JSON: written every run when the preset has a
        # `true_fn`, so existing pickles get a fresh JSON without needing
        # `--recompute-llm`.
        write_truth_json(
            Path(save_path).with_suffix(".truth.json"),
            preset=preset,
            x_grid=extra_fn.x_grid,
            x_llm=extra_fn.x_llm,
            preset_text=extra_fn.language_context,
            x_unit=extra_fn.x_unit,
            y_unit=extra_fn.y_unit,
            log_prefix="[pipeline]",
        )
    else:
        print(
            "[pipeline] --no-llm-panels: skipping LLM scoring "
            "and LLM-side sampling; only the NDP-no-LLM panel will be plotted",
            flush=True,
        )

    if plot_llm_only:
        if extra_fn is None:
            raise click.UsageError(
                "--plot-llm-only requires LLM scoring; do not combine with --no-llm-panels."
            )
        print(
            "[pipeline] --plot-llm-only: writing LLM heatmap and exiting "
            "(no GP, no flow checkpoint, no diffusion sampling)",
            flush=True,
        )
        _plot_llm_only(extra_fn=extra_fn, plot_path=plot_path, true_fn=preset.true_fn)
        return

    # --- Stage 2: NDP flow sampling ---------------------------------------
    del conditioning_time_fraction, debug_log_every

    print(
        f"[pipeline] loading NDP flow from local dir {flow_local_dir}",
        flush=True,
    )
    network = load_ndp_flow_network_local(
        flow_local_dir,
        gamma_scaling_distributional=gamma_scaling_distributional,
        condition_ndp_part=condition_ndp_part,
    )

    print("[pipeline] sampling NDP flow: context only", flush=True)
    y_ndp_grid = sample_ndp_flow_no_llm(
        network=network,
        x_context=x_context,
        y_context=y_context,
        x_grid=x_grid,
        x_range=x_range,
        y_range=y_range,
        n_samples=n_samples,
        seed=1,
        likelihood_noise_std=ndp_likelihood_noise_std,
    )

    if not show_llm_panels:
        if preset.true_fn is not None:
            m = evaluate_against_true_fn(
                y_ndp_grid, x_grid, preset.true_fn, mask=eval_mask,
            )
            print(
                f"[metrics] full range, at the {int(eval_mask.sum())} LLM target x's:",
                flush=True,
            )
            print(
                f"  NDP no-LLM  MAE={m['mae']:.3f}  CRPS={m['crps']:.3f}  NLPD={m['nlpd']:.3f}",
                flush=True,
            )
        _plot_basic_panels(
            y_diffusion_context=y_ndp_grid,
            x_context=x_context,
            y_context=y_context,
            x_grid=x_grid,
            x_unit=x_unit,
            y_unit=y_unit,
            language_context=language_context,
            y_lim=y_range,
            plot_path=plot_path,
            true_fn=preset.true_fn,
        )
        return

    assert extra_fn is not None  # guaranteed by `show_llm_panels` branch above
    print("[pipeline] sampling NDP flow: context + LLM", flush=True)
    y_ndp_llm_grid = sample_ndp_flow_with_llm(
        network=network,
        extra_fn=extra_fn,
        n_samples=n_samples,
        gamma_scaling_distributional=gamma_scaling_distributional,
        gamma_scaling_ndp=gamma_scaling_ndp,
        likelihood_variance=flow_likelihood_variance,
        condition_ndp_part=condition_ndp_part,
        seed=2,
    )

    print("[pipeline] sampling ILLMP (LLM marginal per x)", flush=True)
    y_llm_marginal = sample_llm_marginal(
        extra_fn=extra_fn, n_samples=n_samples, seed=3,
    )

    if preset.true_fn is not None:
        print(
            f"[metrics] full range, at the {int(eval_mask.sum())} LLM target x's:",
            flush=True,
        )
        for name, ys in (
            ("NDP no-LLM", y_ndp_grid),
            ("NDP+LLM", y_ndp_llm_grid),
            ("ILLMP", y_llm_marginal),
        ):
            m = evaluate_against_true_fn(ys, x_grid, preset.true_fn, mask=eval_mask)
            print(
                f"  {name:11s} MAE={m['mae']:.3f}  CRPS={m['crps']:.3f}  NLPD={m['nlpd']:.3f}",
                flush=True,
            )

    _plot_panels(
        extra_fn=extra_fn,
        y_diffusion_context=y_ndp_grid,
        y_diffusion_context_plus_llm=y_ndp_llm_grid,
        y_llm_marginal=y_llm_marginal,
        plot_path=plot_path,
        true_fn=preset.true_fn,
    )


@click.command(help=__doc__.split("\n")[0])
@click.option(
    "--prompt-name",
    type=click.Choice(sorted(PROMPT_PRESETS), case_sensitive=False),
    default=DEFAULT_PROMPT_NAME,
    show_default=True,
    help=(
        "Selects which preset language context to use. The text comes from "
        "PROMPT_PRESETS[prompt_name]. The save and plot paths are derived "
        "from this name unless overridden via --save-path / --plot-path. "
        "ONLY takes effect at LLM (re-)compute time - once the pickle is "
        "saved it carries its own language context."
    ),
)
@click.option(
    "--save-path",
    type=click.Path(path_type=Path),
    default=None,
    show_default=False,
    help="Pickle path. Defaults to ./outputs/{prompt_name}.pkl.",
)
@click.option(
    "--plot-path",
    type=click.Path(path_type=Path),
    default=None,
    show_default=False,
    help="Plot path. Defaults to ./outputs/{prompt_name}_demo.png.",
)
@click.option("--llm-path", default=None)
@click.option("--llm-type", default=None)
@click.option(
    "--qwen-version",
    type=click.Choice(list(QWEN_VERSION_PATHS), case_sensitive=False),
    default=DEFAULT_QWEN_VERSION,
    show_default=True,
    help="Qwen variant to use when --llm-path is not given.",
)
@click.option("--n-samples", type=int, default=800, show_default=True)
@click.option("--n-steps", type=int, default=100, show_default=True,
              help="Ignored; the flow sampler always takes 100 steps.")
@click.option(
    "--gamma-llm",
    type=float,
    default=1.0,
    show_default=True,
    help="Linear scale on the LLM extra-score term. Increase to push the LLM hint harder.",
)
@click.option(
    "--likelihood-variance-llm",
    type=float,
    default=0.001,
    show_default=True,
    help=(
        "Likelihood variance floor in `llm_extra_log_prob`."
    ),
)
@click.option(
    "--conditioning-time-fraction",
    type=float,
    default=0.5,
    show_default=True,
    help="Ignored; accepted so older commands still parse. LLM guidance applies at every t >= 0.05.",
)
@click.option(
    "--debug-log-every",
    type=int,
    default=0,
    show_default=True,
    help="Print |eps_uncond| and |llm_grad| every N steps. 0 = off.",
)
@click.option(
    "--recompute-llm",
    is_flag=True,
    default=False,
    help=(
        "Force re-running the LLM scoring even if --save-path already exists. "
        "By default, the saved LLMExtraScoreFn pickle is reused if present."
    ),
)
@click.option(
    "--think/--no-think",
    "enable_thinking",
    default=True,
    show_default=True,
    help=(
        "Enable Qwen's thinking mode at LLM compute time. The model "
        "generates a `<think>...</think>` reasoning chain ONCE about the "
        "function as a whole, then the same reasoning is reused as a "
        "shared prefix for every per-x query. Costs about one extra "
        "thinking generation per recompute (not per target) and unlocks "
        "multi-step reasoning needed for cyclic / multi-hop tasks (e.g. "
        "seasonal forecasting). Only takes effect on (re-)compute."
    ),
)
@click.option(
    "--max-thinking-tokens",
    type=int,
    default=4096,
    show_default=True,
    help="Cap on tokens generated inside the shared thinking block.",
)
@click.option("--device", default="cpu", show_default=True)
@click.option(
    "--flow-local-dir",
    type=click.Path(path_type=Path),
    default=default_flow_dir,
    show_default=FLOW_DIR_SHOW_DEFAULT,
    help=(
        "Local directory with a `train.py` checkpoint "
        "(expects config.json + run.eqx/flow.eqx). Follows NDP_FLOW_DIR, the "
        "same knob run.py uses."
    ),
)
@click.option(
    "--gamma-scaling-distributional",
    type=float,
    default=1.0,
    show_default=True,
    help=(
        "Strength of the LLM distributional guidance term in the flow "
        "sampler."
    ),
)
@click.option(
    "--condition-ndp-part/--no-condition-ndp-part",
    "condition_ndp_part",
    default=True,
    show_default=True,
    help=(
        "Use `ConditioningDistributionalConvolved1DAndX` (adds an NDP-style "
        "context-hard-constraint term on top of the LLM distributional term). "
        "Without this, the LLM term is the only context anchor — fine when "
        "p_LLM is sharp at context, but if it's diffuse the context band "
        "widens toward the prior."
    ),
)
@click.option(
    "--gamma-scaling-ndp",
    type=float,
    default=50.0,
    show_default=True,
    help="NDP-part scaling (only used when --condition-ndp-part is set).",
)
@click.option(
    "--flow-likelihood-variance",
    type=float,
    default=0.01,
    show_default=True,
    help=(
        "Likelihood floor variance used inside the JAX conditioning "
        "handler. Mirrors this module's own --likelihood-variance-llm but "
        "applies to the NDP flow path."
    ),
)
@click.option(
    "--ndp-likelihood-noise-std",
    type=float,
    default=0.001,
    show_default=True,
    help=(
        "Std of the Gaussian likelihood used by the no-LLM NDP "
        "conditioning sampler (sample_ndp_flow_no_llm)."
    ),
)
@click.option(
    "--key-points-prompt/--no-key-points-prompt",
    "key_points_prompt",
    default=False,
    show_default=True,
    help=(
        "Append an instruction to the thinking-mode system prompt asking the "
        "model to identify key points of the function (peaks, troughs, "
        "changepoints, anchor observations) and pin explicit (x, value) "
        "estimates. Intended to bias per-x decoding toward the structural "
        "shape rather than smooth interpolation between observed points."
    ),
)
@click.option(
    "--n-ctx",
    type=int,
    default=None,
    help=(
        "Override context size: resample N evenly-spaced context points "
        "from the preset's `context_x_range`, reading y from `true_fn`. "
        "Only effective on presets that set `context_x_range` + `true_fn`. "
        "Default: use the static x_context/y_context from the preset. "
        "When set, save_path and plot_path defaults include `_n{N}` so "
        "different N values don't clobber each other's caches."
    ),
)
@click.option(
    "--llm-sparse/--llm-dense",
    "llm_sparse",
    default=True,
    show_default=True,
    help="Sparse: LLM rows only at the scored targets, uniform elsewhere.\n"
         "Dense: broadcast the nearest scored row to every grid point.",
)
@click.option(
    "--show-llm-panels/--no-llm-panels",
    "show_llm_panels",
    default=True,
    show_default=True,
    help=(
        "By default the demo plot has 4 panels: NDP flow without LLM, "
        "NDP+LLM, ILLMP marginal, and the LLM log-prob heatmap. With "
        "`--no-llm-panels` the LLM scoring stage is skipped entirely (no "
        "Qwen load, no NDP+LLM or ILLMP sampling) and only the NDP-no-LLM "
        "panel is written. Useful for quick NDP-only iteration."
    ),
)
@click.option(
    "--plot-llm-only",
    is_flag=True,
    default=False,
    help=(
        "Diagnostic shortcut: build/load the LLM extra-score function and write "
        "ONLY the LLM log-prob heatmap to --plot-path, then exit. Skips the "
        "flow checkpoint loading and all diffusion sampling. Useful for "
        "comparing LLM scoring across commits or models without paying the "
        "flow-sampling cost."
    ),
)
@click.option(
    "--truth-only",
    is_flag=True,
    default=False,
    help=(
        "Load the existing pickle (no Qwen, no flow checkpoint, no sampling, "
        "no plot) and write the sidecar truth JSON next to --save-path, then "
        "exit. Requires the preset to have a `true_fn`."
    ),
)
@click.option(
    "--all-prompts",
    is_flag=True,
    default=False,
    help=(
        "Run the full pipeline once per entry in PROMPT_PRESETS (sorted by name). "
        "Each preset uses its own default --save-path / --plot-path; do not pass "
        "those overrides when using this flag."
    ),
)
@click.option(
    "--n-corr", "n_corr",
    type=int, default=4, show_default=True,
    help=(
        "Number of Langevin corrector steps per ODE step in the flow sampler "
        "(VelocitySampler.num_corrections). 0 = pure flow ODE."
    ),
)
def main(
    prompt_name: str,
    save_path: Path | None,
    plot_path: Path | None,
    llm_path: str | None,
    llm_type: str | None,
    qwen_version: str,
    n_samples: int,
    n_steps: int,
    gamma_llm: float,
    likelihood_variance_llm: float,
    conditioning_time_fraction: float,
    debug_log_every: int,
    recompute_llm: bool,
    enable_thinking: bool,
    max_thinking_tokens: int,
    device: str,
    flow_local_dir: Path,
    gamma_scaling_distributional: float,
    condition_ndp_part: bool,
    gamma_scaling_ndp: float,
    flow_likelihood_variance: float,
    ndp_likelihood_noise_std: float,
    n_ctx: int | None,
    llm_sparse: bool,
    show_llm_panels: bool,
    plot_llm_only: bool,
    truth_only: bool,
    key_points_prompt: bool,
    all_prompts: bool,
    n_corr: int,
) -> None:
    if llm_path is None:
        llm_path = QWEN_VERSION_PATHS[qwen_version.lower()]

    # Apply the sampler override before any FlowProcess is constructed.
    global _FLOW_SAMPLER_NUM_CORRECTIONS
    _FLOW_SAMPLER_NUM_CORRECTIONS = int(n_corr)
    print(f"[sampler] VelocitySampler.num_corrections={_FLOW_SAMPLER_NUM_CORRECTIONS}",
          flush=True)

    # One argument list, shared by the --all-prompts loop and the single-prompt
    # call below. These were previously written out twice and had already drifted:
    # `truth_only` was missing from the --all-prompts copy, so `--all-prompts
    # --truth-only` silently ignored the flag.
    run_kwargs = dict(
        llm_path=llm_path,
        llm_type=llm_type,
        n_samples=n_samples,
        n_steps=n_steps,
        gamma_llm=gamma_llm,
        likelihood_variance_llm=likelihood_variance_llm,
        conditioning_time_fraction=conditioning_time_fraction,
        debug_log_every=debug_log_every,
        recompute_llm=recompute_llm,
        enable_thinking=enable_thinking,
        max_thinking_tokens=max_thinking_tokens,
        device=device,
        flow_local_dir=flow_local_dir,
        gamma_scaling_distributional=gamma_scaling_distributional,
        condition_ndp_part=condition_ndp_part,
        gamma_scaling_ndp=gamma_scaling_ndp,
        flow_likelihood_variance=flow_likelihood_variance,
        ndp_likelihood_noise_std=ndp_likelihood_noise_std,
        n_ctx=n_ctx,
        llm_sparse=llm_sparse,
        show_llm_panels=show_llm_panels,
        plot_llm_only=plot_llm_only,
        truth_only=truth_only,
        key_points_prompt=key_points_prompt,
    )

    if all_prompts:
        if save_path is not None or plot_path is not None:
            raise click.UsageError(
                "--all-prompts cannot be combined with --save-path or --plot-path; "
                "each preset uses its own default cache and plot paths."
            )
        # Two-phase pass:
        #   1. compute every prompt's LLM pickle (torch only — JAX never
        #      runs a sampling op during this phase).
        #   2. load each pickle and run the JAX-side sampling + plotting
        #      (torch is already freed by `compute_and_save_llm_extra_score_fn`).
        # This keeps the two frameworks from sharing peak GPU at any moment.
        common_kwargs = dict(run_kwargs, save_path=None, plot_path=None)
        names = sorted(PROMPT_PRESETS)
        if show_llm_panels:
            print(
                f"\n[all-prompts] === phase 1: LLM compute ({len(names)} "
                f"prompts) ===\n", flush=True,
            )
            for name in names:
                print(f"\n[all-prompts][compute] {name}\n", flush=True)
                _run_for_prompt(prompt_name=name, phase="compute", **common_kwargs)
        print(
            f"\n[all-prompts] === phase 2: NDP sampling + plot "
            f"({len(names)} prompts) ===\n", flush=True,
        )
        for name in names:
            print(f"\n[all-prompts][sample] {name}\n", flush=True)
            _run_for_prompt(prompt_name=name, phase="sample", **common_kwargs)
        return

    _run_for_prompt(
        prompt_name=prompt_name,
        save_path=save_path,
        plot_path=plot_path,
        **run_kwargs,
    )


if __name__ == "__main__":
    main()
