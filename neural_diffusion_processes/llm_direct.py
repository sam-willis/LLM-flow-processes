"""LLM-only "Direct Prompt" baseline (a la Context-is-Key, arXiv:2410.18959).

Reuses the shared-thinking machinery from ``pipeline`` but skips the per-x
logit grid and the NDP flow entirely. Instead:

  1. generate the shared ``<think>...</think>`` block ONCE for the function;
  2. append a single user turn asking for the entire forecast vector;
  3. sample N completions from that prefix (do_sample=True);
  4. parse each completion into a length-T trajectory of ``(x, y)`` pairs;
  5. save samples + plot a single panel of sampled trajectories.

Saved pickle is overlay-ready (no shared format with ``LLMExtraScoreFn``):

    {"preset_name", "language_context", "x_test" (T,), "samples" (n_kept, T),
     "thinking_text", "raw_completions" (list[str]), "n_attempted",
     "model_path"}
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import pickle
import re
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

from neural_diffusion_processes import pipeline as ndp

# lmformatenforcer's transformers integration imports
# `transformers.tokenization_utils.PreTrainedTokenizerBase`, but newer
# transformers moved that name to `transformers.tokenization_utils_base`.
# Patch the alias before importing the integration module so the fix lives in
# project code rather than requiring edits to site-packages.
import transformers.tokenization_utils as _tf_tu  # noqa: E402
import transformers.tokenization_utils_base as _tf_tub  # noqa: E402

if not hasattr(_tf_tu, "PreTrainedTokenizerBase"):
    _tf_tu.PreTrainedTokenizerBase = _tf_tub.PreTrainedTokenizerBase

from lmformatenforcer import RegexParser  # noqa: E402
from lmformatenforcer.integrations.transformers import (  # noqa: E402
    build_transformers_prefix_allowed_tokens_fn,
)


# ---------------------------------------------------------------------------
# x_llm grid: same construction as compute_and_save_llm_extra_score_fn.
# ---------------------------------------------------------------------------
def _build_x_llm(
    preset: ndp.PromptPreset,
    n_llm_targets_override: int | None = None,
) -> tuple[np.ndarray, int]:
    """Return ``(x_llm, x_decimals)`` matching the per-x scorer's grid.

    Delegates to the scorer's own grid builder so the two cannot drift. This
    previously re-derived the rule by hand and had dropped ``llm_x_range``
    support, so LLM-Direct queried a different grid than the scorer for any
    preset that set it.

    ``n_llm_targets_override`` (if set) replaces ``preset.llm_n_targets`` and
    forces linspace_subset on integer-strategy presets so the override applies
    there too. Useful for studying T effects without editing the preset.
    """
    return ndp._build_x_llm_grid(
        x_range=preset.x_range,
        n_grid=int(preset.n_grid),
        llm_x_strategy=preset.llm_x_strategy,
        llm_n_targets=preset.llm_n_targets,
        llm_x_decimals=preset.llm_x_decimals,
        llm_x_range=preset.llm_x_range,
        n_targets_override=n_llm_targets_override,
    )


# ---------------------------------------------------------------------------
# Build the single user turn that asks for the whole forecast vector.
# ---------------------------------------------------------------------------
def _build_direct_user_message(
    x_llm: np.ndarray,
    x_unit: str,
    x_decimals: int,
    x_formatter=None,
) -> str:
    unit = x_unit.lower()
    units = unit + "s"
    x_strs = [ndp._format_x_value(float(v), x_decimals, x_formatter) for v in x_llm]
    listing = ", ".join(x_strs)
    n = len(x_llm)
    if x_formatter is not None:
        list_label = "dates"
        line_label = "date"
    else:
        list_label = units
        line_label = unit
    return (
        f"Now produce the full forecast.\n"
        f"\n"
        f"Forecast at the following {list_label} (in this exact order):\n"
        f"{listing}\n"
        f"\n"
        f"Reply with EXACTLY {n} lines and nothing else. Each line must be "
        f"`{line_label}, value` — the {line_label} from the list above, then "
        f"a comma and a space, then the predicted value as a real number. "
        f"Output the {n} lines in the same order as the list. Do not include "
        f"headers, commentary, units, blank lines, or any text before or "
        f"after the {n} lines."
    )


# ---------------------------------------------------------------------------
# Parse a model completion into a length-T y vector.
# ---------------------------------------------------------------------------
_LINE_RE = re.compile(r"^\s*([^,]+?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")


# ---------------------------------------------------------------------------
# Regex used by lm-format-enforcer to guarantee well-formed output.
# ---------------------------------------------------------------------------
def _build_format_regex(x_llm: np.ndarray, x_decimals: int, x_formatter=None) -> str:
    """Regex that matches exactly ``len(x_llm)`` lines, each
    ``"{x_k}, {y}\\n"`` where ``x_k`` is the k-th x-string at the configured
    precision and ``y`` is any signed real number.

    Anchors: none — lmformatenforcer treats the regex as a full-string match
    spec via its character-level parser, and we want the regex to be
    "complete" (can_end) only after the last line's newline has been emitted.
    """
    x_strs = [ndp._format_x_value(float(v), x_decimals, x_formatter) for v in x_llm]
    y_pat = r"-?\d+(?:\.\d+)?"
    line_parts = [re.escape(s) + r", " + y_pat + r"\n" for s in x_strs]
    return "".join(line_parts)


def _parse_completion(
    text: str, x_llm: np.ndarray, x_decimals: int, x_formatter=None,
) -> np.ndarray | None:
    """Return ``y`` of shape ``(T,)`` if ``text`` parses cleanly, else ``None``.

    Tolerates surrounding whitespace and extra blank lines but requires
    exactly T well-formed ``"x, y"`` lines whose x-values match ``x_llm``
    in order (verbatim when an x_formatter is set; otherwise compared as
    numeric strings at ``x_decimals`` precision).
    """
    expected_x = [ndp._format_x_value(float(v), x_decimals, x_formatter) for v in x_llm]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) != len(x_llm):
        return None
    ys: list[float] = []
    for line, x_expected in zip(lines, expected_x):
        m = _LINE_RE.match(line)
        if m is None:
            return None
        x_str, y_str = m.group(1).strip(), m.group(2)
        if x_formatter is not None:
            if x_str != x_expected:
                return None
        else:
            try:
                x_norm = ndp._float_to_str(float(x_str), x_decimals)
            except ValueError:
                return None
            if x_norm != x_expected:
                return None
        try:
            ys.append(float(y_str))
        except ValueError:
            return None
    return np.asarray(ys, dtype=np.float64)


# ---------------------------------------------------------------------------
# Build the chat-templated prefix string for direct decoding.
# ---------------------------------------------------------------------------
def _render_direct_prefix(
    tokenizer,
    messages_prefix: list[dict],
    direct_user_message: str,
) -> str:
    full_messages = messages_prefix + [
        {"role": "user", "content": direct_user_message},
    ]
    return ndp._render_preserving_thinking(
        tokenizer,
        full_messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ---------------------------------------------------------------------------
# Driver for one preset.
# ---------------------------------------------------------------------------
def run_direct_for_preset(
    *,
    preset_name: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_thinking_tokens: int,
    llm_path: str,
    save_path: Path,
    plot_path: Path,
    recompute: bool,
    batch_size: int,
    enable_thinking: bool = True,
    n_llm_targets: int | None = None,
    enforcer: str = "lmfe",
    reuse_thinking: bool = True,
    append: bool = False,
    # Prompt-scaffolding overrides. All default to the time-series wording, so
    # every registered preset renders exactly as before.
    system_prompt: str | None = None,
    base_system_prompt: str | None = None,
    # Render the observed-data lines at this precision instead of `x_dec`.
    # llm_x_strategy="integer" forces x_dec=0 for targets, which would
    # otherwise also round the context (0.13 km -> "0"). Mirrors
    # `llm_x_data_decimals` in compute_and_save_llm_extra_score_fn.
    x_data_decimals: int | None = None,
    task_noun: str | None = None,
    reasoning_prime: str | None = None,
    format_noun: str | None = None,
) -> None:
    if save_path.exists() and not recompute and not append:
        print(f"[direct] {preset_name}: pickle exists at {save_path}, loading for plot only.", flush=True)
        with open(save_path, "rb") as f:
            payload = pickle.load(f)
    else:
        preset = ndp.PROMPT_PRESETS[preset_name]
        x_llm, x_dec = _build_x_llm(preset, n_llm_targets_override=n_llm_targets)
        y_dec = int(preset.llm_y_decimals)
        x_dec_data = x_dec if x_data_decimals is None else int(x_data_decimals)
        prompt_y_range = preset.prompt_range if preset.prompt_range is not None else preset.y_range

        print(
            f"[direct] preset={preset_name} n_samples={n_samples} "
            f"T={len(x_llm)} x_unit={preset.x_unit!r} x_dec={x_dec} y_dec={y_dec} "
            f"temperature={temperature} top_p={top_p} max_new_tokens={max_new_tokens}",
            flush=True,
        )

        ndp._patch_qwen3_5_gated_delta_net()
        print(f"[direct] loading {llm_path!r}", flush=True)
        model, tokenizer = ndp._load_causal_lm(llm_path, "qwen")

        x_context_np = np.asarray(preset.x_context, dtype=np.float64)
        y_context_np = np.asarray(preset.y_context, dtype=np.float64)

        # Thinking cache. The block is generated greedily (argmax in
        # `_generate_thinking`), so regenerating it is reproducible but not
        # free — at large max_thinking_tokens it can take minutes of decode
        # per invocation. Caching lets a long run be split into chunks that all
        # share one prefix *by construction*, which is what makes their
        # samples poolable into a single ensemble rather than a mixture over
        # prefixes.
        #
        # What is cached is the fully rendered `prefix_text`, not just the
        # thinking. Rebuilding the prefix from thinking alone would mean
        # duplicating the `</think>` split and assistant-message assembly out
        # of `_compute_shared_thinking_state`, which would then silently drift
        # from it. Caching the rendered string has no such failure mode: what
        # is replayed is exactly the bytes the model was fed.
        thinking_cache = save_path.with_suffix(".thinking.json")
        cache_key = dict(preset_name=preset_name, model_path=llm_path,
                         max_thinking_tokens=int(max_thinking_tokens),
                         enable_thinking=bool(enable_thinking),
                         n_targets=int(len(x_llm)))
        cached = None
        if reuse_thinking and thinking_cache.exists():
            import json as _json
            blob = _json.loads(thinking_cache.read_text())
            if all(blob.get(k) == v for k, v in cache_key.items()):
                cached = blob
                print(f"[direct] reusing cached prefix from {thinking_cache} "
                      f"({len(blob['prefix_text'])} chars) — no thinking "
                      f"regeneration", flush=True)
            else:
                differing = [k for k, v in cache_key.items()
                             if blob.get(k) != v]
                print(f"[direct] thinking cache at {thinking_cache} differs on "
                      f"{differing}; regenerating", flush=True)

        if cached is not None:
            thinking_text = cached["thinking_text"]
            prefix_text = cached["prefix_text"]
        elif enable_thinking:
            print(f"[direct] generating shared thinking (max_tokens={max_thinking_tokens})...", flush=True)
            messages_prefix, thinking_text = ndp._compute_shared_thinking_state(
                tokenizer=tokenizer,
                model=model,
                x_train=x_context_np,
                y_train=y_context_np,
                language_context=preset.text,
                x_range=preset.x_range,
                y_range=prompt_y_range,
                num_decimal_x=x_dec_data,
                num_decimal_y=y_dec,
                x_unit=preset.x_unit,
                y_unit=preset.y_unit,
                max_thinking_tokens=max_thinking_tokens,
                x_formatter=preset.x_formatter,
                x_constraint_text=preset.x_constraint_text,
                system_prompt=system_prompt,
                task_noun=task_noun or ndp.DEFAULT_TASK_NOUN,
                reasoning_prime=reasoning_prime or ndp.DEFAULT_REASONING_PRIME,
                format_noun=format_noun or ndp.DEFAULT_FORMAT_NOUN,
            )
            n_think_tokens = len(tokenizer.encode(thinking_text, add_special_tokens=False))
            print(
                f"[direct] thinking generated ({n_think_tokens}/{max_thinking_tokens} tokens)",
                flush=True,
            )
        else:
            print("[direct] --no-think: skipping shared thinking", flush=True)
            general_user_message = ndp._compose_user_message_general(
                x_context_np, y_context_np, preset.text,
                preset.x_range, prompt_y_range,
                x_dec_data, y_dec, preset.x_unit, preset.y_unit,
                x_formatter=preset.x_formatter,
                x_constraint_text=preset.x_constraint_text,
                task_noun=task_noun or ndp.DEFAULT_TASK_NOUN,
                reasoning_prime=reasoning_prime or ndp.DEFAULT_REASONING_PRIME,
                format_noun=format_noun or ndp.DEFAULT_FORMAT_NOUN,
            )
            messages_prefix = [
                {"role": "system",
                 "content": base_system_prompt or ndp.BASE_SYSTEM_PROMPT},
                {"role": "user", "content": general_user_message},
                {"role": "assistant", "content": "Ready to forecast."},
            ]
            thinking_text = ""

        if cached is None:
            direct_user_message = _build_direct_user_message(
                x_llm, preset.x_unit, x_dec, x_formatter=preset.x_formatter,
            )
            prefix_text = _render_direct_prefix(
                tokenizer, messages_prefix, direct_user_message)
            import json as _json
            thinking_cache.parent.mkdir(parents=True, exist_ok=True)
            thinking_cache.write_text(_json.dumps(
                {**cache_key, "thinking_text": thinking_text,
                 "prefix_text": prefix_text}, indent=2))
            print(f"[direct] cached prefix to {thinking_cache}", flush=True)
        print(
            f"[direct] full direct prefix ({len(prefix_text)} chars):\n"
            f"--- prefix ---\n{prefix_text}\n--- end ---",
            flush=True,
        )

        device = next(model.parameters()).device
        prefix_ids = tokenizer.encode(prefix_text, return_tensors="pt").to(device)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

        # Build the format-enforcer prefix_allowed_tokens_fn from the regex.
        # This guarantees every sampled completion matches the requested
        # `(x, y)\n × T` schema; the post-hoc parser then only has to handle
        # surface whitespace.
        #
        # The enforcer is by far the dominant cost of this script: constrained
        # decoding is much slower than unconstrained, and batching can yield
        # less aggregate throughput than a single free stream, because
        # `prefix_allowed_tokens_fn` is a per-sequence Python call on every
        # step and so scales with the batch. Disabling it trades
        # the hard format guarantee for whatever the post-hoc parser can
        # recover; `n_kept/n_attempted` in the log says how much that cost.
        # xgrammar compiles the regex to a token-level DFA once and applies it
        # as a GPU bitmask per step, instead of lmformatenforcer's per-sequence
        # Python callback that returns a token-id list. Same hard guarantee,
        # but the cost stops scaling with the batch.
        xgr_compiled = None
        if enforcer == "xgrammar":
            import xgrammar as xgr
            format_regex = _build_format_regex(x_llm, x_dec,
                                               x_formatter=preset.x_formatter)
            print(f"[direct] xgrammar: compiling regex "
                  f"({len(format_regex)} chars) ...", flush=True)
            import time as _time
            _t0 = _time.time()
            # vocab_size from the config, not the tokenizer: it includes the
            # padding rows the logits tensor actually has, and a bitmask sized
            # to the tokenizer would misalign against them.
            tok_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer, vocab_size=model.config.vocab_size)
            xgr_compiled = xgr.GrammarCompiler(tok_info).compile_regex(format_regex)
            print(f"[direct] xgrammar: compiled in {_time.time() - _t0:.1f}s",
                  flush=True)

        prefix_allowed_tokens_fn = None
        if enforcer == "lmfe":
            format_regex = _build_format_regex(x_llm, x_dec,
                                               x_formatter=preset.x_formatter)
            print(
                f"[direct] format regex (first 200 chars): {format_regex[:200]}",
                flush=True,
            )
            regex_parser = RegexParser(format_regex)
            prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(
                tokenizer, regex_parser,
            )
        elif enforcer == "none":
            print("[direct] format enforcer DISABLED — completions are "
                  "unconstrained and rely on the post-hoc parser", flush=True)

        # Sample in mini-batches so we don't blow up VRAM at large n_samples.
        raw_completions: list[str] = []
        remaining = int(n_samples)
        with torch.inference_mode():
            while remaining > 0:
                bs = min(batch_size, remaining)
                # The xgrammar processor carries per-sequence DFA state, so it
                # must be rebuilt per generate() call — reusing one across
                # batches would resume mid-grammar and reject valid first
                # tokens. Compilation is NOT repeated; only the matcher is.
                logits_processor = None
                if xgr_compiled is not None:
                    import xgrammar.contrib.hf as _xgr_hf
                    logits_processor = [_xgr_hf.LogitsProcessor(xgr_compiled)]
                out_ids = model.generate(
                    input_ids=prefix_ids,
                    do_sample=True,
                    logits_processor=logits_processor,
                    num_return_sequences=bs,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_new_tokens=int(max_new_tokens),
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                )
                gen_ids = out_ids[:, prefix_ids.shape[1]:]
                for row in gen_ids:
                    raw_completions.append(
                        tokenizer.decode(row, skip_special_tokens=True)
                    )
                remaining -= bs
                print(f"[direct] sampled {len(raw_completions)}/{n_samples}", flush=True)

        # Parse.
        kept: list[np.ndarray] = []
        for text in raw_completions:
            y = _parse_completion(text, x_llm, x_dec, x_formatter=preset.x_formatter)
            if y is not None:
                kept.append(y)
        n_kept = len(kept)
        samples = (
            np.stack(kept, axis=0) if n_kept > 0
            else np.zeros((0, len(x_llm)), dtype=np.float64)
        )
        print(
            f"[direct] parsed {n_kept}/{n_samples} completions cleanly "
            f"(shape={samples.shape})",
            flush=True,
        )

        # Chunked accumulation. `--append` grows an existing pickle instead of
        # replacing it, so an 800-sample ensemble can be built as 10 runs of
        # 80 and inspected between chunks. The prefix must match exactly or
        # the pooled samples are a mixture over two different reasoning
        # traces, not a larger draw from one predictive — hence the hard stop
        # rather than a warning.
        if append and save_path.exists():
            with open(save_path, "rb") as f:
                prev = pickle.load(f)
            if prev.get("thinking_text") != thinking_text:
                raise SystemExit(
                    f"--append: {save_path} was generated from a different "
                    f"thinking block. Pooling would mix two prefixes. Delete "
                    f"the pickle, or keep {thinking_cache} so chunks share "
                    f"one prefix.")
            if not np.allclose(np.asarray(prev["x_test"], dtype=np.float64), x_llm):
                raise SystemExit(f"--append: {save_path} has a different x grid.")
            samples = np.concatenate([np.asarray(prev["samples"]), samples], axis=0)
            raw_completions = list(prev["raw_completions"]) + raw_completions
            n_attempted = int(prev.get("n_attempted", 0)) + int(n_samples)
            print(f"[direct] appended to {len(prev['samples'])} existing "
                  f"samples -> {samples.shape[0]} kept / {n_attempted} "
                  f"attempted", flush=True)
        else:
            n_attempted = int(n_samples)

        payload = {
            "preset_name": preset_name,
            "language_context": preset.text,
            "x_test": x_llm,
            "x_decimals": x_dec,
            "samples": samples,
            "thinking_text": thinking_text,
            "raw_completions": raw_completions,
            "n_attempted": n_attempted,
            "model_path": llm_path,
            "x_unit": preset.x_unit,
            "y_unit": preset.y_unit,
            "x_context": x_context_np,
            "y_context": y_context_np,
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[direct] saved {save_path}", flush=True)

        # Free GPU before plotting.
        import gc as _gc
        del model, tokenizer
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # Sidecar truth JSON: written every run when the preset has `true_fn`,
    # so existing pickles get a fresh JSON without needing `--recompute`.
    preset = ndp.PROMPT_PRESETS[payload["preset_name"]]
    if preset.true_fn is not None:
        # `x_test` is this script's name for the LLM target grid, i.e. `x_llm`.
        # The dense grid is rebuilt the same way the sampler builds it, so the
        # sidecar carries both and matches the sampler's schema exactly.
        x_llm_arr = np.asarray(payload["x_test"], dtype=np.float64)
        x_grid_arr = np.linspace(
            preset.x_range[0], preset.x_range[1], int(preset.n_grid), dtype=np.float64,
        )
        ndp.write_truth_json(
            save_path.with_suffix(".truth.json"),
            preset=preset,
            x_grid=x_grid_arr,
            x_llm=x_llm_arr,
            log_prefix="[direct]",
        )

    _plot_direct_panel(payload, plot_path)


# ---------------------------------------------------------------------------
# Single-panel plot: sampled trajectories + context points (+ true_fn if any).
# ---------------------------------------------------------------------------
def _plot_direct_panel(payload: dict, plot_path: Path) -> None:
    import textwrap as _tw

    preset = ndp.PROMPT_PRESETS[payload["preset_name"]]
    x_test = np.asarray(payload["x_test"], dtype=np.float64)
    samples = np.asarray(payload["samples"], dtype=np.float64)
    x_ctx = np.asarray(payload["x_context"], dtype=np.float64)
    y_ctx = payload["y_context"]

    metric_suffix = ""
    if preset.true_fn is not None and samples.shape[0] > 0:
        # Full x range at the LLM target x's, context region included.
        m = ndp.evaluate_against_true_fn(samples, x_test, preset.true_fn)
        print(
            f"[metrics] LLM-Direct full range ({len(x_test)} target x's): "
            f"MAE={m['mae']:.3f} CRPS={m['crps']:.3f} NLPD={m['nlpd']:.3f}",
            flush=True,
        )
        metric_suffix = (
            f"  |  MAE={m['mae']:.2f} CRPS={m['crps']:.2f} "
            f"NLPD={m['nlpd']:.2f}"
        )

    prompt_text = preset.text
    wrapped_prompt = "\n".join(_tw.wrap(prompt_text, width=85)) if prompt_text else ""
    n_prompt_lines = wrapped_prompt.count("\n") + 1 if wrapped_prompt else 0
    extra_h = 0.22 * max(n_prompt_lines, 0)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5 + extra_h))

    if samples.shape[0] > 0:
        for i in range(samples.shape[0]):
            ax.plot(x_test, samples[i], color="tab:blue", alpha=0.25, linewidth=0.8)
        mean = samples.mean(axis=0)
        ax.plot(x_test, mean, color="tab:blue", linewidth=2.0, label=f"LLM-Direct mean (n={samples.shape[0]})")
    else:
        ax.text(
            0.5, 0.5, "0 samples parsed cleanly",
            transform=ax.transAxes, ha="center", va="center", color="red",
        )

    if preset.true_fn is not None:
        x_dense = np.linspace(preset.x_range[0], preset.x_range[1], 401)
        y_true = np.array([preset.true_fn(float(xv)) for xv in x_dense])
        ax.plot(x_dense, y_true, color="black", linewidth=1.5, linestyle="--", label="true_fn")

    ax.scatter(x_ctx, y_ctx, color="black", s=20, zorder=5, label="context")

    ax.set_xlabel(preset.x_unit)
    ax.set_ylabel(preset.y_unit if preset.y_unit else "value")
    ax.set_xlim(preset.x_range)
    ax.set_ylim(preset.y_range)
    ax.set_title(f"{payload['preset_name']} — LLM-Direct (Qwen){metric_suffix}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if wrapped_prompt:
        top = 1.0 - (0.05 * n_prompt_lines + 0.02)
        fig.subplots_adjust(top=top)
        fig.text(
            0.02, 0.995, "prompt:\n" + wrapped_prompt,
            ha="left", va="top", fontsize=8, family="monospace",
            bbox=dict(facecolor="white", edgecolor="grey", alpha=0.9, pad=4),
        )
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"[direct] wrote {plot_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def _default_save_path(prompt_name: str) -> Path:
    return Path(f"./outputs/{prompt_name}_direct.pkl")


def _default_plot_path(prompt_name: str) -> Path:
    return Path(f"./outputs/{prompt_name}_direct_demo.png")


@click.command(help=__doc__.split("\n")[0])
@click.option(
    "--prompt-name",
    type=click.Choice(sorted(ndp.PROMPT_PRESETS), case_sensitive=False),
    default=ndp.DEFAULT_PROMPT_NAME,
    show_default=True,
)
@click.option("--save-path", type=click.Path(path_type=Path), default=None)
@click.option("--plot-path", type=click.Path(path_type=Path), default=None)
@click.option(
    "--qwen-version",
    type=click.Choice(list(ndp.QWEN_VERSION_PATHS), case_sensitive=False),
    default="3.6", show_default=True,
)
@click.option("--llm-path", default=None)
@click.option("--n-samples", type=int, default=800, show_default=True)
@click.option("--temperature", type=float, default=0.7, show_default=True)
@click.option("--top-p", type=float, default=0.9, show_default=True)
@click.option(
    "--max-new-tokens", type=int, default=4096, show_default=True,
    help="Cap on tokens generated for the forecast vector decode.",
)
@click.option(
    "--max-thinking-tokens", type=int, default=4096, show_default=True,
    help="Cap on tokens inside the shared <think> block.",
)
@click.option(
    "--batch-size", type=int, default=8, show_default=True,
    help="num_return_sequences per generate() call. Lower if VRAM-constrained.",
)
@click.option(
    "--enforcer", type=click.Choice(["lmfe", "xgrammar", "none"]),
    default="lmfe", show_default=True,
    help="How to constrain decoding to the (x, y) schema. 'lmfe' is "
         "lmformatenforcer, which dominates runtime (much slower than "
         "unconstrained decoding) because it is "
         "a per-sequence Python callback per token and so scales with "
         "--batch-size. 'xgrammar' compiles the regex to a token-level DFA "
         "once and applies a GPU bitmask, keeping the guarantee without the "
         "per-sequence cost. 'none' drops the guarantee and relies on the "
         "post-hoc parser — check n_kept/n_attempted for the yield lost.",
)
@click.option(
    "--reuse-thinking/--no-reuse-thinking", default=True, show_default=True,
    help="Reuse the rendered prefix cached alongside the pickle "
         "(<save_path>.thinking.json) instead of regenerating the shared "
         "thinking block. Thinking is greedy so it is reproducible anyway; "
         "caching saves that decode time per invocation and guarantees "
         "chunks of one run share a byte-identical prefix.",
)
@click.option(
    "--append", is_flag=True, default=False,
    help="Append to an existing pickle rather than replacing it, so a large "
         "ensemble can be built in chunks and inspected in between. Refuses "
         "to pool chunks whose thinking block or x grid differ.",
)
@click.option(
    "--recompute", is_flag=True, default=False,
    help="Force re-running even if save_path already exists.",
)
@click.option(
    "--think/--no-think", "enable_thinking", default=True, show_default=True,
    help="Generate the shared Qwen <think> block before sampling. With "
         "--no-think the model goes straight from system+user to direct "
         "decoding (saves ~thinking-budget tokens of latency).",
)
@click.option(
    "--all-prompts", is_flag=True, default=False,
    help="Run all PROMPT_PRESETS in sorted order with default save/plot paths.",
)
@click.option(
    "--n-llm-targets", type=int, default=None,
    help=(
        "Override preset.llm_n_targets — number of x positions the LLM "
        "forecasts at. Forces linspace_subset strategy on integer-strategy "
        "presets so the override is meaningful there too. Default: use the "
        "preset's strategy + count unchanged."
    ),
)
def main(
    prompt_name: str,
    save_path: Path | None,
    plot_path: Path | None,
    qwen_version: str,
    llm_path: str | None,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_thinking_tokens: int,
    batch_size: int,
    enforcer: str,
    reuse_thinking: bool,
    append: bool,
    recompute: bool,
    enable_thinking: bool,
    all_prompts: bool,
    n_llm_targets: int | None,
) -> None:
    if llm_path is None:
        llm_path = ndp.QWEN_VERSION_PATHS[qwen_version.lower()]

    if all_prompts:
        if save_path is not None or plot_path is not None:
            raise click.UsageError(
                "--all-prompts cannot be combined with --save-path or --plot-path."
            )
        names = sorted(ndp.PROMPT_PRESETS)
        for name in names:
            print(f"\n[direct][all-prompts] === {name} ===\n", flush=True)
            run_direct_for_preset(
                preset_name=name,
                n_samples=n_samples,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                max_thinking_tokens=max_thinking_tokens,
                llm_path=llm_path,
                save_path=_default_save_path(name),
                plot_path=_default_plot_path(name),
                recompute=recompute,
                batch_size=batch_size,
                enable_thinking=enable_thinking,
                n_llm_targets=n_llm_targets,
                enforcer=enforcer,
                reuse_thinking=reuse_thinking,
                append=append,
            )
        return

    run_direct_for_preset(
        preset_name=prompt_name,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        max_thinking_tokens=max_thinking_tokens,
        llm_path=llm_path,
        save_path=save_path or _default_save_path(prompt_name),
        plot_path=plot_path or _default_plot_path(prompt_name),
        recompute=recompute,
        batch_size=batch_size,
        enable_thinking=enable_thinking,
        n_llm_targets=n_llm_targets,
        enforcer=enforcer,
        reuse_thinking=reuse_thinking,
        append=append,
    )


if __name__ == "__main__":
    main()
