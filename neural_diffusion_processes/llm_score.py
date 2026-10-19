"""The LLM scoring pipeline and its cached artifact.

Produces ``LLMExtraScoreFn``: for each queried x, a distribution over the y-grid
obtained by scoring every candidate value under the model, cached to a pickle so
the (expensive, GPU-bound) scoring runs once per preset.

``LLM_LOG_PROBS_BATCH_SIZE`` lives here because ``_score_y_grid_for_one_target``
reads it. To change it at runtime, assign it on THIS module, not on the
``pipeline`` facade, which only re-exports the value and would not propagate
a write.
"""
from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from neural_diffusion_processes.llm_backend import (
    _compute_prefix_kv,
    _load_causal_lm,
    _patch_qwen3_5_gated_delta_net,
    _render_preserving_thinking,
    _score_suffixes_with_kv,
)
from neural_diffusion_processes.prompts import (
    DEFAULT_FORMAT_NOUN,
    DEFAULT_LLM_PATH,
    DEFAULT_LLM_TYPE,
    DEFAULT_QUERY_VERB,
    DEFAULT_REASONING_PRIME,
    DEFAULT_TASK_NOUN,
    KEY_POINTS_PROMPT_ADDENDUM,
    LLM_BREAK_STR,
    LLM_Y_PREFIX,
    THINKING_SYSTEM_PROMPT,
    _build_chat_prompt,
    _build_x_llm_grid,
    _build_y_grid_step,
    _compose_user_message_general,
    _float_to_str,
    _format_x_value,
)

# Suffix-scoring batch size used by `_score_suffixes_with_kv`: how many
# candidate y-suffixes are scored per forward pass against the shared prefix KV
# cache. Overridable via $LLM_LOG_PROBS_BATCH_SIZE. It is purely a
# memory/throughput knob: lower it if the score pass runs out of GPU memory
# (model weights + prefix KV cache + the padded suffix batch must fit), raise it
# for throughput if there is headroom. Each suffix is scored independently
# against the shared prefix KV cache and each batch pads to its own max_len
# under an attention mask, so the batch width does not change the returned
# log-probs (up to floating-point nondeterminism).
LLM_LOG_PROBS_BATCH_SIZE = int(os.environ.get("LLM_LOG_PROBS_BATCH_SIZE", "32"))


@torch.inference_mode()
def _generate_thinking(
    tokenizer,
    model,
    prefix_tokens: list[int],
    max_thinking_tokens: int = 4096,
    repetition_penalty: float = 1.1,
) -> str:
    """Greedy autoregressive generation from a prefix until ``</think>`` or
    ``max_thinking_tokens``, with a repetition penalty applied to logits.

    Pure ``argmax`` reasoning frequently spirals into self-quotation loops
    (e.g. "Output matches requirements. Thinking block done." repeated). A
    mild repetition penalty (HF-style: divide / multiply logits of already
    generated tokens by ``penalty``) discourages those without breaking
    the chain's coherence.

    The prefix is assumed to end inside an open thinking block (i.e. the
    chat template was applied with ``enable_thinking=True``, so the last
    tokens are ``<think>\\n``). Returns the decoded text the model produces
    (which should normally end with ``</think>``).
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id

    prefix_ids = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
    out = model(input_ids=prefix_ids, use_cache=True)
    cur_kv = out.past_key_values
    cur_logits = out.logits[:, -1, :]  # (1, vocab)
    cur_len = len(prefix_tokens)

    apply_penalty = float(repetition_penalty) != 1.0
    generated_ids: list[int] = []
    for _ in range(max_thinking_tokens):
        if apply_penalty and generated_ids:
            gen_tensor = torch.tensor(
                [generated_ids], dtype=torch.long, device=cur_logits.device,
            )
            score = torch.gather(cur_logits, 1, gen_tensor)
            score = torch.where(
                score < 0, score * repetition_penalty, score / repetition_penalty,
            )
            cur_logits = cur_logits.scatter(1, gen_tensor, score)
        next_id = int(cur_logits[0].argmax().item())
        generated_ids.append(next_id)

        # Run until <|im_end|> (eos) or the budget. We deliberately do NOT
        # stop on `</think>` because the model can emit that token while
        # quoting the system prompt mid-sentence (e.g. "the prompt says:
        # 'After `</think>`...'"), which would prematurely terminate. The
        # natural turn-end is `<|im_end|>`, which is also Qwen's eos.
        if eos_id is not None and next_id == eos_id:
            break

        next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        attn_mask = torch.ones((1, cur_len + 1), dtype=torch.long, device=device)
        out = model(
            input_ids=next_input,
            attention_mask=attn_mask,
            past_key_values=cur_kv,
            use_cache=True,
        )
        cur_kv = out.past_key_values
        cur_logits = out.logits[:, -1, :]
        cur_len += 1

    # Strip the trailing eos token if generation ended on one; the chat
    # template re-appends `<|im_end|>` itself when it renders the
    # assistant turn, so leaving eos in the returned content would
    # produce a doubled close.
    if generated_ids and eos_id is not None and generated_ids[-1] == eos_id:
        generated_ids = generated_ids[:-1]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def _compute_shared_thinking_state(
    tokenizer,
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    language_context: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    num_decimal_x: int,
    num_decimal_y: int,
    x_unit: str,
    y_unit: str,
    max_thinking_tokens: int = 4096,
    key_points_prompt: bool = False,
    x_formatter: Callable[[float], str] | None = None,
    x_constraint_text: str | None = None,
    strip_context: bool = False,
    system_prompt: str | None = None,
    task_noun: str = DEFAULT_TASK_NOUN,
    reasoning_prime: str = DEFAULT_REASONING_PRIME,
    format_noun: str = DEFAULT_FORMAT_NOUN,
) -> tuple[list[dict], str]:
    """Generate the shared thinking text once for all targets.

    Renders the system + general user message via the tokenizer's chat
    template with ``enable_thinking=True, add_generation_prompt=True``
    (which lands in an open ``<think>`` block) and lets the model
    autoregressively generate the body of that block.

    Returns ``(messages_prefix, thinking_text)``:
      * ``messages_prefix`` is a 3-message list ``[system, user, assistant]``
        where the assistant message embeds the generated thinking inline as
        ``"<think>\\n{thinking}\\n\\n"``. Per-target prompts append one more
        ``user`` message and re-render via ``apply_chat_template(...,
        preserve_thinking=True)`` so the past ``<think>`` block survives
        intact (Qwen3.6+).
      * ``thinking_text`` is what the model produced inside the
        ``<think>...</think>`` block, returned separately for logging.
    """
    general_user_message = _compose_user_message_general(
        x_train, y_train, language_context, x_range, y_range,
        num_decimal_x, num_decimal_y, x_unit, y_unit,
        x_formatter=x_formatter,
        x_constraint_text=x_constraint_text,
        strip_context=strip_context,
        task_noun=task_noun,
        reasoning_prime=reasoning_prime,
        format_noun=format_noun,
    )
    _system_prompt = (
        THINKING_SYSTEM_PROMPT if system_prompt is None else system_prompt
    ) + (KEY_POINTS_PROMPT_ADDENDUM if key_points_prompt else "")
    chat_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": _system_prompt},
            {"role": "user", "content": general_user_message},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    # `chat_text` ends with `<|im_start|>assistant\n<think>\n`. Generate
    # everything between there and `<|im_end|>` (or the budget) — both
    # the thinking content AND the post-`</think>` response in one shot.
    prefix_tokens = tokenizer.encode(chat_text)
    raw = _generate_thinking(
        tokenizer, model, prefix_tokens, max_thinking_tokens=max_thinking_tokens
    )
    # If THINKING_TRACE_PATH is set, dump the FULL prompt (system + user +
    # generation prefix) followed by the model's raw generation (the <think>
    # reasoning plus the post-</think> response) so it can be read afterwards.
    _trace_path = os.environ.get("THINKING_TRACE_PATH")
    if _trace_path:
        with open(_trace_path, "w") as _tf:
            _tf.write(
                "########## FULL PROMPT (pre-thinking, chat-templated) ##########\n"
                + chat_text
                + "\n########## MODEL RAW GENERATION (thinking + response) ##########\n"
                + raw
                + "\n########## END ##########\n"
            )
        print(f"[thinking] wrote full prompt+thinking to {_trace_path}", flush=True)

    # Split into thinking content and post-thinking response. If the model
    # never closed the thinking block (budget exhausted mid-thought), force
    # a close and supply a default acknowledgment so the assistant turn is
    # well-formed downstream (Qwen was trained to produce *something* after
    # `</think>`; an empty body is off-distribution).
    if "</think>" in raw:
        thinking_only, response = raw.split("</think>", 1)
        thinking_only = thinking_only.rstrip()
        response = response.strip()
        if not response:
            response = "I have analysed the data. Ready to forecast."
    else:
        thinking_only = raw.rstrip()
        response = "I have analysed the data. Ready to forecast."

    # Embed the parsed pieces inline as the assistant's content. Qwen3.6's
    # `preserve_thinking=True` template renders strings containing
    # `<think>...</think>` verbatim into the multi-turn conversation, so
    # historical reasoning survives downstream calls. Qwen3.5's template
    # (e.g. Qwen3.5-9B) silently strips it; `_render_preserving_thinking`
    # restores it so both versions see the same prompt.
    assistant_content = f"<think>\n{thinking_only}\n</think>\n\n{response}"
    messages_prefix = [
        {"role": "system", "content": _system_prompt},
        {"role": "user", "content": general_user_message},
        {"role": "assistant", "content": assistant_content},
    ]
    # Return the raw generated text for logging — it's what hit the budget,
    # not the trimmed thinking-only slice.
    return messages_prefix, raw


def _build_per_target_prompt_after_shared_thinking(
    messages_prefix: list[dict],
    x_test: float,
    x_unit: str,
    num_decimal_x: int,
    tokenizer,
    x_formatter: Callable[[float], str] | None = None,
    query_verb: str = DEFAULT_QUERY_VERB,
) -> str:
    """Render the canonical multi-turn chat prompt for one target ``x_test``.

    Appends a user message asking for ``x_test`` to ``messages_prefix``
    (which already includes the system, user-general, and
    assistant-with-thinking turns) and re-renders the whole list via
    ``apply_chat_template(preserve_thinking=True, enable_thinking=False,
    add_generation_prompt=True)``. This is the on-distribution form for
    Qwen3.6's shared-thinking pattern. The result ends in an open assistant
    turn; we prime it with ``"{x_test}, "`` so the next token is the y
    value.
    """
    str_x_test = _format_x_value(x_test, num_decimal_x, x_formatter)
    unit = x_unit.lower()
    query_label = "date" if x_formatter is not None else unit
    full_messages = messages_prefix + [
        {"role": "user",
         "content": f"{query_verb} at {query_label}={str_x_test}."},
    ]
    rendered = _render_preserving_thinking(
        tokenizer,
        full_messages,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return rendered + f"{str_x_test}, "


def _score_y_grid_for_one_target(
    non_y_prefix: str,
    y_grid: np.ndarray,
    y_decimal_places: int,
    tokenizer,
    model,
) -> np.ndarray:
    """LLM probability distribution over a y-grid for one target x.

    Returns a 1D numpy array of length ``len(y_grid)`` summing to 1.
    """
    space = " " if LLM_Y_PREFIX.endswith(" ") else ""
    prefix_text = non_y_prefix.rstrip(" ")
    prefix_tokens = tokenizer.encode(prefix_text)

    past_kv, last_logits, plen = _compute_prefix_kv(model, prefix_tokens)

    suffix_tokens_list: list[list[int]] = []
    for y_test in y_grid:
        str_y = _float_to_str(float(y_test), y_decimal_places)
        suffix_text = f"{space}{str_y}{LLM_BREAK_STR}"
        full_tokens = tokenizer.encode(prefix_text + suffix_text)
        suffix_tokens_list.append(full_tokens[len(prefix_tokens):])

    y_logprobs = _score_suffixes_with_kv(
        model,
        tokenizer.pad_token_id,
        past_kv,
        last_logits,
        plen,
        suffix_tokens_list,
        LLM_LOG_PROBS_BATCH_SIZE,
    )

    log_sums = np.array([lp.sum() for lp in y_logprobs])
    # log-sum-exp trick to avoid underflow in exp()
    max_ls = log_sums.max()
    y_probs = np.exp(log_sums - max_ls)
    return (y_probs / y_probs.sum()).astype(np.float64)


def compute_and_save_llm_extra_score_fn(
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_grid: np.ndarray,
    language_context: str,
    save_path: Path,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    x_unit: str = "day",
    y_unit: str = "",
    enable_thinking: bool = True,
    max_thinking_tokens: int = 4096,
    llm_path: str | None = None,
    llm_type: str | None = None,
    prompt_range: tuple[float, float] | None = None,
    llm_x_strategy: str = "integer",
    llm_n_targets: int = 51,
    llm_x_decimals: int = 2,
    llm_y_decimals: int = 2,
    llm_x_data_decimals: int | None = None,
    llm_x_range: tuple[float, float] | None = None,
    key_points_prompt: bool = False,
    x_formatter: Callable[[float], str] | None = None,
    x_constraint_text: str | None = None,
    strip_context: bool = False,
    system_prompt: str | None = None,
    task_noun: str = DEFAULT_TASK_NOUN,
    reasoning_prime: str = DEFAULT_REASONING_PRIME,
    query_verb: str = DEFAULT_QUERY_VERB,
    format_noun: str = DEFAULT_FORMAT_NOUN,
) -> LLMExtraScoreFn:
    """Score each x-grid point against a y-discretization with a HF causal LM,
    wrap the result in a picklable :class:`LLMExtraScoreFn`, and write it to
    disk.

    Loads the model + tokenizer (applying the Qwen 3.5 KV-cache monkey-patches
    first), builds a prompt (context sorted by x) for each target x, and scores
    every y-grid bin using a per-prompt KV cache. The resulting
    ``LLMExtraScoreFn`` instance is both returned and pickled to ``save_path``.
    The runtime knobs (``likelihood_variance`` / ``gamma``) on the returned
    instance are class defaults; the caller should set them explicitly if
    needed.

    Parameters
    ----------
    enable_thinking
        If True, use Qwen's thinking mode: the model generates a
        ``<think>...</think>`` chain of reasoning **once** about the
        function as a whole, then the same thinking is reused as a prefix
        for every per-target y-grid scoring (each target only adds a
        short multi-turn ``"{query_verb} at {unit}={x_test}."`` user query,
        or ``"... at date={x_test}."`` when an x-formatter is set). This
        costs roughly one extra thinking generation per recompute (not per
        target).
    max_thinking_tokens
        Cap on tokens generated for the thinking block and the short
        acknowledgement after it.
    """
    _patch_qwen3_5_gated_delta_net()

    path = llm_path if llm_path is not None else DEFAULT_LLM_PATH
    ltype = llm_type if llm_type is not None else DEFAULT_LLM_TYPE
    print(f"[llm_score] loading model {path!r} (type={ltype!r})", flush=True)
    print(
        f"[llm_score] enable_thinking={enable_thinking} "
        f"max_thinking_tokens={max_thinking_tokens}",
        flush=True,
    )
    model, tokenizer = _load_causal_lm(path, ltype)

    x_context_np = np.asarray(x_context, dtype=np.float64)
    y_context_np = np.asarray(y_context, dtype=np.float64)
    x_grid_np = np.asarray(x_grid, dtype=np.float64)

    # Build the LLM x-scoring grid + decide x rendering precision. The grid
    # range is `llm_x_range` when set (decoupling LLM scoring from the NDP
    # x_range — e.g. NDP on the full sequence, LLM on a sub-range), otherwise
    # `x_range`. Note: the prompt continues to advertise `x_range` so the LLM
    # knows the full data span; only the queried subset shrinks.
    x_llm, x_dec = _build_x_llm_grid(
        x_range=x_range,
        n_grid=len(x_grid_np),
        llm_x_strategy=llm_x_strategy,
        llm_n_targets=llm_n_targets,
        llm_x_decimals=llm_x_decimals,
        llm_x_range=llm_x_range,
        x_grid=x_grid_np,
    )

    # `x_dec` governs how TARGET x values are rendered. Under
    # llm_x_strategy='integer' it is forced to 0, and it also used to
    # govern the observed-data lines -- so integer targets implied integer
    # context, rounding 0.13/2.71 km to "0"/"3" and letting two levels
    # collide. `llm_x_data_decimals` decouples them: queries can sit on
    # whole units while the context keeps its real precision.
    # Default None reproduces the previous behaviour exactly.
    # (Thinking path only. The non-thinking `_build_chat_prompt` branch
    # renders data and target with one precision and ignores this.)
    x_dec_data = x_dec if llm_x_data_decimals is None else int(llm_x_data_decimals)

    # Size the LLM y-grid and the y-data print precision off the inherent
    # precision of `y_context`, not the `y_range` magnitude — e.g. context
    # values (2.5, 2.2) inherently 1-dp give a 0.1-step grid, even with a
    # wide y_range like (-3, 7). The precision is taken as-is from
    # `llm_y_decimals`; no further cap is applied here.
    y_dec = int(llm_y_decimals)
    y_grid = _build_y_grid_step(y_range, y_dec)
    # `prompt_y_range` is what the LLM sees in the "Values represent ..., real
    # numbers in [a, b]" line of the prompt. The y-grid (and therefore the
    # scored bins) is still controlled by `y_range`. Keeping these decoupled
    # lets a preset widen `y_range` for NDP/flow tail mass without misleading
    # the LLM about plausible value bounds.
    prompt_y_range = prompt_range if prompt_range is not None else y_range
    print(
        f"[llm_score] y_grid: y_range={y_range} "
        f"prompt_y_range={prompt_y_range} "
        f"y_dec={y_dec} (from preset.llm_y_decimals) "
        f"step={10.0 ** (-y_dec) if y_dec > 0 else 1.0:g} "
        f"n_y_bins={len(y_grid)}",
        flush=True,
    )

    # If thinking is enabled, do the (expensive) thinking generation ONCE
    # for all targets - the resulting `messages_prefix` is the multi-turn
    # message list (system + user + assistant-with-thinking) that every
    # per-target prompt re-renders via apply_chat_template.
    messages_prefix: list[dict] | None = None
    if enable_thinking:
        print(
            f"[llm_score] generating shared thinking "
            f"(max_tokens={max_thinking_tokens})...",
            flush=True,
        )
        messages_prefix, thinking_text = _compute_shared_thinking_state(
            tokenizer=tokenizer,
            model=model,
            x_train=x_context_np,
            y_train=y_context_np,
            language_context=language_context,
            x_range=x_range,
            y_range=prompt_y_range,
            num_decimal_x=x_dec_data,
            num_decimal_y=y_dec,
            x_unit=x_unit,
            y_unit=y_unit,
            max_thinking_tokens=max_thinking_tokens,
            key_points_prompt=key_points_prompt,
            x_formatter=x_formatter,
            x_constraint_text=x_constraint_text,
            strip_context=strip_context,
            system_prompt=system_prompt,
            task_noun=task_noun,
            reasoning_prime=reasoning_prime,
            format_noun=format_noun,
        )
        n_think_tokens = len(
            tokenizer.encode(thinking_text, add_special_tokens=False)
        )
        truncated = n_think_tokens >= max_thinking_tokens
        print(
            f"[llm_score] shared thinking generated "
            f"({n_think_tokens}/{max_thinking_tokens} tokens"
            f"{' — TRUNCATED at budget' if truncated else ''}; "
            f"full text shown inside the example prompt for target 0 below)",
            flush=True,
        )

    n_grid = len(x_grid_np)
    n_y_bins = len(y_grid)
    uniform_row = np.full(n_y_bins, 1.0 / n_y_bins, dtype=np.float64)

    # Score the LLM at each target x. Output of the loop is `llm_probs` of
    # shape (len(x_llm), n_y_bins) — each row is p(y | x_llm[k], context, text).
    llm_probs = np.empty((len(x_llm), n_y_bins), dtype=np.float64)
    for k, x_int in enumerate(x_llm):
        x_test = float(x_int)
        if enable_thinking:
            assert messages_prefix is not None
            non_y_prefix = _build_per_target_prompt_after_shared_thinking(
                messages_prefix=messages_prefix,
                x_test=x_test,
                x_unit=x_unit,
                num_decimal_x=x_dec,
                tokenizer=tokenizer,
                x_formatter=x_formatter,
                query_verb=query_verb,
            )
        else:
            non_y_prefix = _build_chat_prompt(
                tokenizer=tokenizer,
                x_train=x_context_np,
                y_train=y_context_np,
                x_test=x_test,
                language_context=language_context,
                x_range=x_range,
                y_range=prompt_y_range,
                num_decimal_x=x_dec,
                num_decimal_y=y_dec,
                x_unit=x_unit,
                y_unit=y_unit,
                x_formatter=x_formatter,
                x_constraint_text=x_constraint_text,
                strip_context=strip_context,
            )

        non_y_prefix = non_y_prefix.rstrip(" ")

        if k == 0:
            print(
                f"[llm_score] example LLM prompt "
                f"(x_test={_format_x_value(x_test, x_dec, x_formatter)}):\n"
                f"--- prompt ---\n{non_y_prefix}\n"
                f"--- end ---",
                flush=True,
            )

        llm_probs[k] = _score_y_grid_for_one_target(
            non_y_prefix, y_grid, y_dec, tokenizer, model
        )
        print(
            f"[llm_score] llm target {k + 1}/{len(x_llm)} "
            f"x={_format_x_value(x_test, x_dec, x_formatter)}",
            flush=True,
        )

    # Layout-agnostic compute: persist only the raw LLM outputs
    # (`llm_probs`, `x_llm`). The consumer chooses sparse vs dense at load
    # time via `LLMExtraScoreFn.from_file(..., sparse=...)`. The constructor
    # below builds `prob_xys` at the default (dense) layout for callers
    # that read the field directly, but it is fully derivable from the
    # raw fields and never authoritative.
    fn = LLMExtraScoreFn(
        x_context=x_context_np,
        y_context=y_context_np,
        x_grid=x_grid_np,
        y_grid=y_grid,
        llm_probs=llm_probs,
        x_llm=x_llm,
        language_context=language_context,
        x_range=x_range,
        y_range=y_range,
        x_unit=x_unit,
        y_unit=y_unit,
    )
    fn.save(save_path)

    # Release the LLM's GPU memory before this process moves on to NDP
    # sampling (which uses JAX on the same device). Python's refcount alone
    # would free `model` once we exit the function, but PyTorch's CUDA
    # caching allocator holds the reservation past the del; an explicit
    # `empty_cache()` returns it to the driver. Without this the next JAX
    # op would compete with torch's reserved-but-idle pool.
    import gc as _gc
    del model, tokenizer
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return fn


def _build_prob_xys(
    *,
    llm_probs: np.ndarray,
    x_llm: np.ndarray,
    x_grid: np.ndarray,
    n_y_bins: int,
    sparse: bool,
    verbose: bool = False,
) -> np.ndarray:
    """Build the NDP-grid-aligned `prob_xys` from `llm_probs`. Two layouts:

      - sparse (the default): each scored x's probs are placed at its single
        closest NDP grid index; every other NDP row stays uniform.
      - dense: each NDP grid row copies the LLM row of the nearest scored x,
        so every row carries a real distribution.
    """
    n_grid = len(x_grid)
    uniform_row = np.full(n_y_bins, 1.0 / n_y_bins, dtype=np.float64)
    prob_xys = np.tile(uniform_row, (n_grid, 1))
    if sparse:
        for k, x_k in enumerate(x_llm):
            i = int(np.argmin(np.abs(x_grid - float(x_k))))
            prob_xys[i] = llm_probs[k]
        if verbose:
            print(
                f"[llm_score] sparse fill: scored {len(x_llm)} "
                f"LLM x's, each placed at its closest NDP grid index; "
                f"remaining {n_grid - len(x_llm)} NDP rows kept uniform",
                flush=True,
            )
    else:
        nearest = np.argmin(
            np.abs(x_grid[:, None] - np.asarray(x_llm)[None, :].astype(np.float64)),
            axis=1,
        )
        prob_xys = llm_probs[nearest]
        if verbose:
            print(
                f"[llm_score] dense broadcast: filled all {n_grid} "
                f"NDP grid rows from {len(x_llm)} LLM x's via nearest-neighbor",
                flush=True,
            )
    return prob_xys


def llm_extra_log_prob(
    pred_output: Tensor,
    pred_output_var: Tensor,
    y_grid: Tensor,
    prob_xys: Tensor,
    likelihood_variance: float = 0.01,
    gamma: float = 1.0,
) -> Tensor:
    """
    Convolves a Gaussian (centred on each predicted target value with
    variance ``likelihood_variance + pred_output_var``) against the
    discrete LLM distribution over the y-grid bins, then returns the sum
    of log probabilities across the target points.

    Parameters
    ----------
    pred_output
        Shape ``(n_total, 1)``. Must be the concatenation
        ``[y_context; y_grid]`` - the handler slices the trailing
        ``n_target = prob_xys.shape[0]`` rows.
    pred_output_var
        Scalar tensor with the diffusion-time variance of ``pred_output``.
    y_grid
        Uniformly spaced y-discretization of shape ``(n_y_grid,)``.
    prob_xys
        LLM probabilities of shape ``(n_target, n_y_grid)``. Each row must
        sum to one (or close to it).
    likelihood_variance
        Floor variance to avoid blow-up near `t = 1`. Should be at least
        a few bin widths squared - if it's smaller than the y-grid spacing
        the gradient gets dominated by spurious local maxima at every
        individual y-bin. Scale it with the y-grid spacing set by
        ``llm_y_decimals`` (the default 0.01 suits the default 0.01 spacing).
    gamma
        Linear scaling on the returned log-prob.
    """
    n_target = prob_xys.shape[0]
    pred_target = pred_output[-n_target:, 0]
    var = likelihood_variance + pred_output_var
    # Bins are CENTERED on `y_grid` values (not left-aligned). The LLM
    # scores each y_grid value as a discrete token, so the probability mass
    # at `y_grid[j]` represents the centred bin
    # ``[y_grid[j] - bin_width/2, y_grid[j] + bin_width/2]``. Left-aligning
    # the bins shifts everything by half a bin and produces wrong-sign
    # gradients at sharp peaks (eg the gradient at pred=y_grid[0]=0 was
    # +4.74 instead of zero - it kept pulling pred toward the left-bin
    # centre at 0.05 rather than the LLM mode at 0).
    bin_width = y_grid[1] - y_grid[0]
    half = 0.5 * bin_width
    bin_edges = torch.cat([y_grid - half, (y_grid[-1:] + half)])
    z = (bin_edges[None, :] - pred_target[:, None]) / torch.sqrt(var)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    gaussian_term = cdf[:, 1:] - cdf[:, :-1]  # (n_target, n_y_grid)
    q_tilde = (gaussian_term * prob_xys).sum(dim=-1).clamp_min(1e-30)
    return gamma * torch.log(q_tilde).sum()


class LLMExtraScoreFn:
    """Picklable callable that evaluates the LLM extra log-prob term.

    Holds the precomputed ``prob_xys`` plus enough provenance metadata
    (``x_context``, ``y_context``, ``x_grid``, ``y_grid``, ``language_context``,
    ``x_range``, ``y_range``) to plot and reproduce the demo.

    Calling the instance with ``(pred_output, pred_output_var)`` returns a
    scalar log-prob whose ``torch.autograd.grad`` w.r.t. ``pred_output`` is
    the additive score term for text-conditioned diffusion.

    Pickling stores **only** the data fields above. The runtime knobs
    ``likelihood_variance`` and ``gamma`` are not persisted - they only
    affect ``llm_extra_log_prob`` and can be retuned at any time without
    recomputing the LLM probabilities. Set them at construction or via
    :meth:`from_file` kwargs.

    ``pred_output`` MUST be the concatenation ``[y_context; y_grid]`` of
    shape ``(n_ctx + n_target, 1)`` - the handler slices the trailing
    ``n_target = prob_xys.shape[0]`` rows.
    """

    # Persisted state. The layout-applied `prob_xys` is intentionally NOT
    # in here: it's a derived view of `llm_probs` + `x_llm` at a chosen
    # sparse/dense layout, rebuilt in __init__ at load time so the on-disk
    # pickle is layout-agnostic.
    _DATA_FIELDS = (
        "x_context",
        "y_context",
        "x_grid",
        "y_grid",
        "language_context",
        "x_range",
        "y_range",
        "x_unit",
        "y_unit",
        "llm_probs",
        "x_llm",
    )

    def __init__(
        self,
        *,
        x_context: np.ndarray,
        y_context: np.ndarray,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        llm_probs: np.ndarray,
        x_llm: np.ndarray,
        language_context: str,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        x_unit: str = "day",
        y_unit: str = "",
        likelihood_variance: float = 0.01,
        gamma: float = 1.0,
        sparse: bool = True,
    ) -> None:
        self.x_context = np.asarray(x_context, dtype=np.float64)
        self.y_context = np.asarray(y_context, dtype=np.float64)
        self.x_grid = np.asarray(x_grid, dtype=np.float64)
        self.y_grid = np.asarray(y_grid, dtype=np.float64).reshape(-1)
        self.llm_probs = np.asarray(llm_probs, dtype=np.float64)
        self.x_llm = np.asarray(x_llm, dtype=np.float64)
        self.language_context = str(language_context)
        self.x_range = (float(x_range[0]), float(x_range[1]))
        self.y_range = (float(y_range[0]), float(y_range[1]))
        self.x_unit = str(x_unit)
        self.y_unit = str(y_unit)
        self.likelihood_variance = float(likelihood_variance)
        self.gamma = float(gamma)
        # `prob_xys` is the layout-applied view of llm_probs the diffusion
        # sampler actually consumes. Built here from raw probs at the
        # requested layout so callers don't need to invoke `_build_prob_xys`
        # themselves.
        self.prob_xys = _build_prob_xys(
            llm_probs=self.llm_probs,
            x_llm=self.x_llm,
            x_grid=self.x_grid,
            n_y_bins=int(self.y_grid.shape[0]),
            sparse=bool(sparse),
            verbose=False,
        )
        # Lazy torch cache: (y_grid_t, prob_xys_t, dtype, device).
        self._cache: tuple[Tensor, Tensor, torch.dtype, torch.device] | None = None

    def __call__(self, pred_output: Tensor, pred_output_var: Tensor) -> Tensor:
        if (
            self._cache is None
            or self._cache[2] != pred_output.dtype
            or self._cache[3] != pred_output.device
        ):
            self._cache = (
                torch.as_tensor(self.y_grid, dtype=pred_output.dtype, device=pred_output.device),
                torch.as_tensor(self.prob_xys, dtype=pred_output.dtype, device=pred_output.device),
                pred_output.dtype,
                pred_output.device,
            )
        y_grid_t, prob_xys_t = self._cache[0], self._cache[1]
        return llm_extra_log_prob(
            pred_output,
            pred_output_var,
            y_grid_t,
            prob_xys_t,
            likelihood_variance=self.likelihood_variance,
            gamma=self.gamma,
        )

    # --- pickle support ----------------------------------------------------

    def __getstate__(self) -> dict:
        return {k: getattr(self, k, None) for k in self._DATA_FIELDS}

    def __setstate__(self, state: dict) -> None:
        # Reconstruct via __init__ so the runtime knobs and lazy cache get
        # consistent default values. Loaders that want non-default knobs
        # should use `from_file(..., likelihood_variance=..., gamma=...)`.
        self.__init__(**state)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[llm_score] wrote {path}", flush=True)

    @classmethod
    def from_file(
        cls,
        path: Path,
        *,
        likelihood_variance: float = 0.01,
        gamma: float = 1.0,
        sparse: bool = True,
    ) -> "LLMExtraScoreFn":
        """Load a pickled instance and apply the runtime knobs.

        `prob_xys` is rebuilt at the requested `sparse` layout from the
        persisted `llm_probs` + `x_llm`.
        """
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        obj.likelihood_variance = float(likelihood_variance)
        obj.gamma = float(gamma)
        obj.prob_xys = _build_prob_xys(
            llm_probs=obj.llm_probs,
            x_llm=obj.x_llm,
            x_grid=obj.x_grid,
            n_y_bins=int(obj.y_grid.shape[0]),
            sparse=bool(sparse),
            verbose=False,
        )
        obj._cache = None
        return obj
