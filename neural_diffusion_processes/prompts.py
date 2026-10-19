"""Prompt construction: model paths, system prompts, number formatting, user messages.

Everything that decides what text the LLM sees, and how x/y values are rendered
into it. No torch, no jax -- importing this does not load a model.
"""
from __future__ import annotations

import decimal
from typing import Callable

import numpy as np

# LLM defaults: change via `--qwen-version` to flip between 3.5/3.6, or
# `--llm-path` / `--llm-type` for an arbitrary checkpoint or debug-size model.
QWEN_VERSION_PATHS = {
    "3.5": "Qwen/Qwen3.5-9B",
    "3.6": "Qwen/Qwen3.6-27B",
}


DEFAULT_QWEN_VERSION = "3.6"


DEFAULT_LLM_PATH = QWEN_VERSION_PATHS[DEFAULT_QWEN_VERSION]


DEFAULT_LLM_TYPE = "qwen"


# Per-token-position prompt formatting.
LLM_BREAK_STR = "\n"


LLM_X_PREFIX = ""


LLM_Y_PREFIX = ", "


LLM_Y_PRECISION = 3


LLM_X_PRECISION = 3


# Minimal system prompt used in NO-THINKING mode. Activates the
# "helpful assistant" role and gives the format constraint a stable home
# at maximum context distance from the answer position. The reasoning
# checklist / thinking workflow is OMITTED here because the chat template
# pre-empties the thinking block (`enable_thinking=False`) and the model
# can't actually reason - telling it to think when its thinking has been
# zeroed out is just contradictory.
BASE_SYSTEM_PROMPT = (
    "You are a time-series forecasting assistant.\n"
    "\n"
    "For each forecasting task you are given a description of the time "
    "series, some observed (x, value) pairs, and a target x. Output ONLY "
    "the numeric forecast in the format `x, value`. No commentary, no "
    "units, no explanation.\n"
)


# System prompt used in thinking mode. The thinking block is generated
# ONCE per recompute (not per target x): Qwen thinks about the whole
# function once, then answers many per-x queries reusing the same reasoning.
THINKING_SYSTEM_PROMPT = (
    "You are a time-series forecasting assistant.\n"
    "\n"
    "Workflow:\n"
    "  - You are given a description of the time series and some observed "
    "(x, value) pairs.\n"
    "  - Reason about the function ONCE at the start.\n"
    "  - You will then be asked to forecast specific x values, one at a "
    "time. Reuse your earlier reasoning for each query.\n"
    "\n"
    "Keep your reasoning to roughly 500 words. You don't need to be "
    "exhaustive - sketch the pattern, sanity-check briefly, and stop. "
    "If you find yourself re-deriving the same conclusion, enumerating "
    "cases, or second-guessing a value you already wrote down, stop "
    "and finish.\n"
    "\n"
    "Internally reason step by step. Briefly cover these points:\n"
    "  1. The real-world meaning of the values: what units, what is a "
    "typical magnitude, what is a plausible upper / lower bound.\n"
    "  2. The structure of the series: trend (rising / falling), "
    "seasonality (daily / monthly / yearly cycle), regime changes "
    "(one-time events like a bankruptcy or holiday), or none of the above.\n"
    "  3. The general shape of the function across the full x range. For "
    "seasonal data work out the cycle (for example, if month 1 is "
    "February then month 30 is (30 - 1) mod 12 = 5 -> July, which is "
    "dry in California).\n"
    "  4. A plan for how to map any queried x to its forecast value.\n"
    "\n"
    "Once you have finished reasoning, your visible response is ONLY a "
    "single short acknowledgment sentence such as \"I have analysed the "
    "data.\" or \"Ready to forecast.\" Do not repeat, summarize, or "
    "restate your reasoning in the visible response. Then await the "
    "per-x queries. For each query, output ONLY the numeric forecast in "
    "the format `x, value`. No commentary, no units, no explanation."
)


# --- domain-neutral prompt scaffolding ------------------------------------
# The wording above is time-series specific: it calls itself a "time-series
# forecasting assistant", asks the model to reason about "seasonality (daily /
# monthly / yearly cycle)", and works a calendar example. That is wrong for a
# preset whose x axis is not time -- e.g. a profile indexed by height, where
# roughly a third of the reasoning checklist is inapplicable and the
# seasonality prompt actively misleads.
#
# These are OPT-IN. The DEFAULT_* values below are the time-series wording and
# are the defaults of the `task_noun` / `reasoning_prime` / `query_verb` /
# `format_noun` keyword arguments of the prompt builders; the GENERIC_* values
# are the domain-neutral alternatives a caller can pass instead. Nothing in
# this package passes them, so every prompt uses the DEFAULT_* wording.
DEFAULT_TASK_NOUN = "time series forecasting task"


DEFAULT_REASONING_PRIME = (
    "Think carefully about the function: its general shape across the "
    "full range, any seasonality, trends, regime changes, or events. "
)


DEFAULT_QUERY_VERB = "Forecast"


DEFAULT_FORMAT_NOUN = "forecast"   # "Each {noun} you give will be ..."


GENERIC_TASK_NOUN = "function-fitting task"


GENERIC_REASONING_PRIME = (
    "Think carefully about the function: its general shape across the "
    "full range, any trends, turning points, or changes of regime. "
)


GENERIC_QUERY_VERB = "Predict"


GENERIC_FORMAT_NOUN = "prediction"


# Non-thinking twin of BASE_SYSTEM_PROMPT: same instructions, no temporal
# framing. Not used by default -- llm_direct's --no-think path sends
# BASE_SYSTEM_PROMPT unless `run_direct_for_preset(base_system_prompt=...)`
# is given this.
GENERIC_BASE_SYSTEM_PROMPT = (
    "You are a function-fitting assistant.\n"
    "\n"
    "For each task you are given a description of the function, some "
    "observed (x, value) pairs, and a target x. Output ONLY the numeric "
    "prediction in the format `x, value`. No commentary, no "
    "units, no explanation.\n"
)


# Structural twin of THINKING_SYSTEM_PROMPT with the temporal framing removed.
# The operational instructions (500-word budget, acknowledge-then-await, output
# format) are deliberately preserved verbatim -- only the domain framing and
# the seasonality checklist item change.
GENERIC_THINKING_SYSTEM_PROMPT = (
    "You are a function-fitting assistant.\n"
    "\n"
    "Workflow:\n"
    "  - You are given a description of the function and some observed "
    "(x, value) pairs.\n"
    "  - Reason about the function ONCE at the start.\n"
    "  - You will then be asked to predict specific x values, one at a "
    "time. Reuse your earlier reasoning for each query.\n"
    "\n"
    "Keep your reasoning to roughly 500 words. You don't need to be "
    "exhaustive - sketch the pattern, sanity-check briefly, and stop. "
    "If you find yourself re-deriving the same conclusion, enumerating "
    "cases, or second-guessing a value you already wrote down, stop "
    "and finish.\n"
    "\n"
    "Internally reason step by step. Briefly cover these points:\n"
    "  1. The real-world meaning of the values: what units, what is a "
    "typical magnitude, what is a plausible upper / lower bound.\n"
    "  2. The structure of the function: whether it rises or falls with x, "
    "where its gradient changes, any turning points or distinct regimes, "
    "and whether the domain implies limits the values cannot cross.\n"
    "  3. The general shape of the function across the full x range, "
    "including its behaviour between and beyond the observed points.\n"
    "  4. A plan for how to map any queried x to its predicted value.\n"
    "\n"
    "Once you have finished reasoning, your visible response is ONLY a "
    "single short acknowledgment sentence such as \"I have analysed the "
    "data.\" or \"Ready to predict.\" Do not repeat, summarize, or "
    "restate your reasoning in the visible response. Then await the "
    "per-x queries. For each query, output ONLY the numeric prediction in "
    "the format `x, value`. No commentary, no units, no explanation."
)


# Optional addendum to THINKING_SYSTEM_PROMPT enabled by --key-points-prompt.
# Forces the model to identify key points (peaks, troughs, changepoints,
# anchor observations, etc.) of the function and pin numerical (x, y)
# estimates for each. The intent is that committing to explicit anchors in
# the thinking biases per-x decoding away from smooth-interpolation-between-
# observations and toward the structural shape the model just described.
KEY_POINTS_PROMPT_ADDENDUM = (
    "\n\nWhile reasoning, if the function has any structurally important "
    "points - whatever they happen to be for this data - note each one as a "
    "concrete `(x, estimated_value)` pair. Always write both coordinates: a "
    "bare x without an estimated y is not useful. Do not invent points the "
    "data does not warrant; do not force a fixed list."
)


# Decimal context for `_float_to_str` - matches the upstream `floats_to_str`
# behaviour from `llm_processes.helpers` so the LLM sees identical
# tokenizations.
_LLM_DECIMAL_CTX = decimal.Context(prec=20, rounding=decimal.ROUND_HALF_EVEN)


def _float_to_str(f: float, num_decimal: int) -> str:
    d = _LLM_DECIMAL_CTX.create_decimal(repr(float(f)))
    if num_decimal is not None:
        d = round(d, num_decimal)
    return format(d, "f")


def _format_x_value(
    x: float,
    num_decimal_x: int,
    x_formatter: Callable[[float], str] | None = None,
) -> str:
    if x_formatter is not None:
        return x_formatter(float(x))
    return _float_to_str(float(x), num_decimal_x)


def _build_x_llm_grid(
    *,
    x_range: tuple[float, float],
    n_grid: int,
    llm_x_strategy: str,
    llm_n_targets: int,
    llm_x_decimals: int,
    llm_x_range: tuple[float, float] | None = None,
    x_grid: np.ndarray | None = None,
    n_targets_override: int | None = None,
) -> tuple[np.ndarray, int]:
    """Single source of truth for the LLM target grid ``(x_llm, x_decimals)``.

    The grid spans ``llm_x_range`` when set (decoupling LLM scoring from the NDP
    ``x_range`` -- e.g. NDP over the full sequence, LLM over a sub-range),
    otherwise ``x_range``. The prompt still advertises ``x_range`` so the model
    knows the full data span; only the queried subset shrinks.

    ``n_targets_override`` replaces ``llm_n_targets`` and forces
    ``linspace_subset`` on integer-strategy presets, whose arange length is
    otherwise fixed by the range. Used to study T effects without editing presets.

    This was previously written out three times -- in the per-x scorer, in
    ``_run_for_prompt`` (for its target count), and in ``llm_direct``
    -- and the third copy had dropped ``llm_x_range`` support, so LLM-Direct
    queried a different grid than the scorer for any preset that set it.
    """
    x_grid_np = (
        np.linspace(x_range[0], x_range[1], n_grid, dtype=np.float64)
        if x_grid is None else np.asarray(x_grid, dtype=np.float64)
    )
    llm_grid_range = llm_x_range if llm_x_range is not None else x_range

    strategy = llm_x_strategy
    if n_targets_override is not None and strategy == "integer":
        strategy = "linspace_subset"

    if strategy == "integer":
        if not (float(llm_grid_range[0]).is_integer()
                and float(llm_grid_range[1]).is_integer()):
            raise ValueError(
                f"llm_x_strategy='integer' requires integer llm-grid bounds; "
                f"got {llm_grid_range}."
            )
        x_llm = np.arange(
            int(llm_grid_range[0]), int(llm_grid_range[1]) + 1, dtype=np.int64,
        ).astype(np.float64)
        return x_llm, 0

    if strategy == "linspace_subset":
        llm_dense = (
            x_grid_np if llm_x_range is None
            else np.linspace(llm_grid_range[0], llm_grid_range[1], len(x_grid_np))
        )
        n_dense = len(llm_dense)
        n_targets = (
            int(n_targets_override) if n_targets_override is not None
            else int(llm_n_targets)
        )
        if n_targets > n_dense:
            raise ValueError(
                f"llm_n_targets ({n_targets}) must not exceed n_grid ({n_dense})."
            )
        idx = np.unique(
            np.linspace(0, n_dense - 1, n_targets).round().astype(np.int64)
        )
        return llm_dense[idx], int(llm_x_decimals)

    raise ValueError(
        f"Unknown llm_x_strategy={llm_x_strategy!r}. "
        f"Expected 'integer' or 'linspace_subset'."
    )


def _build_y_grid_step(
    y_range: tuple[float, float], num_decimal_y: int
) -> np.ndarray:
    step = 10.0 ** (-num_decimal_y) if num_decimal_y > 0 else 1.0
    return np.arange(y_range[0], y_range[1] + step, step, dtype=np.float64)


def _format_x_constraint_line(
    units: str,
    x_range: tuple[float, float],
    num_decimal_x: int,
) -> str:
    """Build the prompt line declaring the numeric form of the x-axis.

    When ``num_decimal_x == 0`` the LLM x-grid lands on integers, so the
    prompt says "integers from a to b". Otherwise it falls back to a
    "real numbers in [a, b]" phrasing using the same decimal precision
    that ``_float_to_str`` will use when rendering individual x-values,
    so prompt and data stay consistent.
    """
    cap = units.capitalize()
    if num_decimal_x == 0:
        return (
            f"- {cap} are integers from {int(x_range[0])} "
            f"to {int(x_range[1])}.\n"
        )
    lo = _float_to_str(x_range[0], num_decimal_x)
    hi = _float_to_str(x_range[1], num_decimal_x)
    return f"- {cap} are real numbers in [{lo}, {hi}].\n"


def _compose_user_message(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: float,
    language_context: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    num_decimal_x: int,
    num_decimal_y: int,
    x_unit: str,
    y_unit: str,
    x_formatter: Callable[[float], str] | None = None,
    x_constraint_text: str | None = None,
    strip_context: bool = False,
) -> tuple[str, str]:
    """Build the user-message text for one target ``x_test``.

    Returns ``(user_message, str_x_test)``. The user message contains the
    task framing, language context, constraints, observed data, and a
    "predict the value at {unit} {x_test}" instruction. ``str_x_test`` is
    returned separately because callers need it to prime the assistant
    turn with ``"{x_test}, "`` after the chat template wrapper.
    """
    x_arr = np.asarray(x_train, dtype=np.float64)
    sort_indices = np.argsort(x_arr)
    data_lines = "\n".join(
        f"{_format_x_value(float(x_arr[i]), num_decimal_x, x_formatter)}, "
        f"{_float_to_str(float(y_train[i]), num_decimal_y)}"
        for i in sort_indices
    )
    str_x_test = _format_x_value(x_test, num_decimal_x, x_formatter)

    unit = x_unit.lower()
    units = unit + "s"  # naive plural; "day" -> "days", "month" -> "months"
    yu = y_unit.strip()
    # Treat the placeholder y_unit "value" the same as no unit, so we don't
    # emit the malformed "Values represent value, real numbers in [...]".
    yu_meaningful = yu and yu.lower() != "value"
    value_clause = (
        f"- Values represent {yu}, real numbers in "
        f"[{y_range[0]:.1f}, {y_range[1]:.1f}].\n"
        if yu_meaningful
        else f"- Values are real numbers in [{y_range[0]:.1f}, {y_range[1]:.1f}].\n"
    )
    if x_constraint_text is not None:
        x_clause = f"- {x_constraint_text}\n"
    else:
        x_clause = _format_x_constraint_line(units, x_range, num_decimal_x)
    y_col = yu if yu else "value"
    if x_formatter is not None:
        data_header = f"date, {y_col}"
        predict_clause = f"Predict the value at {str_x_test}. "
        reply_format = "`date, value`"
    else:
        data_header = f"{unit}, {y_col}"
        predict_clause = f"Predict the value at {unit} {str_x_test}. "
        reply_format = f"`{unit}, value`"
    preamble = (
        "I have a time series forecasting task for you.\n"
        if strip_context
        else (
            "I have a time series forecasting task for you. Here is some "
            "context about the task. Make sure to factor in any background "
            "knowledge, and satisfy any constraints:\n"
            f"{language_context.strip()}\n"
            "\n"
            "Constraints:\n"
            f"{value_clause}"
            f"{x_clause}"
        )
    )
    user_message = (
        f"{preamble}"
        "\n"
        f"Observed data ({data_header}):\n"
        f"{data_lines}\n"
        "\n"
        f"{predict_clause}"
        f"Reply in the format {reply_format} with no additional text."
    )
    return user_message, str_x_test


def _compose_user_message_general(
    x_train: np.ndarray,
    y_train: np.ndarray,
    language_context: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    num_decimal_x: int,
    num_decimal_y: int,
    x_unit: str,
    y_unit: str,
    x_formatter: Callable[[float], str] | None = None,
    x_constraint_text: str | None = None,
    strip_context: bool = False,
    task_noun: str = DEFAULT_TASK_NOUN,
    reasoning_prime: str = DEFAULT_REASONING_PRIME,
    format_noun: str = DEFAULT_FORMAT_NOUN,
) -> str:
    """Build a target-free user message for shared-thinking mode.

    Stripped to bare DATA (domain context, numerical bounds, observed
    pairs) PLUS one in-line reasoning prime ("think carefully about the
    function...") intended to get the thinking-mode model to engage
    with seasonal / multi-hop structure. Workflow / format instructions
    are NOT repeated here - they're in :data:`THINKING_SYSTEM_PROMPT`.

    ``strip_context=True`` drops the domain description and the whole
    "Constraints" block, leaving only the observations and the
    reasoning prime (see `PromptPreset.strip_context`).
    """
    x_arr = np.asarray(x_train, dtype=np.float64)
    sort_indices = np.argsort(x_arr)
    data_lines = "\n".join(
        f"{_format_x_value(float(x_arr[i]), num_decimal_x, x_formatter)}, "
        f"{_float_to_str(float(y_train[i]), num_decimal_y)}"
        for i in sort_indices
    )

    unit = x_unit.lower()
    units = unit + "s"
    yu = y_unit.strip()
    # Treat the placeholder y_unit "value" the same as no unit, so we don't
    # emit the malformed "Values represent value, real numbers in [...]".
    yu_meaningful = yu and yu.lower() != "value"
    value_clause = (
        f"- Values represent {yu}, real numbers in "
        f"[{y_range[0]:.1f}, {y_range[1]:.1f}].\n"
        if yu_meaningful
        else f"- Values are real numbers in [{y_range[0]:.1f}, {y_range[1]:.1f}].\n"
    )
    y_col = yu if yu else "value"
    if x_constraint_text is not None:
        x_clause = f"- {x_constraint_text}\n"
    else:
        x_clause = _format_x_constraint_line(units, x_range, num_decimal_x)
    data_header = f"date, {y_col}" if x_formatter is not None else f"{unit}, {y_col}"
    preamble = (
        f"I have a {task_noun} for you.\n"
        if strip_context
        else (
            f"I have a {task_noun} for you. Here's some "
            "context about the task:\n"
            "\n"
            f"{language_context.strip()}\n"
            "\n"
            "Constraints:\n"
            f"{value_clause}"
            f"{x_clause}"
        )
    )
    return (
        f"{preamble}"
        "\n"
        f"Observed data ({data_header}):\n"
        f"{data_lines}\n"
        "\n"
        f"{reasoning_prime}"
        "After you have thought you do not need to respond, just acknowledge "
        "you are ready.\n"
        "\n"
        f"Each {format_noun} you give will be in the format "
        f"`{'date' if x_formatter is not None else unit}, value`."
    )


def _build_chat_prompt(
    tokenizer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: float,
    language_context: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    num_decimal_x: int,
    num_decimal_y: int,
    x_unit: str,
    y_unit: str,
    x_formatter: Callable[[float], str] | None = None,
    x_constraint_text: str | None = None,
    strip_context: bool = False,
) -> str:
    """Build a chat-formatted prompt for one target ``x_test`` (NO thinking).

    Wraps the task framing, observed data, and query in a single user
    message via ``tokenizer.apply_chat_template`` (with
    ``add_generation_prompt=True`` and ``enable_thinking=False``), then
    primes the assistant turn with ``"{x_test}, "`` so the next token is
    the y value being scored.

    Using the model's native chat template is critical for instruction-tuned
    LLMs (like Qwen 3.5 Instruct): without it the model is being fed a raw
    completion-style prompt and most of its instruction-following capability
    is silently lost.
    """
    user_message, str_x_test = _compose_user_message(
        x_train, y_train, x_test, language_context, x_range, y_range,
        num_decimal_x, num_decimal_y, x_unit, y_unit,
        x_formatter=x_formatter,
        x_constraint_text=x_constraint_text,
        strip_context=strip_context,
    )
    chat_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Prime the assistant turn with `{x_test}, ` so the next generated
    # token is the y value. The trailing space is consistent with how
    # `_score_y_grid_for_one_target` builds the suffix tokens.
    return chat_text + f"{str_x_test}, "
