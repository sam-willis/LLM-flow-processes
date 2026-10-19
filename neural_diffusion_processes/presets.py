"""Prompt presets: the demo problems, their data, and their prompt text.

Each ``PromptPreset`` bundles everything domain-specific for one problem -- the
prompt text, the observed context, the ground-truth function, x/y ranges and
units, and how the LLM target grid is laid out.

Note this module does real work at import: two GP draws (seeded, so they are
deterministic) and a ``find_peaks`` pass build the synthetic problems, the NOAA
series are parsed from ``data/``, and ``PROMPT_PRESETS`` is then grown from 11
hand-written entries to 15 by the four ``_ctx20`` twins that
``_register_ctx_scheme_twins`` adds. ``PROMPT_PRESETS`` is a single mutable
dict -- ``scripts/direct_maunaloa.py`` injects entries into it.
"""
from __future__ import annotations

import csv
import datetime as _dt
import math
from dataclasses import dataclass, replace as _dc_replace
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from neural_diffusion_processes import train as _spt


# ---------------------------------------------------------------------------
# Demo problem constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptPreset:
    """A named prompt preset bundling everything domain-specific in one place.

    Holds the LLM prompt text *and* all the numerical setup of the demo
    problem (observed data, ranges), so swapping presets swaps the entire
    experiment in one place.

    Attributes
    ----------
    text
        Free-form domain description fed to the LLM as the language context.
    x_unit
        Singular x-axis unit, e.g. ``"day"`` or ``"month"``. Used in the
        prompt and as the plot x-axis label.
    y_unit
        Full y label (typically a ``"Quantity (unit)"`` description, e.g.
        ``"Stock Price (GBP)"`` or ``"Temperature (\u2103)"``). Used in the
        prompt and as the plot y-axis label.
    x_range, y_range
        Numerical ranges for x and y. ``y_range`` controls the LLM y-grid
        (so it should be tight - irrelevant high values dilute the
        probability mass).
    x_context, y_context
        Observed context points (tuples so the dataclass stays hashable).
        Converted to ``np.ndarray`` at use time.
    n_grid
        Number of x-grid points used by the diffusion sampler (NDP). The
        LLM scores only its target x's (``llm_x_strategy`` /
        ``llm_n_targets``); each is placed at its nearest NDP grid point and
        the remaining grid rows are left uniform (the sparse layout).
    """

    text: str
    x_unit: str  # singular: "day", "month", "hour", ...
    y_unit: str  # full y label, typically "Quantity (unit)" e.g. "Stock Price (GBP)"
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_context: tuple[float, ...]
    y_context: tuple[float, ...]
    n_grid: int = 256  # match training num_points so the model sees a sequence length it was trained on
    # How the LLM scoring x-grid is built. "integer" uses the integer arange
    # of `x_range` (good for naturally-integer indices like days/months —
    # bankruptcy: x_range=(0, 50) gives 51 prompts at x = 0, 1, …, 50). For
    # synthetic presets with x_range like (-2, 2) the integer arange only
    # gives 5 prompts, which is too coarse — use "linspace_subset" instead:
    # x_llm comes from `n_llm_targets` evenly-spaced indices into the NDP
    # x-grid (`np.linspace(0, n_grid-1, llm_n_targets).round()`), deduped,
    # and rendered with the appropriate non-integer decimal precision.
    llm_x_strategy: str = "integer"  # "integer" | "linspace_subset"
    llm_n_targets: int = 51  # only consulted when llm_x_strategy="linspace_subset"
    # Number of decimal places used to render LLM x values in the prompt.
    # Per-preset rather than inferred from the value range. Ignored
    # when llm_x_strategy="integer" (x_dec is forced to 0 there since x_llm
    # is the integer arange of x_range and the LLM should see e.g. "day 5",
    # not "day 5.00").
    llm_x_decimals: int = 2
    # Decimal places used to render LLM y values (and the y-grid step at
    # which the LLM is scored). Per-preset rather than inferred from
    # y_context's inherent precision. Higher values give a finer y-grid at
    # the cost of more bins per scoring call.
    llm_y_decimals: int = 2
    # Optional override for the y-range advertised to the LLM in the prompt
    # (the "Values represent ..., real numbers in [a, b]" line). Useful when
    # `y_range` is widened for NDP/flow tail mass but that wider range would
    # mislead the LLM (e.g. saying stock prices can be negative). The LLM
    # y-grid bins still come from `y_range`; only the prompt text changes.
    # When None, the prompt uses `y_range` directly.
    prompt_range: tuple[float, float] | None = None
    # Optional ground-truth function for synthetic presets. If set, the
    # sample panels and the LLM heatmap overlay the true curve for
    # reference. Evaluated element-wise on the x-grid (pure Python / numpy,
    # not JAX — keeps the plot code standalone).
    true_fn: Callable[[float], float] | None = None
    # If set, the `--n-ctx N` CLI flag resamples N evenly-spaced context
    # x-values from this range and reads y from `true_fn`. When None, the
    # `--n-ctx` flag has no effect and the static `x_context`/`y_context`
    # tuples are used.
    context_x_range: tuple[float, float] | None = None
    # Optional override for the LLM-scoring x-grid range. When set, the
    # x_llm grid is built from THIS range (using `llm_x_strategy`), not
    # from `x_range`. The prompt continues to advertise the full `x_range`
    # so the LLM knows the data span — only the queried subset shrinks.
    # Use case: NDP flow on the full sequence, LLM conditioning on a
    # subrange (e.g. only the first 12 months of a 50-month series).
    # When sparse `prob_xys` is in effect (default), NDP grid points
    # outside `llm_x_range` keep a uniform LLM row — so the LLM term
    # only applies inside the override.
    llm_x_range: tuple[float, float] | None = None
    # Optional callable that renders an x value as a string for the LLM prompt
    # (e.g. `_make_month_formatter(2009, 3)` turns x=1 into "2009-03"). When set:
    #   * data lines and the target-x render use this formatter,
    #   * the "Predict the value at ..." line drops the unit prefix,
    #   * the data-header / format-hint use "date" instead of the unit.
    # Useful for time-indexed series so the LLM doesn't have to do calendar
    # arithmetic from a numeric anchor in the prompt body.
    x_formatter: Callable[[float], str] | None = None
    # Optional override for the bullet-point x constraint line in the prompt.
    # Plain text (no leading "- "); a leading "- " is added at render time.
    # Use to describe a non-numeric x format, e.g. "Dates are in YYYY-MM
    # (year-month) format covering 2009-03 through 2014-02."
    x_constraint_text: str | None = None
    # Strip the prompt down to the observations alone: no domain description
    # and no "Constraints" block, so the user message goes straight
    # from "I have a time series forecasting task for you." to "Observed
    # data". Pair it with `x_unit="x"` / `y_unit="value"` for a fully
    # anonymised series — the LLM sees numbers with no hint of what they
    # measure. Use it to separate what the LLM knows about *this* series
    # from what it can read off the numbers.
    # `text` is still carried on the preset (it labels plots and is stored
    # in the pickle) but never reaches the model.
    strip_context: bool = False


# --- synthetic_1d periodic draw (in-distribution for the training prior) ----
# Drawn from the same periodic × RBF kernel that
# `train.sample_batch` uses to generate training data.
# Kernel hyperparameters land inside the training ranges
# (period ∈ [0.75, 1.25], ls_per ∈ [0.5, 1], ls_rbf ∈ [1, 4], variance = 1) and
# the y_range matches training (-3.5, 3.5) so signal amplitude lives in the
# part of normalized space the model was trained on.
_SYN1D_X_RANGE = (-2.0, 2.0)
_SYN1D_Y_RANGE = (-3.0, 3.0)  # used by changepoint; the periodic tasks have their own below.

_SYN1D_PERIODIC_Y_RANGE = (-3.5, 3.5)
_SYN1D_PERIODIC_PERIOD = 0.9
_SYN1D_PERIODIC_LS_PER = 0.75
_SYN1D_PERIODIC_LS_RBF = 2.0
_SYN1D_PERIODIC_VARIANCE = 1.0
_SYN1D_PERIODIC_JITTER = 1e-4
_SYN1D_PERIODIC_SEED = 42

# Sample the GP once on a dense grid; `_syn1d_periodic_y(x)` linearly
# interpolates from this stored draw, and the 20 context points on
# x ∈ [-2, 0] read off the same draw so context and ground truth agree.
_SYN1D_PERIODIC_GRID_N = 401
_SYN1D_PERIODIC_GRID_X = jnp.linspace(
    _SYN1D_X_RANGE[0], _SYN1D_X_RANGE[1], _SYN1D_PERIODIC_GRID_N
).reshape(-1, 1)
_SYN1D_PERIODIC_GRID_Y = np.asarray(
    _spt._sample_one(
        jax.random.PRNGKey(_SYN1D_PERIODIC_SEED),
        _SYN1D_PERIODIC_GRID_X,
        jnp.float32(_SYN1D_PERIODIC_PERIOD),
        jnp.float32(_SYN1D_PERIODIC_LS_PER),
        jnp.float32(_SYN1D_PERIODIC_LS_RBF),
        jnp.bool_(False),  # is_rbf_only — keep the periodic × RBF mixture
        _SYN1D_PERIODIC_VARIANCE,
        _SYN1D_PERIODIC_JITTER,
    )
)[:, 0]
_SYN1D_PERIODIC_GRID_X_NP = np.asarray(_SYN1D_PERIODIC_GRID_X)[:, 0]


def _syn1d_periodic_y(x: float) -> float:
    return float(np.interp(x, _SYN1D_PERIODIC_GRID_X_NP, _SYN1D_PERIODIC_GRID_Y))


# 20 context x-values uniformly on x ∈ [-2, 0] — left half only, so the
# right half of the prediction grid is pure extrapolation for the model.
_SYN1D_PERIODIC_X = tuple(np.linspace(-2.0, 0.0, 20).tolist())
_SYN1D_PERIODIC_Y = tuple(_syn1d_periodic_y(xv) for xv in _SYN1D_PERIODIC_X)

from scipy.signal import find_peaks as _find_peaks
_SYN1D_PERIODIC_GRID_DX = float(_SYN1D_PERIODIC_GRID_X_NP[1] - _SYN1D_PERIODIC_GRID_X_NP[0])
_SYN1D_PERIODIC_PEAK_MIN_DIST = max(1, int(0.7 * _SYN1D_PERIODIC_PERIOD / _SYN1D_PERIODIC_GRID_DX))
_SYN1D_PERIODIC_MAX_IDX, _ = _find_peaks(
    _SYN1D_PERIODIC_GRID_Y, distance=_SYN1D_PERIODIC_PEAK_MIN_DIST,
)
_SYN1D_PERIODIC_MIN_IDX, _ = _find_peaks(
    -_SYN1D_PERIODIC_GRID_Y, distance=_SYN1D_PERIODIC_PEAK_MIN_DIST,
)


_SYN1D_PERIODIC_TEXT = (
    f"A function with quasi-periodic structure: a periodic kernel of period "
    f"{_SYN1D_PERIODIC_PERIOD:.4g} and lengthscale "
    f"{_SYN1D_PERIODIC_LS_PER:.4g} modulated by an RBF kernel of lengthscale "
    f"{_SYN1D_PERIODIC_LS_RBF:.4g}. Output variance "
    f"{_SYN1D_PERIODIC_VARIANCE:.4g}."
)


# --- synthetic_1d piecewise (changepoint) preset --------------------------
# Piecewise-linear with a single changepoint, values normalized to span the
# full y_range.
# Concrete t / l0 / l1 / r1 (instead of sampling) keep the demo reproducible
# and give the diffusion sampler a clean reference curve.
_SYN1D_CHANGEPOINT_T = 0.625  # normalized position in [0.25, 0.75]
_SYN1D_CHANGEPOINT_L0 = 0.15  # raw left-edge height
_SYN1D_CHANGEPOINT_L1 = 0.9   # raw peak height at the changepoint
_SYN1D_CHANGEPOINT_R1 = 0.1   # raw right-edge height


def _syn1d_changepoint_y(x: float) -> float:
    z = (x - _SYN1D_X_RANGE[0]) / (_SYN1D_X_RANGE[1] - _SYN1D_X_RANGE[0])
    t = _SYN1D_CHANGEPOINT_T
    l0, l1, r1 = (
        _SYN1D_CHANGEPOINT_L0,
        _SYN1D_CHANGEPOINT_L1,
        _SYN1D_CHANGEPOINT_R1,
    )
    if z <= t:
        raw = l0 + (l1 - l0) * (z / max(t, 1e-6))
    else:
        raw = l1 + (r1 - l1) * ((z - t) / max(1.0 - t, 1e-6))
    raw_min = min(l0, l1, r1)
    raw_max = max(l0, l1, r1)
    raw01 = (raw - raw_min) / (raw_max - raw_min + 1e-9)
    return _SYN1D_Y_RANGE[0] + raw01 * (_SYN1D_Y_RANGE[1] - _SYN1D_Y_RANGE[0])


# Context points straddle the changepoint: three on the rising left segment
# and two on the falling right segment, so the LLM term has to place the kink
# rather than interpolating a smooth curve through nearby samples.
_SYN1D_CHANGEPOINT_X = (-1.8, -1.0, -0.2, 1.0, 1.8)
_SYN1D_CHANGEPOINT_Y = tuple(
    _syn1d_changepoint_y(xv) for xv in _SYN1D_CHANGEPOINT_X
)

_SYN1D_CHANGEPOINT_X_LOC = (
    _SYN1D_X_RANGE[0]
    + _SYN1D_CHANGEPOINT_T * (_SYN1D_X_RANGE[1] - _SYN1D_X_RANGE[0])
)


# --- mixture-of-kernels preset with an out-of-prior linear trend ----------
# A test case drawn from the kernels `train.sample_batch` generates training
# data from. The preset stores a dense GT draw and a `context_x_range`,
# so the `--n-ctx N` flag can resample N evenly-spaced context points from
# the GT on demand.
_MIX_X_RANGE = (-2.0, 2.0)
_MIX_Y_RANGE = (-3.5, 3.5)            # matches training y_range
_MIX_CTX_X_RANGE = (-2.0, 0.0)        # left-half context → right-half extrap
_MIX_GRID_N = 401
_MIX_VARIANCE = 1.0
_MIX_JITTER = 1e-4
_MIX_DEFAULT_N_CTX = 20

# Periodic regime: kernel hparams mid-range of training prior.
_MIX_PERIODIC_PERIOD = 1.0    # mid of [0.75, 1.25]
_MIX_PERIODIC_LS_PER = 0.75   # mid of [0.5, 1.0]
_MIX_PERIODIC_LS_RBF = 2.5    # mid of [1.0, 4.0]
_MIX_PERIODIC_SEED = 42


_MIX_GRID_X = jnp.linspace(_MIX_X_RANGE[0], _MIX_X_RANGE[1], _MIX_GRID_N).reshape(-1, 1)
_MIX_GRID_X_NP = np.asarray(_MIX_GRID_X)[:, 0]


def _draw_gt(seed: int, period: float, ls_per: float, ls_rbf: float, is_rbf_only: bool) -> np.ndarray:
    y = _spt._sample_one(
        jax.random.PRNGKey(seed),
        _MIX_GRID_X,
        jnp.float32(period), jnp.float32(ls_per), jnp.float32(ls_rbf),
        jnp.bool_(is_rbf_only),
        _MIX_VARIANCE, _MIX_JITTER,
    )
    return np.asarray(y)[:, 0]


_MIX_PERIODIC_TREND_SLOPE = 0.45
_MIX_PERIODIC_GRID_Y = _draw_gt(
    _MIX_PERIODIC_SEED,
    _MIX_PERIODIC_PERIOD, _MIX_PERIODIC_LS_PER, _MIX_PERIODIC_LS_RBF,
    is_rbf_only=False,
) + _MIX_PERIODIC_TREND_SLOPE * np.asarray(_MIX_GRID_X_NP)


def _mix_periodic_y(x: float) -> float:
    return float(np.interp(x, _MIX_GRID_X_NP, _MIX_PERIODIC_GRID_Y))


_MIX_PERIODIC_CTX_X = tuple(np.linspace(*_MIX_CTX_X_RANGE, _MIX_DEFAULT_N_CTX).tolist())
_MIX_PERIODIC_CTX_Y = tuple(_mix_periodic_y(xv) for xv in _MIX_PERIODIC_CTX_X)


# --- Observed series, read from the NOAA source files in data/ -------------
# data/ holds the NOAA distributions. Everything the presets need is
# parsed out of them here, so the provenance of every number is the file
# beside it and re-downloading is a one-line diff rather than a re-typing job.
#
# Parsed at import, so a missing or truncated file is an ImportError for the
# whole package rather than a wrong number in an output.
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _noaa_lines(filename: str) -> list[list[str]]:
    """Whitespace-split fields of every non-comment line of a NOAA file."""
    path = _DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; the Mauna Loa presets cannot be built "
            f"without it. It is the NOAA distribution file, tracked under "
            f"data/."
        )
    return [ln.split() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def _window(rows, key, start, count, what: str) -> list:
    """`count` consecutive rows from the one whose key == start.

    Raises rather than silently returning a short series: a NOAA re-download
    that drops or shifts rows must fail loudly, not quietly move the data a
    preset describes.
    """
    keys = [key(r) for r in rows]
    try:
        i = keys.index(start)
    except ValueError:
        raise ValueError(f"{what}: no row for {start} in the NOAA file") from None
    got = rows[i:i + count]
    if len(got) != count:
        raise ValueError(
            f"{what}: wanted {count} rows from {start}, the file has "
            f"{len(got)}"
        )
    return got


def _flask_monthly(filename: str, start: tuple[int, int], count: int) -> tuple:
    """`value` column of a surface-flask monthly file (site year month value)."""
    rows = _noaa_lines(filename)
    got = _window(rows, lambda r: (int(r[1]), int(r[2])), start, count, filename)
    return tuple(float(r[3]) for r in got)


def _co2_monthly(start: tuple[int, int], count: int):
    """(average, mid-month date) from co2_mm_mlo.txt.

    Columns: year month decimal_date average deseasonalised ... The date is
    reconstructed from `decimal_date`, which is where the YYYY-MM-DD strings
    the prompts show have always come from.
    """
    rows = _noaa_lines("co2_mm_mlo.txt")
    got = _window(rows, lambda r: (int(r[0]), int(r[1])), start, count,
                  "co2_mm_mlo.txt")
    values = tuple(float(r[3]) for r in got)
    dates = []
    for r in got:
        year = int(r[0])
        frac = float(r[2]) - year
        days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        d = _dt.date(year, 1, 1) + _dt.timedelta(days=round(frac * days))
        dates.append(d.isoformat())
    return values, tuple(dates)


def _co2_weekly(start: tuple[int, int, int], count: int) -> tuple:
    """`CO2 molfrac` from co2_weekly_mlo.txt.

    Columns: year month day decimal_date ppm #days ... -- the ppm is field 4,
    not field 3, which is the decimal date.
    """
    rows = _noaa_lines("co2_weekly_mlo.txt")
    got = _window(rows, lambda r: (int(r[0]), int(r[1]), int(r[2])), start,
                  count, "co2_weekly_mlo.txt")
    return tuple(float(r[4]) for r in got)


_MAUNA_LOA_MONTHLY_CO2, _MAUNA_LOA_MONTHLY_DATES = _co2_monthly((2019, 1), 64)
_MAUNA_LOA_MONTH_X = np.arange(1, 1 + len(_MAUNA_LOA_MONTHLY_CO2), dtype=np.float64)
_MAUNA_LOA_MONTHLY_CO2_NP = np.asarray(_MAUNA_LOA_MONTHLY_CO2, dtype=np.float64)


def _mauna_loa_monthly_y(x: float) -> float:
    return float(np.interp(float(x), _MAUNA_LOA_MONTH_X, _MAUNA_LOA_MONTHLY_CO2_NP))


def _make_monthly_dates_formatter(
    dates: tuple[str, ...],
) -> Callable[[float], str]:
    """Render integer month index x (1-indexed) as the corresponding
    YYYY-MM-DD date from `dates`."""
    def fmt(x: float) -> str:
        return dates[int(round(float(x))) - 1]
    return fmt


_MAUNA_LOA_MONTH_WEEK_CENTERS = (
    (np.arange(len(_MAUNA_LOA_MONTHLY_CO2)) + 0.5) * (52.0 / 12.0)
)


def _mauna_loa_weekly_y(x: float) -> float:
    return float(np.interp(
        float(x), _MAUNA_LOA_MONTH_WEEK_CENTERS, _MAUNA_LOA_MONTHLY_CO2_NP,
    ))


_MAUNA_LOA_WEEKLY_CO2 = _co2_weekly((2019, 1, 6), 256)
_MAUNA_LOA_WEEKLY_CO2_NP = np.asarray(_MAUNA_LOA_WEEKLY_CO2, dtype=np.float64)
_MAUNA_LOA_WEEK_X = np.arange(1, 1 + len(_MAUNA_LOA_WEEKLY_CO2), dtype=np.float64)


def _mauna_loa_co2_weekly_y(x: float) -> float:
    return float(np.interp(
        float(x), _MAUNA_LOA_WEEK_X, _MAUNA_LOA_WEEKLY_CO2_NP,
    ))

_MAUNA_LOA_MONTHLY_CH4_2009 = _flask_monthly(
    "ch4_mlo_surface-flask_1_ccgg_month.txt", (2009, 3), 60)
_MAUNA_LOA_MONTHLY_CH4_2009_NP = np.asarray(
    _MAUNA_LOA_MONTHLY_CH4_2009, dtype=np.float64,
)


def _mauna_loa_ch4_2009_monthly_y(x: float) -> float:
    return float(np.interp(
        float(x),
        np.arange(1, 1 + len(_MAUNA_LOA_MONTHLY_CH4_2009_NP), dtype=np.float64),
        _MAUNA_LOA_MONTHLY_CH4_2009_NP,
    ))


def _make_month_formatter(
    start_year: int, start_month: int
) -> Callable[[float], str]:
    """Render integer month index x (x=1 -> (start_year, start_month)) as YYYY-MM."""
    def fmt(x: float) -> str:
        m_idx = int(round(float(x))) - 1
        year = start_year + (start_month - 1 + m_idx) // 12
        month = (start_month - 1 + m_idx) % 12 + 1
        return f"{year:04d}-{month:02d}"
    return fmt


def _make_weekly_formatter(
    start_year: int, start_month: int, start_day: int
) -> Callable[[float], str]:
    """Render integer week index x (x=1 -> the given anchor date) as YYYY-MM-DD,
    stepping 7 days per unit increment.
    """
    import datetime as _dt
    anchor = _dt.date(start_year, start_month, start_day)
    def fmt(x: float) -> str:
        w_idx = int(round(float(x))) - 1
        d = anchor + _dt.timedelta(days=7 * w_idx)
        return d.isoformat()
    return fmt

_MAUNA_LOA_MONTHLY_N2O_2020 = _flask_monthly(
    "n2o_mlo_surface-flask_1_ccgg_month.txt", (2020, 1), 60)
_MAUNA_LOA_MONTHLY_N2O_2020_NP = np.asarray(
    _MAUNA_LOA_MONTHLY_N2O_2020, dtype=np.float64,
)


def _mauna_loa_n2o_2020_monthly_y(x: float) -> float:
    return float(np.interp(
        float(x),
        np.arange(1, 1 + len(_MAUNA_LOA_MONTHLY_N2O_2020_NP), dtype=np.float64),
        _MAUNA_LOA_MONTHLY_N2O_2020_NP,
    ))

_MAUNA_LOA_MONTHLY_SF6_2020 = _flask_monthly(
    "sf6_mlo_surface-flask_1_ccgg_month.txt", (2020, 1), 60)
_MAUNA_LOA_MONTHLY_SF6_2020_NP = np.asarray(
    _MAUNA_LOA_MONTHLY_SF6_2020, dtype=np.float64,
)


def _mauna_loa_sf6_2020_monthly_y(x: float) -> float:
    return float(np.interp(
        float(x),
        np.arange(1, 1 + len(_MAUNA_LOA_MONTHLY_SF6_2020_NP), dtype=np.float64),
        _MAUNA_LOA_MONTHLY_SF6_2020_NP,
    ))


PROMPT_PRESETS: dict[str, PromptPreset] = {
    # The text states only the function family, with no numeric parameters,
    # which probes how much the LLM term leans on exact values rather than on
    # recognising the shape.
    "changepoint": PromptPreset(
        text="A piecewise linear function with a changepoint.",
        x_unit="x",
        y_unit="value",
        x_range=_SYN1D_X_RANGE,
        y_range=_SYN1D_Y_RANGE,
        x_context=_SYN1D_CHANGEPOINT_X,
        y_context=_SYN1D_CHANGEPOINT_Y,
        llm_x_strategy="linspace_subset",
        true_fn=_syn1d_changepoint_y,
    ),
    "bankruptcy": PromptPreset(
        text=(
            "A small UK company's daily stock price in GBP over 50 trading "
            "days. On day 30 the company enters compulsory liquidation and "
            "is permanently delisted from the London Stock Exchange."
        ),
        x_unit="day",
        y_unit="Stock Price (GBP)",
        x_range=(0.0, 50.0),
        # Wide symmetric y_range so the NDP has natural mass on both sides
        # of zero (post-bankruptcy regime hugs 0 with negative-side tails).
        # `prompt_range` is the narrower range advertised to the LLM, so the
        # prompt doesn't tell it that "stock prices can be negative" — the
        # LLM y-grid bins still come from `y_range`.
        y_range=(-3.0, 7.0),
        prompt_range=(0.0, 7.0),
        x_context=(1.0, 2.0),
        y_context=(2.5, 2.3),
        # Natural-language stock prices: LLM emits 1-decimal values like
        # "2.5"; scoring at 0.01 step would spread probability uniformly
        # across all 1001 bins. Match the inherent 1-decimal precision.
        llm_y_decimals=1,
    ),
    "sandiego": PromptPreset(
        text=(
            "Monthly average precipitation in San Diego in inches over 50 months, with x=1 is January."
        ),
        x_unit="month",
        y_unit="Avg. Precip. (in)",
        x_range=(0.0, 50.0),
        # y_range chosen so the NDP's training-natural amplitude band covers
        # the LLM's plausible mass: centered at the expected mean (~1 inch),
        # width 4 keeps the same normalization scale. Lower edge -1 is
        # physically zero-prob for the LLM (negative precipitation) but it
        # makes the NDP's symmetric band line up with the data —
        # `prompt_range` advertises the physical [0, 3] to the LLM only.
        y_range=(-1.0, 3.0),
        prompt_range=(0.0, 3.0),
        # Month 1 = January, month 2 = February (as the prompt text states).
        x_context=(1.0, 2.0),
        y_context=(2.0, 1.7),
        # Precipitation in inches: LLM emits 2-decimal values like "0.73".
        llm_y_decimals=2,
    ),
    # Same setup as `sandiego`, with an extra anchor at x=26 (≈ year 3 February)
    # so the LLM has direct evidence the seasonal cycle repeats partway through
    # the observation window.
    "sandiego_long": PromptPreset(
        text=(
            "Monthly average precipitation in San Diego in inches over 50 months, with x=1 is January."
        ),
        x_unit="month",
        y_unit="Avg. Precip. (in)",
        x_range=(0.0, 50.0),
        y_range=(-1.0, 3.0),
        prompt_range=(0.0, 3.0),
        x_context=(1.0, 2.0, 26.0),
        y_context=(2.0, 1.7, 2.15),
        llm_y_decimals=2,
    ),
    # Mauna Loa CH4 (ppb), Mar 2009 - Feb 2014, first 30 months as context.
    # The task description deliberately states no seasonal-cycle facts and
    # gives no peak/trough planning instruction: every Mauna Loa preset here
    # withholds them, so the LLM infers the structure from the data alone.
    "mauna_loa_ch4_2009": PromptPreset(
        text=(
            "Monthly mean atmospheric CH4 (methane) concentration in ppb at "
            "Mauna Loa Observatory from March 2009 through February 2014. "
            "The provided observations cover the first 30 months (through "
            "August 2011); forecast the remainder."
        ),
        x_unit="month",
        y_unit="CH4 (ppb)",
        x_range=(0.0, 60.0),
        y_range=(1780.0, 1860.0),
        prompt_range=(1780.0, 1860.0),
        x_context=tuple(float(i) for i in range(1, 31)),
        y_context=tuple(_mauna_loa_ch4_2009_monthly_y(float(i)) for i in range(1, 31)),
        llm_y_decimals=1,
        true_fn=_mauna_loa_ch4_2009_monthly_y,
        x_formatter=_make_month_formatter(2009, 3),
        x_constraint_text=(
            "Dates are in YYYY-MM (year-month) format covering "
            "2009-03 through 2014-02."
        ),
    ),
    # Mauna Loa N2O (ppb), Jan 2020 - Dec 2024, first 30 months as context.
    # Most recent 60 months available. Near-monotonic ~1 ppb/yr trend, very
    # weak seasonality -- stated here and NOT in `text`, which would hand the
    # LLM the answer to most of the forecast.
    "mauna_loa_n2o_2020": PromptPreset(
        text=(
            "Monthly mean atmospheric N2O (nitrous oxide) concentration in "
            "ppb at Mauna Loa Observatory from January 2020 through December "
            "2024. The provided observations cover the first 30 months "
            "(through June 2022); forecast the remainder."
        ),
        x_unit="month",
        y_unit="N2O (ppb)",
        x_range=(0.0, 60.0),
        y_range=(332.0, 340.0),
        prompt_range=(332.0, 340.0),
        x_context=tuple(float(i) for i in range(1, 31)),
        y_context=tuple(_mauna_loa_n2o_2020_monthly_y(float(i)) for i in range(1, 31)),
        llm_y_decimals=2,
        true_fn=_mauna_loa_n2o_2020_monthly_y,
        x_formatter=_make_month_formatter(2020, 1),
        x_constraint_text=(
            "Dates are in YYYY-MM (year-month) format covering "
            "2020-01 through 2024-12."
        ),
    ),
    # Mauna Loa SF6 (ppt), Jan 2020 - Dec 2024, first 30 months as context.
    # Most recent 60 months available. Near-linear ~0.4 ppt/yr trend,
    # negligible seasonality -- stated here and NOT in `text`, for the same
    # reason as N2O above.
    "mauna_loa_sf6_2020": PromptPreset(
        text=(
            "Monthly mean atmospheric SF6 (sulfur hexafluoride) concentration "
            "in ppt at Mauna Loa Observatory from January 2020 through "
            "December 2024. The provided observations cover the first 30 "
            "months (through June 2022); forecast the remainder."
        ),
        x_unit="month",
        y_unit="SF6 (ppt)",
        x_range=(0.0, 60.0),
        y_range=(10.0, 12.5),
        prompt_range=(10.0, 12.5),
        x_context=tuple(float(i) for i in range(1, 31)),
        y_context=tuple(_mauna_loa_sf6_2020_monthly_y(float(i)) for i in range(1, 31)),
        llm_y_decimals=2,
        true_fn=_mauna_loa_sf6_2020_monthly_y,
        x_formatter=_make_month_formatter(2020, 1),
        x_constraint_text=(
            "Dates are in YYYY-MM (year-month) format covering "
            "2020-01 through 2024-12."
        ),
    ),
    "mauna_loa_co2_interp_monthly": PromptPreset(
        text=(
            "Weekly mean atmospheric CO2 concentration in ppm at Mauna Loa "
            "Observatory across roughly 5 years (256 weekly observations from "
            "2019-01-06 through 2023-11-26). The provided observations are 20 "
            "sparsely-and-randomly sampled weeks from the first half (through "
            "2021-06-13); forecast the remainder."
        ),
        x_unit="week",
        y_unit="CO2 (ppm)",
        x_range=(0.0, 256.0),
        y_range=(405.0, 430.0),
        prompt_range=(405.0, 430.0),
        x_context=(
            2.0, 5.0, 9.0, 21.0, 31.0, 35.0, 57.0, 61.0, 69.0, 71.0,
            72.0, 75.0, 78.0, 80.0, 91.0, 93.0, 96.0, 110.0, 120.0, 128.0,
        ),
        y_context=tuple(
            _mauna_loa_weekly_y(float(w)) for w in (
                2, 5, 9, 21, 31, 35, 57, 61, 69, 71,
                72, 75, 78, 80, 91, 93, 96, 110, 120, 128,
            )
        ),
        llm_y_decimals=1,
        llm_x_strategy="linspace_subset",
        llm_n_targets=64,
        llm_x_decimals=0,
        true_fn=_mauna_loa_weekly_y,
        x_formatter=_make_weekly_formatter(2019, 1, 6),
        x_constraint_text=(
            "Dates are in YYYY-MM-DD (week-start, Sunday-anchored) format "
            "covering 2019-01-06 through 2023-11-26."
        ),
    ),
    # Same 20-pt random context schedule as `mauna_loa_co2_interp_monthly`,
    # but y values come from the actual NOAA weekly CO2 file
    # (co2_weekly_mlo.txt) rather than from monthly means interpolated to a
    # weekly grid, so this series carries genuine sub-monthly variability.
    "mauna_loa_co2_weekly": PromptPreset(
        text=(
            "Weekly mean atmospheric CO2 concentration in ppm at Mauna Loa "
            "Observatory across roughly 5 years (256 weekly observations from "
            "2019-01-06 through 2023-11-26). The provided observations are 20 "
            "sparsely-and-randomly sampled weeks from the first half (through "
            "2021-06-13); forecast the remainder."
        ),
        x_unit="week",
        y_unit="CO2 (ppm)",
        x_range=(0.0, 256.0),
        y_range=(405.0, 430.0),
        prompt_range=(405.0, 430.0),
        x_context=(
            2.0, 5.0, 9.0, 21.0, 31.0, 35.0, 57.0, 61.0, 69.0, 71.0,
            72.0, 75.0, 78.0, 80.0, 91.0, 93.0, 96.0, 110.0, 120.0, 128.0,
        ),
        y_context=tuple(
            _mauna_loa_co2_weekly_y(float(w)) for w in (
                2, 5, 9, 21, 31, 35, 57, 61, 69, 71,
                72, 75, 78, 80, 91, 93, 96, 110, 120, 128,
            )
        ),
        llm_y_decimals=1,
        llm_x_strategy="linspace_subset",
        llm_n_targets=64,
        llm_x_decimals=0,
        true_fn=_mauna_loa_co2_weekly_y,
        x_formatter=_make_weekly_formatter(2019, 1, 6),
        x_constraint_text=(
            "Dates are in YYYY-MM-DD (week-start, Sunday-anchored) format "
            "covering 2019-01-06 through 2023-11-26."
        ),
    ),
    # --- mixture-of-kernels preset with an out-of-prior linear trend ---
    # The periodic part is drawn from the kernels `train.sample_batch` trains
    # on, with hyperparameters mid-range within the training prior; the added
    # linear trend is outside that prior. Default context: 20 points on x ∈ [-2, 0]
    # (left half), so the right half tests extrapolation. Use the
    # `--n-ctx N` CLI flag to resample N points from the dense GT.
    "periodic_with_lin_trend": PromptPreset(
        text="A periodic function with a linear trend",
        x_unit="x",
        y_unit="value",
        x_range=_MIX_X_RANGE,
        y_range=_MIX_Y_RANGE,
        x_context=_MIX_PERIODIC_CTX_X,
        y_context=_MIX_PERIODIC_CTX_Y,
        llm_x_strategy="linspace_subset",
        true_fn=_mix_periodic_y,
        context_x_range=_MIX_CTX_X_RANGE,
    ),
    "montreal": PromptPreset(
        text=(
            "Daily average temperature in Montreal in degrees Celsius over "
            "50 days in May and June, starting May 1st."
        ),
        x_unit="day",
        y_unit="Temperature (\u2103)",
        x_range=(0.0, 50.0),
        # Montreal May/June daily means range roughly 5-25 \u2103.
        y_range=(0.0, 30.0),
        # May 1 / 2 in Montreal averages ~14\u2103.
        x_context=(1.0, 2.0),
        y_context=(14.0, 14.0),
        # Multi-day weather correlations.
        llm_y_decimals=1,
    ),
}


# --- Context-scheme twins for the fine-tuning set --------------------------
# The four fine-tuning series do not sample their context the same way. CO2
# observes 20 scattered weeks out of the 127 available in its context window;
# the three gas series observe every month of theirs. Both schemes hand over
# the first half of the domain and forecast the second, so the horizon matches
# and the only difference is observation density. Comparing CO2 against the
# gases therefore confounds "which series" with "which context scheme".
#
# This breaks the confound by reading every series at the same density:
#
#   <preset>_ctx20    20 scattered native observations from the context window
#
# so a CO2-vs-gas comparison is about the series rather than about how much of
# it was observed. For CO2 `_ctx20` reproduces the base preset's context
# exactly; the three gases are subsampled from their full month-by-month
# context.
#
# Two things have to move together with x_context:
#   * y_context comes from `true_fn`, which reproduces every preset's own
#     y_context exactly (max |true_fn(x_ctx) - y_ctx| == 0 on all 4).
#   * `text` describes the sampling scheme in words ("the first 30 months",
#     "20 sparsely-and-randomly sampled weeks"), so it is rewritten to match.
#     Leaving it would hand the LLM a prompt that contradicts its own data.
# The last context point is always retained, fixing the forecast region at
# x > cutoff across both schemes.
_CTX_SCHEME_BASES = (
    "mauna_loa_co2_weekly",
    "mauna_loa_ch4_2009",
    "mauna_loa_n2o_2020",
    "mauna_loa_sf6_2020",
)


def _register_ctx_scheme_twins(
    bases=_CTX_SCHEME_BASES, n_scatter: int = 20, seed: int = 0,
) -> None:
    """Register `<base>_ctx20` for each base preset."""
    marker = "The provided observations "
    for base_name in bases:
        base = PROMPT_PRESETS[base_name]
        xc = [float(v) for v in base.x_context]
        # All four series sit on integer x (weeks or months), so the available
        # observations in the window are the unit-spaced ticks it spans.
        ticks = [float(v) for v in
                 range(int(math.ceil(min(xc))), int(math.floor(max(xc))) + 1)]
        unit = "week" if "weekly" in base_name else "month"

        head, sep, tail = base.text.partition(marker)
        if not sep or "; forecast the remainder." not in tail:
            raise RuntimeError(
                f"{base_name}: prompt no longer states its context scheme in "
                f"the expected form; update _register_ctx_scheme_twins")
        through = tail[tail.index("(through "):tail.index(")") + 1]

        if len(xc) == n_scatter:
            x20 = xc                      # CO2: keep the base preset's draw
        else:
            rng = np.random.default_rng(seed)
            pool = [v for v in ticks if v != ticks[-1]]
            x20 = sorted(rng.choice(pool, size=n_scatter - 1, replace=False)
                         .tolist() + [ticks[-1]])

        sentence = (
            f"{marker}are {n_scatter} sparsely-and-randomly sampled {unit}s "
            f"from the first half {through}; forecast the remainder."
        )
        PROMPT_PRESETS[f"{base_name}_ctx20"] = _dc_replace(
            base,
            text=head + sentence,
            x_context=tuple(float(v) for v in x20),
            y_context=tuple(float(base.true_fn(float(v))) for v in x20),
        )


_register_ctx_scheme_twins()

# The CLI default preset.
DEFAULT_PROMPT_NAME = "periodic_with_lin_trend"


def _default_save_path(
    prompt_name: str,
    enable_thinking: bool = True,
    n_ctx: int | None = None,
) -> Path:
    suffix = "" if enable_thinking else "_nothink"
    n_part = f"_n{n_ctx}" if n_ctx is not None else ""
    return Path(f"./outputs/{prompt_name}{n_part}{suffix}.pkl")


def _default_plot_path(
    prompt_name: str,
    enable_thinking: bool = True,
    n_ctx: int | None = None,
    sparse: bool = False,
) -> Path:
    suffix = "" if enable_thinking else "_nothink"
    n_part = f"_n{n_ctx}" if n_ctx is not None else ""
    sparse_part = "_sparse" if sparse else ""
    return Path(f"./outputs/{prompt_name}{n_part}{sparse_part}{suffix}_demo.png")
