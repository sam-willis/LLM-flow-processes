"""Figure panels for the NDP / NDP+LLM / LLM-marginal comparisons.

matplotlib + numpy only. ``LLMExtraScoreFn`` is referenced in annotations only,
so it is imported under ``TYPE_CHECKING`` to avoid a cycle back through the
scoring module.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from neural_diffusion_processes.pipeline import LLMExtraScoreFn

def _midpoint_edges(arr: np.ndarray) -> np.ndarray:
    """Cell edges for ``pcolormesh`` from bin centres (midpoints, ends mirrored)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 1:
        return np.array([arr[0] - 0.5, arr[0] + 0.5])
    mids = 0.5 * (arr[:-1] + arr[1:])
    return np.concatenate([
        [arr[0] - (mids[0] - arr[0])],
        mids,
        [arr[-1] + (arr[-1] - mids[-1])],
    ])


def _scored_columns(prob_xys: np.ndarray) -> tuple[np.ndarray, bool]:
    """Which x-grid columns the LLM actually scored, and whether the run is sparse.

    Sparse runs fill unscored rows with a constant ``1/n_y_bins``; detect those
    exactly. Returns ``(is_scored, sparse_mode)``.
    """
    uniform_val = 1.0 / prob_xys.shape[1]
    is_scored = ~np.all(np.isclose(prob_xys, uniform_val, atol=1e-12), axis=1)
    return is_scored, not bool(is_scored.all())


def _plot_basic_panels(
    *,
    y_diffusion_context: np.ndarray,
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_grid: np.ndarray,
    x_unit: str,
    y_unit: str,
    language_context: str,
    y_lim: tuple[float, float],
    plot_path: Path,
    true_fn: Callable[[float], float] | None = None,
) -> None:
    """Single-panel plot (NDP flow no LLM) for fast NDP-only iteration.

    Use when the LLM scoring stage has been skipped (`--no-llm-panels`)
    so we don't pay for Qwen and the LLM-conditioned sampler just to throw
    the panels away. Layout matches the NDP-no-LLM panel in `_plot_panels`
    so visual comparisons stay one-to-one.
    """
    x_label = x_unit
    y_label = y_unit if y_unit else "value"

    fig, axes = plt.subplots(1, 1, figsize=(5, 4.8), squeeze=False)
    axes = axes[0]

    true_y = (
        np.asarray([true_fn(float(xv)) for xv in x_grid], dtype=np.float64)
        if true_fn is not None
        else None
    )

    sample_panels = [
        ("NDP flow (no LLM)", y_diffusion_context),
    ]
    n_show_samples = 2
    individual_colors = ["tab:orange", "tab:green"]
    for ax, (title, ys) in zip(axes, sample_panels):
        med = np.median(ys, axis=0)
        lo = np.quantile(ys, 0.025, axis=0)
        hi = np.quantile(ys, 0.975, axis=0)
        ax.fill_between(
            x_grid, lo, hi, color="C0", alpha=0.20, label="95% band",
        )
        ax.plot(x_grid, med, color="C0", linewidth=1.4, label="median")
        for s in range(min(n_show_samples, ys.shape[0])):
            ax.plot(
                x_grid, ys[s],
                color=individual_colors[s], alpha=0.85, linewidth=0.9,
            )
        if true_y is not None:
            ax.plot(
                x_grid, true_y,
                color="black", linestyle="--", linewidth=1.5,
                zorder=9, label="true function",
            )
        ax.scatter(x_context, y_context, color="red", zorder=10, label="context")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.axhline(0.0, color="grey", linewidth=0.5, alpha=0.5)
    axes[0].set_ylim(float(y_lim[0]), float(y_lim[1]))
    axes[0].legend(loc="best", fontsize=8)

    wrapped_prompt = "\n".join(textwrap.wrap(language_context, width=120))
    fig.suptitle(wrapped_prompt, fontsize=11, y=0.99)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"[plotting] wrote {plot_path}", flush=True)


def _plot_llm_only(
    extra_fn: LLMExtraScoreFn,
    plot_path: Path,
    true_fn: Callable[[float], float] | None = None,
) -> None:
    """Single-panel plot of just the LLM log-prob heatmap.

    Same heatmap as panel 4 of `_plot_panels`, used by `--plot-llm-only` to
    diagnose LLM scoring in isolation (no GP, no flow checkpoint, no diffusion
    sampling).
    """
    x_context = extra_fn.x_context
    y_context = extra_fn.y_context
    x_grid = extra_fn.x_grid
    yg = extra_fn.y_grid
    x_label = extra_fn.x_unit
    y_label = extra_fn.y_unit if extra_fn.y_unit else "value"

    is_scored, sparse_mode = _scored_columns(extra_fn.prob_xys)

    p = extra_fn.prob_xys.T
    log_p = np.log(np.clip(p, 1e-12, None))
    if sparse_mode and bool(is_scored.any()):
        x_show = x_grid[is_scored]
        log_p_show = log_p[:, is_scored]
    else:
        x_show = x_grid
        log_p_show = log_p

    x_edges = _midpoint_edges(x_show)
    y_edges = _midpoint_edges(np.asarray(yg))
    cmap = matplotlib.colormaps["viridis"].copy()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    im = ax.pcolormesh(x_edges, y_edges, log_p_show, cmap=cmap, shading="flat")
    if true_fn is not None:
        true_y = np.asarray([true_fn(float(xv)) for xv in x_grid], dtype=np.float64)
        ax.plot(
            x_grid, true_y,
            color="white", linestyle="--", linewidth=1.5,
            zorder=9, label="true function",
        )
    ax.scatter(x_context, y_context, color="red", zorder=10, label="context")
    ax.set_ylim(float(yg[0]), float(yg[-1]))
    ax.set_title(f"LLM log p(value | {x_label}, context, text)", fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="best", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log p")

    wrapped_prompt = "\n".join(textwrap.wrap(extra_fn.language_context, width=120))
    fig.suptitle(wrapped_prompt, fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"[plotting] wrote {plot_path}", flush=True)


def _plot_panels(
    extra_fn: LLMExtraScoreFn,
    y_diffusion_context: np.ndarray,
    y_diffusion_context_plus_llm: np.ndarray,
    y_llm_marginal: np.ndarray,
    plot_path: Path,
    true_fn: Callable[[float], float] | None = None,
) -> None:
    """4-panel comparison plot.

    Panels 0-2 show samples (each ``y_*`` shape ``(n_samples, n_grid)``):
        0. NDP flow samples, no LLM term.
        1. NDP flow samples + LLM extra-score term.
        2. ILLMP marginal: i.i.d. draws from ``extra_fn.prob_xys`` per x
           (pure LLM marginal, no cross-x correlation). Visualised the
           same way as panels 0-1 (median + 95% band + a couple of sample
           trajectories) so spreads are directly comparable; the sample
           lines are jagged by design since neighbouring x's are
           independent.
    Panel 3 is the raw LLM probability distribution.
    """
    x_context = extra_fn.x_context
    y_context = extra_fn.y_context
    x_grid = extra_fn.x_grid
    yg = extra_fn.y_grid
    x_label = extra_fn.x_unit  # "day", "month", ...
    # y_unit holds the full y label (e.g. "Stock Price (GBP)"), not just the
    # bare unit - use it directly. Falls back to "value" if empty.
    y_label = extra_fn.y_unit if extra_fn.y_unit else "value"

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    # Share y-axis across the three sample panels (so they're directly
    # comparable). The heatmap panel uses its own y-axis (the LLM y-grid).
    for i in range(1, 3):
        axes[i].sharey(axes[0])

    # Ground-truth overlay (optional; only for synthetic presets that carry
    # a closed-form function).
    true_y = (
        np.asarray([true_fn(float(xv)) for xv in x_grid], dtype=np.float64)
        if true_fn is not None
        else None
    )

    is_scored, sparse_mode = _scored_columns(extra_fn.prob_xys)

    sample_panels = [
        ("NDP flow (no LLM)", y_diffusion_context, "line"),
        ("NDP flow + LLM", y_diffusion_context_plus_llm, "line"),
        # ILLMP draws are i.i.d. across x (no cross-x correlation in the
        # sampler), so the sample-trajectories will be jagged on purpose
        # — the panel still shows a median + 95% band like the others
        # so spreads are directly comparable across panels.
        ("ILLMP (LLM marginal)", y_llm_marginal, "line_illmp"),
    ]
    n_show_samples = 2
    individual_colors = ["tab:orange", "tab:green"]
    for ax, (title, ys, kind) in zip(axes[:3], sample_panels):
        if kind in ("line", "line_illmp"):
            # ILLMP samples are i.i.d. across x: only the columns the LLM
            # actually scored carry a real distribution. In sparse mode the
            # other columns are uniform 1/n_y_bins, so quantiles over them
            # would be meaningless; restrict the band + trajectories to the
            # scored columns.
            if kind == "line_illmp" and sparse_mode:
                x_band = x_grid[is_scored]
                ys_band = ys[:, is_scored]
            else:
                x_band = x_grid
                ys_band = ys
            med = np.median(ys_band, axis=0)
            lo = np.quantile(ys_band, 0.025, axis=0)
            hi = np.quantile(ys_band, 0.975, axis=0)
            ax.fill_between(
                x_band, lo, hi, color="C0", alpha=0.20, label="95% band",
            )
            ax.plot(x_band, med, color="C0", linewidth=1.4, label="median")
            n_actual = min(n_show_samples, ys_band.shape[0])
            for s in range(n_actual):
                ax.plot(
                    x_band, ys_band[s],
                    color=individual_colors[s], alpha=0.85, linewidth=0.9,
                )
        if true_y is not None:
            ax.plot(
                x_grid, true_y,
                color="black", linestyle="--", linewidth=1.5,
                zorder=9, label="true function",
            )
        ax.scatter(x_context, y_context, color="red", zorder=10, label="context")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.axhline(0.0, color="grey", linewidth=0.5, alpha=0.5)
    # Match the LLM y-grid extent so all four panels share a common y-axis.
    axes[0].set_ylim(float(yg[0]), float(yg[-1]))  # propagates to panels 1-2 via sharey
    axes[0].legend(loc="best", fontsize=8)

    # Panel 4: log p of the LLM distribution per (x, y_bin) as a pcolormesh
    # grid. In sparse mode, only the scored x-columns are drawn; the unscored
    # columns are dropped from the plot entirely so the result is a contiguous
    # grid of colored cells without white gaps.
    p = extra_fn.prob_xys.T  # transpose so y is vertical, x is horizontal
    log_p = np.log(np.clip(p, 1e-12, None))
    if sparse_mode and bool(is_scored.any()):
        x_show = x_grid[is_scored]
        log_p_show = log_p[:, is_scored]
    else:
        x_show = x_grid
        log_p_show = log_p

    x_edges = _midpoint_edges(x_show)
    y_edges = _midpoint_edges(np.asarray(yg))
    cmap = matplotlib.colormaps["viridis"].copy()
    im_llm = axes[3].pcolormesh(
        x_edges, y_edges, log_p_show, cmap=cmap, shading="flat",
    )
    if true_y is not None:
        axes[3].plot(
            x_grid, true_y,
            color="white", linestyle="--", linewidth=1.5,
            zorder=9, label="true function",
        )
    axes[3].scatter(x_context, y_context, color="red", zorder=10, label="context")
    axes[3].set_ylim(float(yg[0]), float(yg[-1]))
    axes[3].set_title(f"LLM log p(value | {x_label}, context, text)", fontsize=10)
    axes[3].set_xlabel(x_label)
    axes[3].set_ylabel(y_label)
    axes[3].legend(loc="best", fontsize=8)
    fig.colorbar(im_llm, ax=axes[3], fraction=0.046, pad=0.04, label="log p")

    # Shared figure title: the language prompt itself, wrapped to fit.
    wrapped_prompt = "\n".join(textwrap.wrap(extra_fn.language_context, width=120))
    fig.suptitle(wrapped_prompt, fontsize=11, y=0.99)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"[plotting] wrote {plot_path}", flush=True)
