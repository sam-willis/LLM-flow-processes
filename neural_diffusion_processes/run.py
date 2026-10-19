"""Running the method with its default settings.

The samplers themselves live in `pipeline`; this is the layer above them --
which checkpoint, how many samples, which guidance strengths -- pinned in one
place so every caller samples the same way.

Paths are environment-overridable so an alternative prior or a re-run can go
through the same code without edits:

    NDP_FLOW_DIR      the flow checkpoint to sample from
    NDP_OUTPUTS_DIR   where LLM caches, truth sidecars and sample caches live
    NDP_N_CORR        Langevin corrector steps (see below)
    NDP_TAU           corrector step scale, default 0.1 (read in `pipeline`)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from neural_diffusion_processes import pipeline as ndp
from neural_diffusion_processes.paths import default_flow_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Method display names -------------------------------------------------
# Single source of truth for the four method names. Update here to rename
# them everywhere.
METHOD_TITLES: dict[str, str] = {
    "direct":  "LLM-Direct",
    "ndp":     "NDP",
    "ndp_llm": "LLM-FP (ours)",
    "illmp":   "LLM-Independent",
}

# Where the LLM pickles, truth sidecars and sample caches live. Defaults to
# outputs/ under the repository root, which is where `pipeline` writes its LLM
# score pickles when run from there. Overridable via NDP_OUTPUTS_DIR (same
# spirit as NDP_FLOW_DIR below) so a re-run can read and write its own artifact
# directory without touching the default one. Pair it with a suffix, or a
# re-run will collide with the default names inside its own directory.
OUTPUTS_DIR = Path(os.environ.get(
    "NDP_OUTPUTS_DIR", PROJECT_ROOT / "outputs",
))

# The flow checkpoint to sample from. Overridable via NDP_FLOW_DIR so an
# alternative prior (e.g. a fine-tune) can be run through the same pipeline
# without edits. Pair it with a samples-cache suffix, or the run will overwrite
# the cache for the default prior.
FLOW_DIR = default_flow_dir()

# --- Default sampling / scoring hyperparameters --------------------------
DEFAULTS = {
    "n_samples": 800,
    "likelihood_variance_llm": 0.001,
    "gamma_llm": 1.0,
    "llm_sparse": True,
    "gamma_scaling_distributional": 1.0,
    "gamma_scaling_ndp": 50.0,
    "condition_ndp_part": True,
    "flow_likelihood_variance": 0.01,
    "ndp_likelihood_noise_std": 0.001,
}

# 4 Langevin corrections, matching the pipeline CLI's `--n-corr 4` default.
# NDP_N_CORR overrides it — 0 gives the pure flow ODE. Any non-default value
# MUST be paired with its own samples-cache suffix: the
# sampler setting is not recorded in the npz, so a cache written at one n_corr
# is indistinguishable from one written at another.
ndp._FLOW_SAMPLER_NUM_CORRECTIONS = int(os.environ.get("NDP_N_CORR", "4"))


# Rebuilds the LLM's additive score term from the cache written by
# compute_and_save_llm_extra_score_fn, filling the three knobs from DEFAULTS.
# Calling LLMExtraScoreFn.from_file directly means knowing those values,
# which is exactly what DEFAULTS exists to pin down.
# --- Loaders ---------------------------------------------------------------
def load_extra_fn(
    preset_name: str, *, suffix: str = "",
    likelihood_variance: float | None = None,
    gamma: float | None = None,
    sparse: bool | None = None,
):
    """Rebuild the LLM score term for ``preset_name`` from its cache."""
    # The caches were pickled from a script running as __main__, so the class
    # is recorded as __main__.LLMExtraScoreFn. Without this alias any caller
    # that is itself __main__ -- i.e. every standalone script -- fails with
    # "Can't get attribute 'LLMExtraScoreFn'". Aliasing here means callers do
    # not have to know that.
    import __main__ as _main
    if not hasattr(_main, "LLMExtraScoreFn"):
        _main.LLMExtraScoreFn = ndp.LLMExtraScoreFn
    return ndp.LLMExtraScoreFn.from_file(
        OUTPUTS_DIR / f"{preset_name}{suffix}.pkl",
        likelihood_variance=(
            likelihood_variance if likelihood_variance is not None
            else DEFAULTS["likelihood_variance_llm"]
        ),
        gamma=gamma if gamma is not None else DEFAULTS["gamma_llm"],
        sparse=sparse if sparse is not None else DEFAULTS["llm_sparse"],
    )


def load_network(
    *,
    gamma_scaling_distributional: float | None = None,
    condition_ndp_part: bool | None = None,
):
    print(f"[fig] loading flow checkpoint from {FLOW_DIR}", flush=True)
    return ndp.load_ndp_flow_network_local(
        FLOW_DIR,
        gamma_scaling_distributional=(
            gamma_scaling_distributional if gamma_scaling_distributional is not None
            else DEFAULTS["gamma_scaling_distributional"]
        ),
        condition_ndp_part=(
            condition_ndp_part if condition_ndp_part is not None
            else DEFAULTS["condition_ndp_part"]
        ),
    )


def sample_for_preset(
    preset_name: str, network, extra_fn, **overrides,
) -> dict:
    preset = ndp.PROMPT_PRESETS[preset_name]
    n_samples = overrides.get("n_samples", DEFAULTS["n_samples"])
    gamma_scaling_distributional = overrides.get(
        "gamma_scaling_distributional", DEFAULTS["gamma_scaling_distributional"]
    )
    gamma_scaling_ndp = overrides.get("gamma_scaling_ndp", DEFAULTS["gamma_scaling_ndp"])
    flow_likelihood_variance = overrides.get(
        "flow_likelihood_variance", DEFAULTS["flow_likelihood_variance"]
    )
    ndp_likelihood_noise_std = overrides.get(
        "ndp_likelihood_noise_std", DEFAULTS["ndp_likelihood_noise_std"]
    )
    condition_ndp_part = overrides.get("condition_ndp_part", DEFAULTS["condition_ndp_part"])
    # y-range used for the *flow's* normalisation. Defaults to the preset's own.
    # Override it with the checkpoint's
    # training range when the two differ and the prior is amplitude-sensitive:
    # a change-point-trained prior min-max normalises every draw to a fixed
    # span, so driving it through a different range rescales its samples by the
    # ratio of the two. This is only the normalisation constant — the LLM's
    # y_grid stays in data units either way.
    y_range = tuple(overrides.get("y_range") or preset.y_range)

    x_grid = np.linspace(
        preset.x_range[0], preset.x_range[1], preset.n_grid, dtype=np.float64
    )
    x_context = np.asarray(preset.x_context, dtype=np.float64)
    y_context = np.asarray(preset.y_context, dtype=np.float64)

    print(f"[fig] {preset_name}: {METHOD_TITLES['ndp']}", flush=True)
    y_ndp = ndp.sample_ndp_flow_no_llm(
        network=network, x_context=x_context, y_context=y_context, x_grid=x_grid,
        x_range=preset.x_range, y_range=y_range,
        n_samples=n_samples, seed=1,
        gamma_scaling_ndp=gamma_scaling_ndp,
        likelihood_noise_std=ndp_likelihood_noise_std,
    )
    print(f"[fig] {preset_name}: {METHOD_TITLES['ndp_llm']}", flush=True)
    # `sample_ndp_flow_with_llm` reads its flow normalisation off `extra_fn`,
    # so the override has to go through the object. Restore it afterwards —
    # the heatmap panel and the metrics code read the same field and expect
    # the preset's range.
    saved_y_range = extra_fn.y_range
    extra_fn.y_range = y_range
    try:
        y_ndp_llm = ndp.sample_ndp_flow_with_llm(
            network=network, extra_fn=extra_fn, n_samples=n_samples,
            gamma_scaling_distributional=gamma_scaling_distributional,
            gamma_scaling_ndp=gamma_scaling_ndp,
            likelihood_variance=flow_likelihood_variance,
            condition_ndp_part=condition_ndp_part, seed=2,
        )
    finally:
        extra_fn.y_range = saved_y_range
    print(f"[fig] {preset_name}: {METHOD_TITLES['illmp']}", flush=True)
    y_illmp = ndp.sample_llm_marginal(extra_fn=extra_fn, n_samples=n_samples, seed=3)

    return {
        "x_grid": x_grid, "x_context": x_context, "y_context": y_context,
        "y_ndp": y_ndp, "y_ndp_llm": y_ndp_llm, "y_illmp": y_illmp,
    }


# --- Plot helpers ---------------------------------------------------------
def cached_sample_for_preset(
    preset_name: str, *, extra_fn, cache_path: Path | None = None,
    force_resample: bool = False, network=None,
    suffix: str = "",
    **overrides,
) -> dict:
    """Load samples from an npz cache, falling back to sampling on miss.

    Cache shape mirrors ``sample_for_preset`` (numpy arrays only). Useful for
    fast iteration: the slow flow sampling runs once, every subsequent call
    just reloads the cache.

    `cache_path` defaults to ``OUTPUTS_DIR/{preset}{suffix}_samples.npz``.
    Pass ``force_resample=True`` to bypass the cache. The flow network is
    loaded lazily on cache miss (so cache hits skip both sampling AND
    checkpoint loading).
    """
    if cache_path is None:
        cache_path = OUTPUTS_DIR / f"{preset_name}{suffix}_samples.npz"
    if cache_path.exists() and not force_resample:
        print(f"[fig] loading cached samples from {cache_path}", flush=True)
        with np.load(cache_path) as data:
            return {k: data[k] for k in data.files}
    if network is None:
        network = load_network(
            gamma_scaling_distributional=overrides.get("gamma_scaling_distributional"),
            condition_ndp_part=overrides.get("condition_ndp_part"),
        )
    samples = sample_for_preset(preset_name, network, extra_fn, **overrides)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **samples)
    print(f"[fig] saved samples to {cache_path}", flush=True)
    return samples
