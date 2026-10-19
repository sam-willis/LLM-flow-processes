"""Loading a trained flow checkpoint and drawing samples from it.

The three samplers go through ``FlowProcess._get_sampler``, which
``pipeline`` patches at import so the sampler knobs
(``_FLOW_SAMPLER_*``) stay settable on that module -- `run.py` assigns to
them there. The patch mutates the class, so it applies here too once that module
has been imported.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from neural_diffusion_processes.config import (
    DiffusionModelConfig,
    LLMFlowConfig2,
    TransformerConfig,
    load_model,
)
from neural_diffusion_processes.core import (
    ConditionalFlow,
    ConditioningDistributionalConvolved1D,
    ConditioningDistributionalConvolved1DAndX,
    ConditioningX,
    FlowProcess,
    VelocitySampler,
    XPoints,
    XYPoints,
    sample_n_conditional,
)

def load_ndp_flow_network_local(
    local_dir: str | Path,
    *,
    gamma_scaling_distributional: float = 1.0,
    condition_ndp_part: bool = False,
):
    """Load an NDP flow checkpoint from a local ``train.py`` output dir.

    Expects ``local_dir/config.json`` (written at training start) and one of
    ``local_dir/run.eqx`` (current default) or ``local_dir/flow.eqx`` (legacy).
    The config fully specifies the network architecture — no extra kwargs
    needed from the caller.

    Raises ``FileNotFoundError`` if config.json or the weights file is absent.
    """
    local_dir = Path(local_dir)
    config_path = local_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. `train.py` writes "
            f"this at the start of training; for runs launched before that "
            f"feature existed, create it manually from your CLI args."
        )
    with open(config_path) as f:
        cfg = json.load(f)

    noise_model = TransformerConfig(
        num_layers=cfg["n_layers"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        input_dim=cfg.get("input_dim", 1),
        output_dim=cfg.get("output_dim", 1),
        intermediate_size=cfg["intermediate_size"],
        dropout_rate=cfg.get("dropout_rate", 0.0),
        attention_dropout_rate=cfg.get("attention_dropout_rate", 0.0),
        t_sin_features=cfg["t_sin_features"],
        t_mlp_hidden_dim=cfg["t_mlp_hidden_dim"],
        t_embedding_dim=cfg["t_embedding_dim"],
        xy_embedding_dim=cfg["xy_embedding_dim"],
        use_rope=cfg.get("use_rope", False),
        fourier_x_features=cfg.get("fourier_x_features", 0),
        fourier_x_base=cfg.get("fourier_x_base", 2.0),
    )
    model_config = DiffusionModelConfig(
        diffusion=LLMFlowConfig2(
            condition_ndp_part=condition_ndp_part,
            gamma_scaling_distributional=gamma_scaling_distributional,
        ),
        noise_model=noise_model,
    )

    # Prefer the newer naming; fall back to the legacy one.
    for name in ("run.eqx", "flow.eqx"):
        ckpt = local_dir / name
        if ckpt.exists():
            return load_model(model_config, str(ckpt))
    raise FileNotFoundError(
        f"No run.eqx or flow.eqx in {local_dir}"
    )


def _make_points_from_np(
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_grid: np.ndarray,
) -> tuple[XYPoints, XPoints]:
    """Build JAX `XYPoints` context + `XPoints` target from numpy inputs.

    Mask convention: 0 = active point (`filter_points` keeps `mask == 0`).
    """
    x_ctx = jnp.asarray(x_context, dtype=jnp.float32).reshape(-1, 1)
    y_ctx = jnp.asarray(y_context, dtype=jnp.float32).reshape(-1, 1)
    x_tgt = jnp.asarray(x_grid, dtype=jnp.float32).reshape(-1, 1)
    ctx_mask = jnp.zeros(x_ctx.shape[0], dtype=jnp.bool_)
    tgt_mask = jnp.zeros(x_tgt.shape[0], dtype=jnp.bool_)
    context = XYPoints(x=x_ctx, y=y_ctx, mask=ctx_mask)
    target = XPoints(x=x_tgt, mask=tgt_mask)
    return context, target


def sample_ndp_flow_no_llm(
    *,
    network,
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_grid: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    n_samples: int,
    seed: int = 1,
    gamma_scaling_ndp: float = 50.0,
    likelihood_noise_std: float = 0.001,
    chunk_size: int | None = None,
) -> np.ndarray:
    """NDP flow sampling with only the context-hard-constraint (no LLM term).

    Returns `(n_samples, n_grid)` numpy array of y values on `x_grid`.

    `chunk_size` splits the sample vmap without changing the draws (see
    `sample_ndp_flow_with_llm`); it defaults to `$NDP_SAMPLE_CHUNK`.
    """
    if chunk_size is None:
        chunk_size = int(os.environ.get("NDP_SAMPLE_CHUNK", "0")) or None
    process = FlowProcess(
        x_range=x_range, y_range=y_range, gamma_scaling_ndp=gamma_scaling_ndp,
    )
    context, target = _make_points_from_np(x_context, y_context, x_grid)
    key = jax.random.PRNGKey(int(seed))
    target_pred, _ctx_pred, _traj = process.sample_n_conditionals(
        network, target, context, key=key, n_samples=n_samples,
        likelihood_noise_std=likelihood_noise_std, chunk_size=chunk_size,
    )
    return np.asarray(target_pred.y[..., 0])


def sample_ndp_flow_with_llm(
    *,
    network,
    extra_fn: "LLMExtraScoreFn",
    n_samples: int,
    gamma_scaling_distributional: float = 1.0,
    gamma_scaling_ndp: float = 50.0,
    likelihood_variance: float = 0.01,
    condition_ndp_part: bool = False,
    seed: int = 2,
    chunk_size: int | None = None,
) -> np.ndarray:
    """NDP flow sampling conditioned on a precomputed LLM `prob_xys` grid.

    Rebuilds a `ConditioningDistributionalConvolved1D[AndX]` handler from
    the cached `extra_fn.prob_xys` (so the LLM is not re-run) and drives the
    velocity sampler.

    Peak device memory scales with `n_samples * len(extra_fn.y_grid)` — the
    handler builds a Gaussian CDF over the y-grid per sample per solver step.
    Presets with a fine y-grid overflow an 80 GB card (2515 bins x 800 samples
    asks for a single 64 GiB allocation). `chunk_size` splits the sample vmap
    without changing the draws; it defaults to `$NDP_SAMPLE_CHUNK` so a calling
    script can set it without threading a new argument through every caller.
    """
    if chunk_size is None:
        chunk_size = int(os.environ.get("NDP_SAMPLE_CHUNK", "0")) or None
    x_range = extra_fn.x_range
    y_range = extra_fn.y_range
    process = FlowProcess(x_range=x_range, y_range=y_range)

    context, target = _make_points_from_np(
        extra_fn.x_context, extra_fn.y_context, extra_fn.x_grid,
    )
    normalised_context = process.normalise_xys(context)
    normalised_target = process.normalise_xs(target)
    normalised_y_grid = process.normalise_y_value(
        jnp.asarray(extra_fn.y_grid, dtype=jnp.float32).reshape(1, -1)
    )
    prob_xys_j = jnp.asarray(extra_fn.prob_xys, dtype=jnp.float32)

    if condition_ndp_part:
        handler = ConditioningDistributionalConvolved1DAndX(
            target=normalised_target,
            context=normalised_context,
            y_grid=normalised_y_grid,
            prob_xys=prob_xys_j,
            likelihood_variance=likelihood_variance,
            gamma_scaling_distributional=gamma_scaling_distributional,
            gamma_scaling_ndp=gamma_scaling_ndp,
        )
    else:
        handler = ConditioningDistributionalConvolved1D(
            target=normalised_target,
            context=normalised_context,
            y_grid=normalised_y_grid,
            prob_xys=prob_xys_j,
            likelihood_variance=likelihood_variance,
            gamma_scaling_distributional=gamma_scaling_distributional,
        )

    conditional = ConditionalFlow(network, handler)
    sampler = process._get_sampler()

    key = jax.random.PRNGKey(int(seed))
    norm_target_pred, _ctx_pred, _traj = sample_n_conditional(
        n_samples, key, conditional, handler, sampler, chunk_size=chunk_size,
    )
    target_pred = process.denormalise_xys(norm_target_pred)
    return np.asarray(target_pred.y[..., 0])


def sample_llm_marginal(
    *,
    extra_fn: "LLMExtraScoreFn",
    n_samples: int,
    seed: int = 3,
) -> np.ndarray:
    """Draw independent samples from the LLM per-x marginal (ILLMP baseline).

    For each target x, draw y from the discrete distribution
    ``extra_fn.prob_xys[i]`` over the y-grid bins — pure LLM marginals,
    no cross-x correlation, no flow prior.

    Returns `(n_samples, n_target)` numpy array.
    """
    rng = np.random.default_rng(int(seed))
    prob_xys = np.asarray(extra_fn.prob_xys, dtype=np.float64)
    y_grid = np.asarray(extra_fn.y_grid, dtype=np.float64).reshape(-1)
    # Normalise each row defensively — rounding in `_score_y_grid_for_one_target`
    # can leave rows a hair off 1.0, which np.random.choice rejects.
    probs = prob_xys / prob_xys.sum(axis=1, keepdims=True)
    n_target = probs.shape[0]
    out = np.empty((n_samples, n_target), dtype=np.float64)
    for i in range(n_target):
        out[:, i] = rng.choice(y_grid, size=n_samples, p=probs[i])
    return out
