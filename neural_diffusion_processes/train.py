"""Trainer: flow-matching NDP on GP data from a mixture of kernels.

Samples GP data on the fly, each example from either a locally-periodic
(periodic x RBF) kernel or a pure RBF kernel with equal probability, trains
a ``FlowMatchingTransformer`` via flow matching (``core.batch_loss``), and
writes a checkpoint plus a ground-truth vs. sampled plot.

Example
-------
    uv run python neural_diffusion_processes/train.py \\
        --n-steps 10000 \\
        --plot-path outputs/periodic_flow.png \\
        --checkpoint-path outputs/periodic_flow.eqx
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys

# Cap JAX preallocation at 45% so two concurrent runs can share the GPU.
# Override via `XLA_PYTHON_CLIENT_MEM_FRACTION=<value>` in the environment.
# Must be set BEFORE the first jax import.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import click
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from tqdm import tqdm

# Persist compiled XLA to disk so re-runs skip the 30-120s compile step.
jax.config.update("jax_compilation_cache_dir", "./.jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1)

from neural_diffusion_processes.core import XPoints, XYPoints
from neural_diffusion_processes.core import FlowMatchingTransformer
from neural_diffusion_processes.core import batch_loss
from neural_diffusion_processes.core import FlowProcess


# ---------------------------------------------------------------------------
# Data: locally-periodic GP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplerCfg:
    batch_size: int
    num_points: int
    x_range: Tuple[float, float]
    period_range: Tuple[float, float]
    ls_per_range: Tuple[float, float]
    ls_rbf_range: Tuple[float, float]
    # Pure-RBF branch: lengthscale range used when a sample is drawn from the
    # RBF-only mixture component. Marginal variance at each x is `variance`
    # for both branches (k_per(0,0)=k_rbf(0,0)=1), so y-norms match.
    ls_rbf_only_range: Tuple[float, float]
    # Bernoulli prob that a given sample is drawn from the pure-RBF kernel
    # rather than the locally-periodic (periodic x RBF) kernel. 0.5 = equal mix.
    rbf_only_prob: float
    variance: float
    jitter: float
    # "stratified" = one uniform draw per equal-width stratum; min pairwise
    # spacing >= range/(2N), prevents near-singular Cholesky.
    # "uniform" = i.i.d. uniform over x_range (legacy, backward-compat).
    x_sampling: str = "stratified"
    # If either is > 0, every batch is split deterministically into two
    # branches (50% periodic / 50% rbf_only) and the kernel-relevant hparams
    # within each branch are LHS-stratified across this many equal-width
    # bins per dimension. Per-branch bin counts let the resolution match
    # each branch's parameter-space dimensionality:
    #   - periodic branch (3D: period, ls_per, ls_rbf): n_strat_bins_periodic
    #     gives n_p^3 joint cells; keep small (~4) so cells stay populated.
    #   - rbf_only branch (1D: ls_rbf_only): n_strat_bins_rbf_only — no joint
    #     sparsity penalty, can go higher (~16-32).
    # Each dimension uses an independent random permutation of cyclic bin
    # indices so per-dim marginals are exact-balanced AND joint indices are
    # decorrelated rather than collapsing to the diagonal. Within-bin offset
    # is uniform over the bin width.
    # Constraint per branch: n_strat_bins_* must be <= batch_size/2.
    # If a branch's bin count is 0 while the other is >0, the 50/50 split
    # is still forced but that branch falls back to uniform draws.
    # Both 0 = legacy Bernoulli/uniform behaviour.
    n_strat_bins_periodic: int = 0
    n_strat_bins_rbf_only: int = 0
    # Which data-generating family to train on.
    #   "mixture"     : the locally-periodic / RBF-only GP mixture above.
    #   "changepoint" : piecewise-linear with one changepoint (see
    #                   `_changepoint_one`) — the same process the
    #                   `changepoint` demo preset uses.
    #   "mixture+changepoint" : per-sample Bernoulli(`cp_prob`) choice between
    #                   the two above, on shared x. Both branches keep their
    #                   own hparam draws, so this is a genuine mixture of the
    #                   two priors rather than an interpolation of them.
    # In "changepoint" mode every kernel hparam field above is ignored.
    family: str = "mixture"
    # Span the changepoint samples are min-max normalised into. Deliberately
    # narrower than the model's y_range so the data does not sit hard against
    # the edge of normalised space: with y_range (-3.5, 3.5) and this at
    # (-3, 3), samples occupy +-6/7 of the normalised band, leaving the
    # headroom a GP draw of variance 1 would have had.
    cp_y_range: Tuple[float, float] = (-3.0, 3.0)
    # "mixture+changepoint" only: Bernoulli prob that a given sample is a
    # changepoint draw rather than a GP-mixture draw. Bernoulli (not a
    # deterministic split) on purpose — core's `uniform_grid` timestep
    # sampler ties t to batch index, so any index<->family correlation would
    # confine each family to its own slice of the diffusion trajectory.
    cp_prob: float = 0.5


def mixture_gram(
    xa: jax.Array,
    xb: jax.Array,
    period: jax.Array,
    ls_per: jax.Array,
    ls_rbf: jax.Array,
    is_rbf_only: jax.Array,
    variance: float,
) -> jax.Array:
    # xa: [Na, 1], xb: [Nb, 1] -> [Na, Nb]
    # When is_rbf_only is True the periodic factor is replaced by 1 so the
    # kernel is pure RBF with lengthscale `ls_rbf`; otherwise it's the
    # locally-periodic product k_per * k_rbf. Diagonal == variance in both
    # branches.
    d = xa[:, None, 0] - xb[None, :, 0]
    k_per = jnp.exp(-2.0 * jnp.sin(math.pi * d / period) ** 2 / (ls_per ** 2))
    k_rbf = jnp.exp(-0.5 * (d / ls_rbf) ** 2)
    k_per_eff = jnp.where(is_rbf_only, 1.0, k_per)
    return variance * k_per_eff * k_rbf


def _sample_one(
    key: jax.Array,
    x: jax.Array,
    period: jax.Array,
    ls_per: jax.Array,
    ls_rbf: jax.Array,
    is_rbf_only: jax.Array,
    variance: float,
    jitter: float,
) -> jax.Array:
    # x: [N, 1] -> y: [N, 1]
    K = mixture_gram(x, x, period, ls_per, ls_rbf, is_rbf_only, variance)
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(x.shape[0]))
    eps = jax.random.normal(key, (x.shape[0], 1))
    return L @ eps


def _changepoint_one(
    key: jax.Array, x: jax.Array, y_range: Tuple[float, float]
) -> jax.Array:
    """One piecewise-linear draw with a single changepoint. x: [N,1] -> [N,1].

    Changepoint position `t` uniform on [0.25, 0.75] in normalised x, three
    uniform corner heights, then a per-sample min-max rescale onto `y_range`.

    That trailing rescale is the one thing worth knowing about this prior: it
    makes *every* draw attain both endpoints of `y_range` exactly, so the
    marginal at the extremes is an atom rather than a tail. It is kept because
    it is what the `changepoint` demo presets and their `true_fn` already do —
    training on anything else would put the model's prior and the evaluation
    curve in different families.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    t = jax.random.uniform(k1, (), minval=0.25, maxval=0.75)
    l0 = jax.random.uniform(k2, (), minval=0.0, maxval=1.0)
    l1 = jax.random.uniform(k3, (), minval=0.0, maxval=1.0)
    r1 = jax.random.uniform(k4, (), minval=0.0, maxval=1.0)
    z = x[:, 0]  # x already normalised to [0, 1] by the caller
    t_safe = jnp.maximum(t, 1e-6)
    one_minus_t = jnp.maximum(1.0 - t, 1e-6)
    raw = jnp.where(
        z <= t,
        l0 + (l1 - l0) * (z / t_safe),
        l1 + (r1 - l1) * ((z - t) / one_minus_t),
    )
    raw01 = (raw - jnp.min(raw)) / (jnp.max(raw) - jnp.min(raw) + 1e-9)
    y = y_range[0] + raw01 * (y_range[1] - y_range[0])
    return y[:, None], jnp.stack([t, l0, l1, r1])


def _sample_changepoint_batch(
    key: jax.Array, cfg: SamplerCfg, x: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """[B,N,1] x -> ([B,N,1] y, [B,4] params)."""
    z = (x - cfg.x_range[0]) / (cfg.x_range[1] - cfg.x_range[0])
    return jax.vmap(_changepoint_one, in_axes=(0, 0, None))(
        jax.random.split(key, cfg.batch_size), z, cfg.cp_y_range
    )


def sample_batch(
    key: jax.Array, cfg: SamplerCfg, progress: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Returns (x, y, mask) with x:[B,N,1], y:[B,N,1], mask:[B,N] (bool).

    ``progress`` is a scalar in [0, 1]. At 0, each kernel-hparam range
    collapses to its max (smoothest / longest-period end of the prior);
    at 1 the full configured range is used. Upper bounds are fixed.
    """
    (k_x, k_per, k_lp, k_lr, k_lro, k_rbo, k_y, k_m, k_perm,
     k_cp, k_cp_sel) = jax.random.split(key, 11)

    def _draw_x(k: jax.Array) -> jax.Array:
        if cfg.x_sampling == "stratified":
            stratum_w = (cfg.x_range[1] - cfg.x_range[0]) / cfg.num_points
            strata = cfg.x_range[0] + jnp.arange(cfg.num_points) * stratum_w
            in_stratum = jax.random.uniform(
                k,
                (cfg.batch_size, cfg.num_points, 1),
                minval=0.0, maxval=stratum_w,
            )
            return strata[None, :, None] + in_stratum  # already sorted
        return jax.random.uniform(
            k,
            (cfg.batch_size, cfg.num_points, 1),
            minval=cfg.x_range[0],
            maxval=cfg.x_range[1],
        ).sort(axis=1)

    x = _draw_x(k_x)

    if cfg.family == "changepoint":
        # No kernel, no Cholesky, so no curriculum and no retry loop: the
        # draw cannot fail. `progress` is accepted and ignored.
        y, cp_params = _sample_changepoint_batch(k_y, cfg, x)
        n_on = jax.random.randint(
            k_m, (cfg.batch_size,),
            minval=cfg.num_points // 2, maxval=cfg.num_points + 1,
        )
        mask = jnp.arange(cfg.num_points)[None, :] >= n_on[:, None]
        # Same 4-tuple shape as the mixture branch so downstream diagnostics
        # keep working; contents are (t, l0, l1, r1) instead of kernel hparams.
        meta = tuple(cp_params[:, i] for i in range(4))
        return x, y, mask, jnp.int32(0), meta

    def _lerp_min(rng: Tuple[float, float]) -> jax.Array:
        return rng[1] + progress * (rng[0] - rng[1])

    period_min = _lerp_min(cfg.period_range)
    ls_per_min = _lerp_min(cfg.ls_per_range)
    ls_rbf_min = _lerp_min(cfg.ls_rbf_range)
    ls_rbf_only_min = _lerp_min(cfg.ls_rbf_only_range)

    n_p_bins = cfg.n_strat_bins_periodic
    n_r_bins = cfg.n_strat_bins_rbf_only
    strat_active = (n_p_bins > 0) or (n_r_bins > 0)

    if strat_active:
        # Latin-hypercube-style stratification. Split the batch into a
        # deterministic periodic half (idx 0..half-1) and rbf_only half
        # (idx half..B-1), ignoring cfg.rbf_only_prob. Per-branch bin counts
        # match each branch's parameter-space dimensionality (3D periodic
        # uses fewer bins to keep joint cells populated; 1D rbf_only can go
        # higher freely). Each dim's bin assignment is a random permutation
        # of cyclic indices so per-dim marginals are exact-balanced AND
        # joint indices are decorrelated rather than diagonal-collapsed.
        # Within-bin offset is uniform over bin width.
        # rbf_only branch's period / ls_per / ls_rbf_periodic are kernel-
        # irrelevant (k_per_eff = 1, ls_rbf is overridden by jnp.where below)
        # so they get plain uniform fillers regardless of bin counts.
        half = cfg.batch_size // 2
        n_rbf = cfg.batch_size - half
        is_rbf_only = jnp.arange(cfg.batch_size) >= half

        def _stratify(key: jax.Array, n_total: int, lo: jax.Array,
                      hi: jax.Array, n_bins: int) -> jax.Array:
            perm_key, off_key = jax.random.split(key)
            bw = (hi - lo) / n_bins
            base = jnp.arange(n_total) % n_bins
            bin_idx = jax.random.permutation(perm_key, base)
            offset = jax.random.uniform(off_key, (n_total,), minval=0.0, maxval=bw)
            return lo + bin_idx * bw + offset

        # ---- periodic half ----
        if n_p_bins > 0:
            period_per = _stratify(
                k_per, half, period_min, cfg.period_range[1], n_p_bins,
            )
            ls_per_per = _stratify(
                k_lp, half, ls_per_min, cfg.ls_per_range[1], n_p_bins,
            )
            ls_rbf_per = _stratify(
                k_lr, half, ls_rbf_min, cfg.ls_rbf_range[1], n_p_bins,
            )
        else:
            # Force 50/50 branch but keep periodic-half hparams uniform.
            k_unif = jax.random.split(k_per, 3)
            period_per = jax.random.uniform(
                k_unif[0], (half,),
                minval=period_min, maxval=cfg.period_range[1],
            )
            ls_per_per = jax.random.uniform(
                k_unif[1], (half,),
                minval=ls_per_min, maxval=cfg.ls_per_range[1],
            )
            ls_rbf_per = jax.random.uniform(
                k_unif[2], (half,),
                minval=ls_rbf_min, maxval=cfg.ls_rbf_range[1],
            )

        # ---- rbf_only half ----
        if n_r_bins > 0:
            ls_rbf_only_rbf = _stratify(
                k_lro, n_rbf, ls_rbf_only_min,
                cfg.ls_rbf_only_range[1], n_r_bins,
            )
        else:
            ls_rbf_only_rbf = jax.random.uniform(
                k_lro, (n_rbf,),
                minval=ls_rbf_only_min, maxval=cfg.ls_rbf_only_range[1],
            )

        # ---- fillers for kernel-irrelevant slots ----
        k_fill = jax.random.split(k_rbo, 4)
        period_fill = jax.random.uniform(
            k_fill[0], (n_rbf,), minval=period_min, maxval=cfg.period_range[1],
        )
        ls_per_fill = jax.random.uniform(
            k_fill[1], (n_rbf,), minval=ls_per_min, maxval=cfg.ls_per_range[1],
        )
        ls_rbf_periodic_fill = jax.random.uniform(
            k_fill[2], (n_rbf,), minval=ls_rbf_min, maxval=cfg.ls_rbf_range[1],
        )
        ls_rbf_only_fill = jax.random.uniform(
            k_fill[3], (half,),
            minval=ls_rbf_only_min, maxval=cfg.ls_rbf_only_range[1],
        )

        period = jnp.concatenate([period_per, period_fill])
        ls_per = jnp.concatenate([ls_per_per, ls_per_fill])
        ls_rbf_periodic = jnp.concatenate([ls_rbf_per, ls_rbf_periodic_fill])
        ls_rbf_only = jnp.concatenate([ls_rbf_only_fill, ls_rbf_only_rbf])

        # Decouple batch index from branch / hparam bins. Without this, the
        # deterministic [periodic_half, rbf_only_half] concatenation locks
        # branch and bin to batch index — and core.batch_loss's default
        # `uniform_grid` timestep sampler uses `t[i] = i/B + dt`, so every
        # batch would land periodic samples in t in [0, 0.5] and rbf_only
        # in t in [0.5, 1.0]. The model would never see either branch
        # outside its half of the diffusion trajectory. A single full-batch
        # permutation breaks every such index<->property correlation.
        perm = jax.random.permutation(k_perm, cfg.batch_size)
        period = period[perm]
        ls_per = ls_per[perm]
        ls_rbf_periodic = ls_rbf_periodic[perm]
        ls_rbf_only = ls_rbf_only[perm]
        is_rbf_only = is_rbf_only[perm]
    else:
        period = jax.random.uniform(
            k_per, (cfg.batch_size,), minval=period_min, maxval=cfg.period_range[1],
        )
        ls_per = jax.random.uniform(
            k_lp, (cfg.batch_size,), minval=ls_per_min, maxval=cfg.ls_per_range[1],
        )
        ls_rbf_periodic = jax.random.uniform(
            k_lr, (cfg.batch_size,), minval=ls_rbf_min, maxval=cfg.ls_rbf_range[1],
        )
        ls_rbf_only = jax.random.uniform(
            k_lro, (cfg.batch_size,),
            minval=ls_rbf_only_min, maxval=cfg.ls_rbf_only_range[1],
        )
        is_rbf_only = jax.random.bernoulli(
            k_rbo, p=cfg.rbf_only_prob, shape=(cfg.batch_size,),
        )
    ls_rbf = jnp.where(is_rbf_only, ls_rbf_only, ls_rbf_periodic)

    def _draw_ys(x_in: jax.Array, k: jax.Array) -> jax.Array:
        return jax.vmap(_sample_one, in_axes=(0, 0, 0, 0, 0, 0, None, None))(
            jax.random.split(k, cfg.batch_size),
            x_in,
            period,
            ls_per,
            ls_rbf,
            is_rbf_only,
            cfg.variance,
            cfg.jitter,
        )

    def _draw_xy(k: jax.Array):
        kx, ky = jax.random.split(k)
        x_new = _draw_x(kx)
        return x_new, _draw_ys(x_new, ky)

    # Cholesky on near-singular K can NaN when x has near-duplicate points.
    # Keep kernel hparams fixed (deterministic curriculum) but resample x and
    # the GP noise together up to `max_retries` times. Bounded so the JIT
    # trace stays static.
    max_retries = 4

    def _cond(state):
        attempt, _key, _x, y = state
        return jnp.logical_and(jnp.any(jnp.isnan(y)), attempt < max_retries)

    def _body(state):
        attempt, key, _x, _y = state
        new_key, sub = jax.random.split(key)
        x_new, y_new = _draw_xy(sub)
        return attempt + 1, new_key, x_new, y_new

    y_init = _draw_ys(x, k_y)
    n_retries, _, x, y = jax.lax.while_loop(
        _cond, _body, (jnp.int32(0), k_y, x, y_init)
    )

    if cfg.family == "mixture+changepoint":
        # Draw a changepoint y for every sample on the x the GP branch settled
        # on (post-retry, so both branches see identical x), then pick per
        # sample. The redundant changepoint draws are near-free — the batch's
        # cost is the [B, N, N] Cholesky above, which runs either way — and
        # doing it this way keeps the whole thing one static JIT trace.
        y_cp, _cp_params = _sample_changepoint_batch(k_cp, cfg, x)
        is_cp = jax.random.bernoulli(k_cp_sel, p=cfg.cp_prob, shape=(cfg.batch_size,))
        y = jnp.where(is_cp[:, None, None], y_cp, y)

    # Random mask: 50-100% of points active per sample. Active = False (==0).
    n_on = jax.random.randint(
        k_m,
        (cfg.batch_size,),
        minval=cfg.num_points // 2,
        maxval=cfg.num_points + 1,
    )
    mask = jnp.arange(cfg.num_points)[None, :] >= n_on[:, None]
    # meta exposes per-sample kernel hparams + branch flag for diagnostics
    # (e.g. per-branch / per-bin gradient-norm logging). All [B] arrays. Under
    # "mixture+changepoint" these describe the GP draw that was made for each
    # slot, which for the changepoint-selected slots was discarded — so filter
    # by family before trusting them there.
    meta = (period, ls_per, ls_rbf, is_rbf_only)
    return x, y, mask, n_retries, meta


# ---------------------------------------------------------------------------
# Normalisation (matches ``Process.normalise_*_value`` math)
# ---------------------------------------------------------------------------


def norm_x(x: jax.Array, r: Tuple[float, float]) -> jax.Array:
    return ((x - r[0]) / (r[1] - r[0]) - 0.5) * 4.0


def norm_y(y: jax.Array, r: Tuple[float, float]) -> jax.Array:
    return ((y - r[0]) / (r[1] - r[0]) - 0.5) * 2.0


# ---------------------------------------------------------------------------
# Train step (JITted)
# ---------------------------------------------------------------------------


def _ema_update(smooth: eqx.Module, network: eqx.Module, decay: float) -> eqx.Module:
    sP, _ = eqx.partition(smooth, eqx.is_array)
    nP, nS = eqx.partition(network, eqx.is_array)
    newP = jax.tree.map(lambda s, n: decay * s + (1.0 - decay) * n, sP, nP)
    return eqx.combine(newP, nS)


def make_train_chunk(sampler_cfg: SamplerCfg, y_range: Tuple[float, float], optim, ema_decay: float):
    """Data-parallel pmap'd training chunk with `lax.scan` inner loop.

    On 1 device, behaves like the previous JIT path (pmean is a no-op). On N>1
    devices, gradients are all-reduced via pmean each inner step so each
    device gets the average gradient over the full effective batch.

    Caller must pass replicated `network`/`smooth_network`/`opt_state` (leading
    axis = n_devices) and `chunk_keys`/`chunk_progress` shaped
    `(n_devices, chunk_size, ...)` so the leading axis maps across devices.
    """

    @eqx.filter_pmap(axis_name="device")
    def train_chunk(network, smooth_network, opt_state, chunk_keys, chunk_progress):
        net_params, net_static = eqx.partition(network, eqx.is_array)
        smooth_params, smooth_static = eqx.partition(smooth_network, eqx.is_array)

        def _one_step(carry, inputs):
            net_params, smooth_params, opt_state = carry
            key, progress = inputs
            network = eqx.combine(net_params, net_static)
            smooth_network = eqx.combine(smooth_params, smooth_static)
            data_key, loss_key = jax.random.split(key)
            x_raw, y_raw, mask, n_retries, _meta = sample_batch(data_key, sampler_cfg, progress)
            x_n = norm_x(x_raw, sampler_cfg.x_range)
            y_n = norm_y(y_raw, y_range)
            pts = XYPoints(x=x_n, y=y_n, mask=mask)
            loss, grads = eqx.filter_value_and_grad(batch_loss)(network, pts, loss_key)
            grads = jax.lax.pmean(grads, axis_name="device")
            loss = jax.lax.pmean(loss, axis_name="device")
            # Norms are computed pre-update on the post-pmean grads and the
            # current params; cheap, and these are the canonical pre-NaN signals.
            grad_norm = optax.global_norm(grads)
            param_norm = optax.global_norm(net_params)
            updates, opt_state = optim.update(grads, opt_state, net_params)
            network = eqx.apply_updates(network, updates)
            smooth_network = _ema_update(smooth_network, network, ema_decay)
            net_params, _ = eqx.partition(network, eqx.is_array)
            smooth_params, _ = eqx.partition(smooth_network, eqx.is_array)
            metrics = (loss, grad_norm, param_norm, n_retries)
            return (net_params, smooth_params, opt_state), metrics

        init = (net_params, smooth_params, opt_state)
        (net_params, smooth_params, opt_state), chunk_metrics = jax.lax.scan(
            _one_step, init, (chunk_keys, chunk_progress)
        )
        chunk_losses, chunk_grad_norm, chunk_param_norm, chunk_retries = chunk_metrics
        network = eqx.combine(net_params, net_static)
        smooth_network = eqx.combine(smooth_params, smooth_static)
        return (network, smooth_network, opt_state,
                chunk_losses, chunk_grad_norm, chunk_param_norm, chunk_retries)

    return train_chunk


def _replicate(tree):
    """Replicate the array leaves of a pytree across all local devices."""
    arr, static = eqx.partition(tree, eqx.is_array)
    arr = jax.device_put_replicated(arr, jax.local_devices())
    return eqx.combine(arr, static)


def _take_first_device(tree):
    """Inverse of `_replicate`: take device-0 slice for host-side use."""
    arr, static = eqx.partition(tree, eqx.is_array)
    arr = jax.tree.map(lambda x: x[0], arr)
    return eqx.combine(arr, static)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def build_network(
    key: jax.Array,
    hidden_dim: int,
    n_layers: int,
    num_heads: int,
    intermediate_size: int,
    xy_embedding_dim: int,
    t_sin_features: int,
    t_mlp_hidden_dim: int,
    t_embedding_dim: int,
    dropout_rate: float,
    attention_dropout_rate: float,
    use_rope: bool,
    fourier_x_features: int,
    fourier_x_base: float,
) -> FlowMatchingTransformer:
    return FlowMatchingTransformer(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_heads=num_heads,
        input_dim=1,
        output_dim=1,
        intermediate_size=intermediate_size,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        t_embedding_dim=t_embedding_dim,
        t_sin_features=t_sin_features,
        t_mlp_hidden_dim=t_mlp_hidden_dim,
        xy_embedding_dim=xy_embedding_dim,
        use_rope=use_rope,
        fourier_x_features=fourier_x_features,
        fourier_x_base=fourier_x_base,
        key=key,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _sample_gt(
    key: jax.Array,
    n: int,
    cfg: SamplerCfg,
    progress: float = 1.0,
    rbf_only_prob_override: float | None = None,
    cp_prob_override: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw `n` fresh GP samples. ``progress`` matches training curriculum (default 1=full range).

    ``rbf_only_prob_override`` lets the caller force the mixture branch:
    ``0.0`` -> only locally-periodic samples, ``1.0`` -> only RBF-only samples.
    Used by the plot to render a fixed split of each kernel type.

    ``cp_prob_override`` does the same for the "mixture+changepoint" family's
    outer branch: ``0.0`` -> only GP-mixture samples, ``1.0`` -> only
    changepoint samples. Ignored by the single-family modes.
    """
    rbf_p = cfg.rbf_only_prob if rbf_only_prob_override is None else rbf_only_prob_override
    cp_p = cfg.cp_prob if cp_prob_override is None else cp_prob_override
    sub_cfg = SamplerCfg(
        batch_size=n,
        num_points=cfg.num_points,
        x_range=cfg.x_range,
        period_range=cfg.period_range,
        ls_per_range=cfg.ls_per_range,
        ls_rbf_range=cfg.ls_rbf_range,
        ls_rbf_only_range=cfg.ls_rbf_only_range,
        rbf_only_prob=rbf_p,
        variance=cfg.variance,
        jitter=cfg.jitter,
        x_sampling=cfg.x_sampling,
        family=cfg.family,
        cp_y_range=cfg.cp_y_range,
        cp_prob=cp_p,
    )
    x, y, _, _, _ = sample_batch(key, sub_cfg, jnp.asarray(progress, dtype=jnp.float32))
    return np.asarray(x), np.asarray(y)


def make_plot(
    plot_path: Path,
    network: eqx.Module,
    sampler_cfg: SamplerCfg,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    loss_history: np.ndarray,
    wall_clock_s: np.ndarray,
    key: jax.Array,
    gt_key: jax.Array,
    n_examples: int = 4,
    n_grid: int = 128,
    resume_steps=None,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    process = FlowProcess(x_range=x_range, y_range=y_range, gamma_scaling_ndp=50.0)

    k_uncond, k_cond = jax.random.split(key, 2)
    k_gt_per, k_gt_rbf, k_cond_gt = jax.random.split(gt_key, 3)

    # Row 1: ground truth, split across the training family's branches so the
    # plot self-documents what the model is being fit to. Each segment is one
    # forced draw: rbf_only_prob_override picks the kernel branch (0.0
    # locally-periodic / 1.0 pure-RBF) and cp_prob_override picks the outer
    # family branch under "mixture+changepoint" (0.0 GP / 1.0 changepoint).
    # With n_examples=4: mixture -> 2 periodic + 2 rbf; changepoint -> 4
    # changepoint; mixture+changepoint -> 1 periodic + 1 rbf + 2 changepoint.
    if sampler_cfg.family == "changepoint":
        segments = [(n_examples, "changepoint", 0.0, None)]
    elif sampler_cfg.family == "mixture+changepoint":
        n_gp = n_examples // 2
        segments = [
            (n_gp - n_gp // 2, "periodic", 0.0, 0.0),
            (n_gp // 2, "rbf", 1.0, 0.0),
            (n_examples - n_gp, "changepoint", 0.0, 1.0),
        ]
    else:
        segments = [
            (n_examples - n_examples // 2, "periodic", 0.0, None),
            (n_examples // 2, "rbf", 1.0, None),
        ]
    # Two-segment families keep their original keys so their GT rows stay
    # byte-identical to previously saved plots; the 3-segment one needs more.
    gt_keys = (list(jax.random.split(k_gt_per, len(segments)))
               if len(segments) > 2 else [k_gt_per, k_gt_rbf])
    gt_x_parts, gt_y_parts, gt_kind = [], [], []
    for (n_seg, kind, rbf_p, cp_p), seg_key in zip(segments, gt_keys):
        if n_seg <= 0:
            continue
        sx, sy = _sample_gt(
            seg_key, n_seg, sampler_cfg,
            rbf_only_prob_override=rbf_p, cp_prob_override=cp_p,
        )
        gt_x_parts.append(sx)
        gt_y_parts.append(sy)
        gt_kind.extend([kind] * n_seg)
    gt_x = np.concatenate(gt_x_parts, axis=0)
    gt_y = np.concatenate(gt_y_parts, axis=0)

    # Row 2: unconditional samples
    x_grid = jnp.linspace(x_range[0], x_range[1], n_grid).reshape(-1, 1)
    target = XPoints(x=x_grid, mask=jnp.zeros(n_grid, dtype=bool))
    uncond_y_list = []
    for ki in jax.random.split(k_uncond, n_examples):
        y_pts, _ = process.sample(network, target, ki)
        uncond_y_list.append(np.asarray(y_pts.y))
    uncond_y = np.stack(uncond_y_list, axis=0)  # [n_examples, n_grid, 1]
    x_grid_np = np.asarray(x_grid)

    # Row 3: conditional samples. Build context from one fresh periodic GT
    # sample (the harder regime — RBF context is approximately handled by any
    # GP-style model, periodic is the discriminating test). Under
    # "mixture+changepoint" the changepoint branch is the newly-added
    # capability and the one worth watching, so the context comes from there
    # instead.
    cond_kind = ("changepoint" if sampler_cfg.family == "mixture+changepoint"
                 else "periodic")
    ctx_x_all, ctx_y_all = _sample_gt(
        k_cond_gt, 1, sampler_cfg, rbf_only_prob_override=0.0,
        cp_prob_override=(1.0 if cond_kind == "changepoint" else None),
    )
    # Pick 5 sorted context points spread across x_range.
    sel = np.linspace(0, sampler_cfg.num_points - 1, 5, dtype=int)
    ctx_x = ctx_x_all[0, sel]  # [5, 1]
    ctx_y = ctx_y_all[0, sel]  # [5, 1]
    context = XYPoints(
        x=jnp.asarray(ctx_x),
        y=jnp.asarray(ctx_y),
        mask=jnp.zeros(ctx_x.shape[0], dtype=bool),
    )
    cond_target_xy, cond_context_xy, _ = process.sample_n_conditionals(
        model=network,
        target=target,
        context=context,
        key=k_cond,
        n_samples=n_examples,
        likelihood_noise_std=0.001,
    )
    cond_y = np.asarray(cond_target_xy.y)  # [n_samples, n_grid, 1]
    cond_x = np.asarray(cond_target_xy.x)  # [n_samples, n_grid, 1]

    # Layout
    fig, axes = plt.subplots(4, n_examples, figsize=(3.2 * n_examples, 10.0))

    for i in range(n_examples):
        ax = axes[0, i]
        ax.plot(gt_x[i, :, 0], gt_y[i, :, 0], color="C0", lw=1.2)
        ax.set_ylim(y_range)
        ax.set_xlim(x_range)
        if i == 0:
            ax.set_ylabel("ground truth")
        ax.set_title(f"GT #{i} ({gt_kind[i]})")

    for i in range(n_examples):
        ax = axes[1, i]
        ax.plot(x_grid_np[:, 0], uncond_y[i, :, 0], color="C1", lw=1.2)
        ax.set_ylim(y_range)
        ax.set_xlim(x_range)
        if i == 0:
            ax.set_ylabel("unconditional")
        ax.set_title(f"uncond #{i}")

    for i in range(n_examples):
        ax = axes[2, i]
        ax.plot(cond_x[i, :, 0], cond_y[i, :, 0], color="C2", lw=1.2)
        ax.scatter(ctx_x[:, 0], ctx_y[:, 0], color="black", s=24, zorder=3, label="context")
        ax.set_ylim(y_range)
        ax.set_xlim(x_range)
        if i == 0:
            ax.set_ylabel("conditional")
            ax.legend(loc="upper right", fontsize=7)
        ax.set_title(f"cond #{i} ({cond_kind} ctx)")

    # Loss curves span the last row: left half vs step, right half vs wall-clock.
    gs = axes[3, 0].get_gridspec()
    for a in axes[3, :]:
        a.remove()
    half = n_examples // 2
    ax_step = fig.add_subplot(gs[3, :half])
    ax_time = fig.add_subplot(gs[3, half:])
    # Cap y at the initial loss to hide training-dynamics outliers (Adam-spike
    # bumps to ~1e16 etc.) without flattening the useful descent.
    initial_loss = float(np.nanmax(loss_history[: min(10, len(loss_history))]))
    ymax = max(initial_loss, float(np.nanmax(loss_history[: min(200, len(loss_history))])))
    finite_tail = loss_history[np.isfinite(loss_history)]
    ymin = float(np.nanmin(finite_tail)) if finite_tail.size else 1e-3
    ymin = max(ymin * 0.9, 1e-4)

    steps = np.arange(1, len(loss_history) + 1)  # +1 so step 0 is representable on log x-axis
    ax_step.plot(steps, loss_history, color="C3", lw=1.0)
    ax_step.set_xscale("log")
    ax_step.set_yscale("log")
    ax_step.set_ylim(ymin, ymax)
    ax_step.set_xlabel("step (log)")
    ax_step.set_ylabel("loss (log)")
    ax_step.set_title("loss vs step")
    # Shift wall-clock by +1 s so very-early log-axis points don't collapse.
    ax_time.plot(wall_clock_s + 1.0, loss_history, color="C3", lw=1.0)
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_ylim(ymin, ymax)
    ax_time.set_xlabel("wall-clock +1 s (log)")
    ax_time.set_title("loss vs wall-clock")

    # Annotate resume/warm-start boundaries on both loss panels.
    if resume_steps:
        for i, rs in enumerate(resume_steps):
            if rs <= 0 or rs > len(loss_history):
                continue
            label = "resume" if i == 0 else None
            ax_step.axvline(rs, color="gray", linestyle="--", lw=0.8, alpha=0.7, label=label)
            wc_at_rs = float(wall_clock_s[rs - 1])
            ax_time.axvline(wc_at_rs + 1.0, color="gray", linestyle="--", lw=0.8, alpha=0.7, label=label)
        ax_step.legend(loc="lower left", fontsize=7)
        ax_time.legend(loc="lower left", fontsize=7)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    # Persist raw arrays next to the PNG for post-hoc re-plotting.
    npz_path = plot_path.with_suffix(".npz")
    np.savez(
        npz_path,
        gt_x=gt_x,
        gt_y=gt_y,
        x_grid=x_grid_np,
        uncond_y=uncond_y,
        cond_x=cond_x,
        cond_y=cond_y,
        ctx_x=np.asarray(ctx_x),
        ctx_y=np.asarray(ctx_y),
        loss_history=np.asarray(loss_history),
        wall_clock_s=np.asarray(wall_clock_s),
        x_range=np.asarray(x_range),
        y_range=np.asarray(y_range),
        resume_steps=np.asarray(resume_steps or [], dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Run config (metadata)
# ---------------------------------------------------------------------------


# Architecture kwargs that must match byte-for-byte to re-load a checkpoint.
# On --resume-from, these are overridden from the saved config.json if it
# exists (and we warn when the CLI disagrees). Non-architecture config is
# written out purely for documentation / downstream loaders.
_ARCH_KEYS = (
    "hidden_dim", "n_layers", "num_heads", "intermediate_size",
    "xy_embedding_dim", "t_sin_features", "t_mlp_hidden_dim",
    "t_embedding_dim", "use_rope", "fourier_x_features", "fourier_x_base",
    "dropout_rate", "attention_dropout_rate",
)


def _write_config_json(config_path: Path, cfg: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def _load_config_json(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def _save_resume_state(
    ckpt_dir: Path,
    network: eqx.Module,
    smooth_network: eqx.Module,
    opt_state,
    key: jax.Array,
    step: int,
    losses: np.ndarray,
    wall_clock_s: np.ndarray,
    resume_steps,
) -> None:
    """Atomically write resume.eqx + resume.npz so a kill mid-write is safe."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = (network, smooth_network, opt_state, key)
    tmp_eqx = ckpt_dir / "resume.eqx.tmp"
    eqx.tree_serialise_leaves(str(tmp_eqx), state)
    os.replace(str(tmp_eqx), str(ckpt_dir / "resume.eqx"))
    # numpy.savez auto-appends .npz, so use a name that already ends in .npz.
    tmp_npz = ckpt_dir / "resume.tmp.npz"
    np.savez(
        str(tmp_npz),
        step=np.int64(step),
        losses=losses[:step].astype(np.float32),
        wall_clock_s=wall_clock_s[:step].astype(np.float32),
        resume_steps=np.asarray(resume_steps, dtype=np.int64),
    )
    os.replace(str(tmp_npz), str(ckpt_dir / "resume.npz"))


def _load_resume_state(resume_path: Path, template_state):
    """Load a resume checkpoint. Two supported modes:

    (1) Full resume: `resume_path/resume.eqx` + `resume_path/resume.npz` exist.
        Restores network, smooth_network, opt_state, key, step, history exactly.

    (2) Warm-start from a finished run: `resume_path/flow.eqx` exists (main
        training's end-of-run checkpoint, contains smooth_network only). Loads
        those weights into both network and smooth_network, with a fresh
        opt_state and key. If `resume_path/run.npz` is also present, its
        loss/wall-clock history is kept for plot continuity and `step` is set
        to its length.

    Returns: (network, smooth_network, opt_state, key, step, hist_losses,
              hist_wall_clock, prior_resume_steps, mode) where mode is "full"
              or "warm", and prior_resume_steps is the list of step indices
              where previous resumes occurred (for plot annotation).
    """
    resume_path = Path(resume_path)
    template_net, template_smooth, template_opt, template_key = template_state

    full_eqx = resume_path / "resume.eqx"
    full_npz = resume_path / "resume.npz"
    if full_eqx.exists() and full_npz.exists():
        state = eqx.tree_deserialise_leaves(str(full_eqx), template_state)
        d = np.load(str(full_npz))
        step = int(d["step"])
        prior_resume_steps = list(map(int, d["resume_steps"])) if "resume_steps" in d.files else []
        return (*state, step, np.asarray(d["losses"]), np.asarray(d["wall_clock_s"]),
                prior_resume_steps, "full")

    # Accept either the old (`flow.eqx`) or new (`run.eqx`) checkpoint name.
    candidates = [resume_path / "run.eqx", resume_path / "flow.eqx"]
    weights_eqx = next((p for p in candidates if p.exists()), None)
    if weights_eqx is not None:
        smooth_network = eqx.tree_deserialise_leaves(str(weights_eqx), template_smooth)
        hist_losses = np.empty(0, dtype=np.float32)
        hist_wall = np.empty(0, dtype=np.float32)
        prior_resume_steps: list = []
        run_npz = resume_path / "run.npz"
        if run_npz.exists():
            d = np.load(str(run_npz))
            hist_losses = np.asarray(d["loss_history"]).astype(np.float32)
            hist_wall = np.asarray(d["wall_clock_s"]).astype(np.float32)
            if "resume_steps" in d.files:
                prior_resume_steps = list(map(int, d["resume_steps"]))
        step = len(hist_losses)
        return (
            smooth_network, smooth_network, template_opt, template_key,
            step, hist_losses, hist_wall, prior_resume_steps, "warm",
        )

    raise click.UsageError(
        f"--resume-from: no resume.eqx/resume.npz or run.eqx/flow.eqx found in {resume_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--n-steps", type=int, default=300_000, show_default=True)
@click.option("--batch-size", type=int, default=2046, show_default=True,
              help="Must be divisible by the number of local devices: the 2046 default\n"
                   "shards across 1, 2, 3 or 6 devices, not 4.")
@click.option("--num-points", type=int, default=256, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
# data
@click.option("--x-range", type=float, nargs=2, default=(-2.0, 2.0), show_default=True)
@click.option("--period-range", type=float, nargs=2, default=(0.75, 1.25), show_default=True)
@click.option("--ls-per-range", type=float, nargs=2, default=(0.5, 1.0), show_default=True)
@click.option("--ls-rbf-range", type=float, nargs=2, default=(1.0, 4.0), show_default=True)
@click.option("--ls-rbf-only-range", type=float, nargs=2, default=(0.1, 1.0), show_default=True,
              help="Lengthscale range for the pure-RBF mixture branch.")
@click.option("--rbf-only-prob", type=float, default=0.5, show_default=True,
              help="Bernoulli mixture weight: probability that any given training "
                   "sample is drawn from the pure-RBF kernel rather than the "
                   "locally-periodic (periodic x RBF) kernel. 0.0 disables the RBF "
                   "branch (back to periodic-only); 1.0 is RBF-only.")
@click.option("--family",
              type=click.Choice(["mixture", "changepoint", "mixture+changepoint"]),
              default="mixture", show_default=True,
              help="Data-generating family. 'mixture' = the locally-periodic / "
                   "RBF-only GP mixture. 'changepoint' = piecewise-linear with "
                   "one changepoint (the same process as the changepoint demo "
                   "presets); "
                   "all kernel-hparam options are ignored in that mode. "
                   "'mixture+changepoint' = per-sample Bernoulli(--cp-prob) "
                   "draw from one or the other, both at their full hparam "
                   "ranges.")
@click.option("--cp-prob", type=float, default=0.5, show_default=True,
              help="'mixture+changepoint' only: probability that a given "
                   "training sample is a changepoint draw rather than a "
                   "GP-mixture draw. 0.0 is plain --family mixture, 1.0 is "
                   "plain --family changepoint. Ignored by the other families.")
@click.option("--cp-y-range", type=float, nargs=2, default=(-3.0, 3.0),
              show_default=True,
              help="Span each changepoint sample is min-max normalised into. "
                   "Keep it inside --y-range so the data has headroom in "
                   "normalised space.")
@click.option("--variance", type=float, default=1.0, show_default=True)
@click.option("--y-range", type=float, nargs=2, default=(-3.5, 3.5), show_default=True)
@click.option("--jitter", type=float, default=1e-5, show_default=True)
@click.option("--x-sampling",
              type=click.Choice(["stratified", "uniform"], case_sensitive=False),
              default="stratified", show_default=True,
              help="x-position sampling: 'stratified' = one draw per equal-width "
                   "stratum (Sobol-like, prevents near-singular Cholesky); "
                   "'uniform' = legacy i.i.d. uniform.")
@click.option("--n-strat-bins-periodic", type=int, default=0, show_default=True,
              help="Bins per dimension for the periodic branch's stratified "
                   "kernel hparams (period, ls_per, ls_rbf). With 3 stratified "
                   "dims this gives n_p^3 joint cells; keep small (~4) so each "
                   "cell stays populated. 0 = uniform draws within periodic "
                   "branch (50/50 split still forced if --n-strat-bins-rbf-only "
                   ">0). Required: n_strat_bins_periodic <= batch_size/2.")
@click.option("--n-strat-bins-rbf-only", type=int, default=0, show_default=True,
              help="Bins for the rbf_only branch's stratified ls_rbf_only. 1D "
                   "axis -> no joint sparsity penalty, can go higher (~16-32). "
                   "0 = uniform draws within rbf_only branch (50/50 split "
                   "still forced if --n-strat-bins-periodic >0). Required: "
                   "n_strat_bins_rbf_only <= batch_size/2. If both this and "
                   "--n-strat-bins-periodic are 0 the legacy Bernoulli/uniform "
                   "sampler is used; --rbf-only-prob is ignored when either >0.")
# optimizer
@click.option("--lr", type=float, default=3e-4, show_default=True)
@click.option("--ema-decay", type=float, default=0.999, show_default=True)
@click.option("--grad-clip", type=float, default=1.0, show_default=True)
@click.option("--lr-decay-steps", type=int, default=0, show_default=True,
              help="If >0, use cosine LR decay over this many total steps "
                   "(end value = 1%% of init lr). 0 = constant LR.")
@click.option("--lr-warmup-steps", type=int, default=0, show_default=True,
              help="Linear warmup steps from 0 -> lr at training start. "
                   "Only effective when --lr-decay-steps > 0.")
# model
@click.option("--hidden-dim", type=int, default=192, show_default=True)
@click.option("--n-layers", type=int, default=12, show_default=True)
@click.option("--num-heads", type=int, default=4, show_default=True)
@click.option("--intermediate-size", type=int, default=768, show_default=True)
@click.option("--xy-embedding-dim", type=int, default=768, show_default=True)
@click.option("--t-sin-features", type=int, default=192, show_default=True)
@click.option("--t-mlp-hidden-dim", type=int, default=768, show_default=True)
@click.option("--t-embedding-dim", type=int, default=64, show_default=True)
@click.option("--dropout-rate", type=float, default=0.0, show_default=True)
@click.option("--attention-dropout-rate", type=float, default=0.0, show_default=True)
@click.option("--use-rope/--no-rope", default=False, show_default=True)
@click.option("--fourier-x-features", type=int, default=16, show_default=True,
              help="If K>0, expand x into [sin(B^0 x), cos(B^0 x), ..., sin(B^{K-1} x), cos(B^{K-1} x)] "
                   "per input dim before the linear token embedding. 0 disables.")
@click.option("--fourier-x-base", type=float, default=2.0, show_default=True,
              help="Geometric base for Fourier feature frequencies.")
# output
@click.option("--checkpoint-path", type=click.Path(path_type=Path),
              default=Path("outputs/periodic_run.eqx"), show_default=True)
@click.option("--plot-path", type=click.Path(path_type=Path),
              default=Path("outputs/periodic_run.png"), show_default=True)
@click.option("--log-every", type=int, default=100, show_default=True)
@click.option("--plot-every", type=int, default=2000, show_default=True,
              help="Save an intermediate plot every N steps (set 0 to disable).")
@click.option("--plot-log10/--no-plot-log10", default=False, show_default=True,
              help="If set, ignore --plot-every and instead plot at log-spaced "
                   "milestones (1k, 2.5k, 5k, 10k, 25k, 50k, 100k, ...). All "
                   "milestones must be multiples of --chunk-size.")
@click.option("--curriculum-steps", type=int, default=0, show_default=True,
              help="Linear ramp steps for kernel-hparam curriculum. 0 = disabled (full range from step 0). "
                   "Progress interpolates each range's min from max (smooth) to configured min over N steps.")
@click.option("--chunk-size", type=int, default=50, show_default=True,
              help="Inner steps per JIT call via lax.scan. Bigger = less Python/JAX dispatch overhead, "
                   "coarser per-step logging. Must divide n_steps.")
@click.option("--checkpoint-every", type=int, default=0, show_default=True,
              help="Save full resume state (resume.eqx + resume.npz) every N steps. "
                   "0 = disabled. Saved to the same directory as --checkpoint-path.")
@click.option("--resume-from", type=click.Path(path_type=Path), default=None,
              help="Resume from a directory. Full resume if it contains resume.eqx+resume.npz; "
                   "warm-start from weights only if it contains flow.eqx (run.npz history preserved if present).")
def main(
    n_steps: int,
    batch_size: int,
    num_points: int,
    seed: int,
    x_range,
    period_range,
    ls_per_range,
    ls_rbf_range,
    ls_rbf_only_range,
    rbf_only_prob: float,
    family: str,
    cp_prob: float,
    cp_y_range,
    variance: float,
    y_range,
    jitter: float,
    x_sampling: str,
    n_strat_bins_periodic: int,
    n_strat_bins_rbf_only: int,
    lr: float,
    ema_decay: float,
    grad_clip: float,
    lr_decay_steps: int,
    lr_warmup_steps: int,
    hidden_dim: int,
    n_layers: int,
    num_heads: int,
    intermediate_size: int,
    xy_embedding_dim: int,
    t_sin_features: int,
    t_mlp_hidden_dim: int,
    t_embedding_dim: int,
    dropout_rate: float,
    attention_dropout_rate: float,
    use_rope: bool,
    fourier_x_features: int,
    fourier_x_base: float,
    checkpoint_path: Path,
    plot_path: Path,
    log_every: int,
    plot_every: int,
    plot_log10: bool,
    curriculum_steps: int,
    chunk_size: int,
    checkpoint_every: int,
    resume_from,
) -> None:
    print(f"jax devices: {jax.devices()}")

    x_range_t = (float(x_range[0]), float(x_range[1]))
    y_range_t = (float(y_range[0]), float(y_range[1]))

    # If resuming, load the saved architecture from config.json and override
    # any CLI values to prevent shape-mismatch at eqx deserialise time. Print
    # a warning for every field that disagrees.
    if resume_from is not None:
        config_path = Path(resume_from) / "config.json"
        if config_path.exists():
            saved = _load_config_json(config_path)
            _locals = locals()
            for k in _ARCH_KEYS:
                if k in saved and saved[k] != _locals[k]:
                    print(f"  [config] overriding {k}: CLI={_locals[k]!r} -> saved={saved[k]!r}")
            hidden_dim = saved.get("hidden_dim", hidden_dim)
            n_layers = saved.get("n_layers", n_layers)
            num_heads = saved.get("num_heads", num_heads)
            intermediate_size = saved.get("intermediate_size", intermediate_size)
            xy_embedding_dim = saved.get("xy_embedding_dim", xy_embedding_dim)
            t_sin_features = saved.get("t_sin_features", t_sin_features)
            t_mlp_hidden_dim = saved.get("t_mlp_hidden_dim", t_mlp_hidden_dim)
            t_embedding_dim = saved.get("t_embedding_dim", t_embedding_dim)
            use_rope = saved.get("use_rope", use_rope)
            fourier_x_features = saved.get("fourier_x_features", fourier_x_features)
            fourier_x_base = saved.get("fourier_x_base", fourier_x_base)
            dropout_rate = saved.get("dropout_rate", dropout_rate)
            attention_dropout_rate = saved.get("attention_dropout_rate", attention_dropout_rate)
    n_devices = jax.local_device_count()
    if batch_size % n_devices != 0:
        raise click.UsageError(
            f"--batch-size ({batch_size}) must be divisible by number of "
            f"local devices ({n_devices})."
        )
    local_batch_size = batch_size // n_devices
    print(
        f"data-parallel across {n_devices} device(s); "
        f"effective batch={batch_size}, per-device batch={local_batch_size}"
    )

    # Guard: each branch gets local_batch_size//2 samples, so a per-branch
    # bin count above that leaves bins empty. Soft warn past sensible limits
    # (3D periodic crowds joint cells fast; 1D rbf_only has more headroom).
    max_useful = local_batch_size // 2
    for label, n in [("--n-strat-bins-periodic", n_strat_bins_periodic),
                     ("--n-strat-bins-rbf-only", n_strat_bins_rbf_only)]:
        if n > max_useful:
            raise click.UsageError(
                f"{label}={n} exceeds per-device local_batch_size//2 = "
                f"{max_useful}. With cyclic bin assignment, only "
                f"{max_useful} bins per branch can be populated; the rest "
                f"stay empty in every batch. Lower {label} or raise "
                f"--batch-size."
            )
    if n_strat_bins_periodic > 8:
        print(
            f"warning: --n-strat-bins-periodic={n_strat_bins_periodic} > 8. "
            f"With 3 stratified dims that's >512 joint cells; many will be "
            f"empty per batch. Consider <=8."
        )
    if n_strat_bins_rbf_only > 64:
        print(
            f"warning: --n-strat-bins-rbf-only={n_strat_bins_rbf_only} > 64. "
            f"Bin width is now finer than the model can resolve and per-bin "
            f"gradient signal shrinks. Consider <=64."
        )

    sampler_cfg = SamplerCfg(
        batch_size=local_batch_size,  # per-device; each device samples its own slice
        num_points=num_points,
        x_range=x_range_t,
        period_range=(float(period_range[0]), float(period_range[1])),
        ls_per_range=(float(ls_per_range[0]), float(ls_per_range[1])),
        ls_rbf_range=(float(ls_rbf_range[0]), float(ls_rbf_range[1])),
        ls_rbf_only_range=(float(ls_rbf_only_range[0]), float(ls_rbf_only_range[1])),
        rbf_only_prob=float(rbf_only_prob),
        variance=variance,
        jitter=jitter,
        x_sampling=x_sampling.lower(),
        n_strat_bins_periodic=int(n_strat_bins_periodic),
        n_strat_bins_rbf_only=int(n_strat_bins_rbf_only),
        family=family,
        cp_y_range=(float(cp_y_range[0]), float(cp_y_range[1])),
        cp_prob=float(cp_prob),
    )
    if family in ("changepoint", "mixture+changepoint"):
        lo, hi = sampler_cfg.cp_y_range
        if lo < y_range_t[0] or hi > y_range_t[1]:
            raise click.UsageError(
                f"--cp-y-range {lo},{hi} must sit inside --y-range "
                f"{y_range_t[0]},{y_range_t[1]}; otherwise samples clip "
                f"outside the normalised band the model is trained on.")
        if family == "changepoint":
            print(f"data family: changepoint (piecewise-linear, one changepoint) "
                  f"spanning [{lo}, {hi}] inside y_range {list(y_range_t)}; "
                  f"kernel hparam options ignored")
        else:
            if not 0.0 <= cp_prob <= 1.0:
                raise click.UsageError(
                    f"--cp-prob {cp_prob} must be in [0, 1].")
            print(f"data family: mixture+changepoint — per-sample "
                  f"Bernoulli(cp_prob={cp_prob:.2f}) between the GP mixture "
                  f"and piecewise-linear changepoint draws spanning "
                  f"[{lo}, {hi}] inside y_range {list(y_range_t)}")

    key = jax.random.PRNGKey(seed)
    key, net_key = jax.random.split(key)
    # Suppress the ~80-line per-layer parameter dump from FlowMatchingTransformer.__init__;
    # the same info is in config.json and a one-liner summary follows below.
    with contextlib.redirect_stdout(io.StringIO()):
        network = build_network(
            key=net_key,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            xy_embedding_dim=xy_embedding_dim,
            t_sin_features=t_sin_features,
            t_mlp_hidden_dim=t_mlp_hidden_dim,
            t_embedding_dim=t_embedding_dim,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            use_rope=use_rope,
            fourier_x_features=fourier_x_features,
            fourier_x_base=fourier_x_base,
        )
    smooth_network = network

    total_params = sum(
        x.size for x in jax.tree.leaves(eqx.filter(network, eqx.is_array))
    )
    print(
        f"model: FlowMatchingTransformer  params={total_params/1e6:.2f}M  "
        f"layers={n_layers}  hidden={hidden_dim}  rope={use_rope}  "
        f"fourier_x={fourier_x_features}"
    )
    print(
        f"training: n_steps={n_steps}  batch={batch_size}  lr={lr}  "
        f"ema={ema_decay}  grad_clip={grad_clip}  devices={n_devices}  "
        f"chunk_size={chunk_size}"
    )
    print(
        f"data: x_range={x_range_t}  period={tuple(sampler_cfg.period_range)}  "
        f"ls_per={tuple(sampler_cfg.ls_per_range)}  ls_rbf={tuple(sampler_cfg.ls_rbf_range)}  "
        f"ls_rbf_only={tuple(sampler_cfg.ls_rbf_only_range)}  "
        f"rbf_only_prob={sampler_cfg.rbf_only_prob:.2f}  variance={variance}"
    )
    print(
        f"output: ckpt={checkpoint_path}  plot={plot_path}  "
        f"log_every={log_every}  plot_every={plot_every}  "
        f"checkpoint_every={checkpoint_every}"
    )

    # Persist everything needed to reconstruct the model and reproduce the
    # data distribution. Downstream loaders (e.g. flow.py)
    # can read this JSON to build the matching TransformerConfig.
    _write_config_json(
        checkpoint_path.parent / "config.json",
        dict(
            # architecture
            hidden_dim=hidden_dim, n_layers=n_layers, num_heads=num_heads,
            intermediate_size=intermediate_size, xy_embedding_dim=xy_embedding_dim,
            t_sin_features=t_sin_features, t_mlp_hidden_dim=t_mlp_hidden_dim,
            t_embedding_dim=t_embedding_dim, use_rope=use_rope,
            fourier_x_features=fourier_x_features, fourier_x_base=fourier_x_base,
            dropout_rate=dropout_rate, attention_dropout_rate=attention_dropout_rate,
            input_dim=1, output_dim=1,
            # data / sampler
            x_range=list(x_range_t), y_range=list(y_range_t),
            period_range=list(sampler_cfg.period_range),
            ls_per_range=list(sampler_cfg.ls_per_range),
            ls_rbf_range=list(sampler_cfg.ls_rbf_range),
            ls_rbf_only_range=list(sampler_cfg.ls_rbf_only_range),
            rbf_only_prob=sampler_cfg.rbf_only_prob,
            variance=variance, jitter=jitter, num_points=num_points,
            # training
            batch_size=batch_size, lr=lr, ema_decay=ema_decay,
            grad_clip=grad_clip, seed=seed, n_steps=n_steps,
            chunk_size=chunk_size, curriculum_steps=curriculum_steps,
            lr_decay_steps=lr_decay_steps, lr_warmup_steps=lr_warmup_steps,
            x_sampling=x_sampling.lower(),
            n_strat_bins_periodic=int(n_strat_bins_periodic),
            n_strat_bins_rbf_only=int(n_strat_bins_rbf_only),
            family=family,
            cp_y_range=list(sampler_cfg.cp_y_range),
            cp_prob=sampler_cfg.cp_prob,
        ),
    )

    if lr_decay_steps > 0:
        if lr_warmup_steps > 0:
            learning_rate_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                warmup_steps=lr_warmup_steps,
                decay_steps=lr_decay_steps,  # total schedule length, includes warmup
                end_value=lr * 0.01,
            )
            print(
                f"lr schedule: warmup_cosine warmup={lr_warmup_steps} "
                f"peak={lr:.2e} total={lr_decay_steps} end={lr * 0.01:.2e}"
            )
        else:
            learning_rate_schedule = optax.cosine_decay_schedule(
                init_value=lr, decay_steps=lr_decay_steps, alpha=0.01,
            )
            print(
                f"lr schedule: cosine init={lr:.2e} "
                f"decay_steps={lr_decay_steps} alpha=0.01"
            )
    else:
        learning_rate_schedule = lr
        print(f"lr schedule: constant {lr:.2e}")
    optim = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate_schedule, eps=1e-6),
    )
    opt_state = optim.init(eqx.filter(network, eqx.is_array))

    if n_steps % chunk_size != 0:
        raise click.UsageError(
            f"--chunk-size ({chunk_size}) must divide --n-steps ({n_steps})."
        )
    n_chunks = n_steps // chunk_size
    train_chunk = make_train_chunk(sampler_cfg, y_range_t, optim, ema_decay)

    # Fixed key for GT rows + conditioning context across all plots, so
    # intermediate snapshots are directly comparable step-to-step.
    gt_key = jax.random.PRNGKey(seed + 10_000_000)

    start_step = 0
    hist_losses = np.empty(0, dtype=np.float32)
    hist_wall = np.empty(0, dtype=np.float32)
    wall_offset = 0.0
    resume_steps: list = []
    if resume_from is not None:
        template_state = (network, smooth_network, opt_state, key)
        (network, smooth_network, opt_state, key,
         start_step, hist_losses, hist_wall, prior_resume_steps, mode) = _load_resume_state(
            Path(resume_from), template_state
        )
        resume_steps = list(prior_resume_steps)
        # Record this resume as a new annotation point (skip if it's a clean
        # full-resume re-entry at the same step we just saved at, which would
        # duplicate the marker).
        if not resume_steps or resume_steps[-1] != start_step:
            resume_steps.append(start_step)
        print(f"resume ({mode}) from {resume_from}: step={start_step}  "
              f"history_len={len(hist_losses)}  resume_markers={resume_steps}")
        if start_step % chunk_size != 0:
            raise click.UsageError(
                f"resume step {start_step} not divisible by chunk_size {chunk_size}"
            )
        if start_step >= n_steps:
            raise click.UsageError(
                f"resume step {start_step} >= --n-steps {n_steps}; nothing to do "
                f"(extend with a larger --n-steps)"
            )
        wall_offset = float(hist_wall[-1]) if len(hist_wall) else 0.0

    losses = np.empty(n_steps, dtype=np.float32)
    wall_clock_s = np.empty(n_steps, dtype=np.float32)
    # New per-step instrumentation arrays. NaN-fill so plot/forensic code can
    # treat any unrecorded step (e.g. before resume point) as missing.
    grad_norms = np.full(n_steps, np.nan, dtype=np.float32)
    param_norms = np.full(n_steps, np.nan, dtype=np.float32)
    retries_per_step = np.zeros(n_steps, dtype=np.int32)
    ema_history = np.full(n_steps, np.nan, dtype=np.float32)
    if len(hist_losses):
        losses[: len(hist_losses)] = hist_losses
    if len(hist_wall):
        wall_clock_s[: len(hist_wall)] = hist_wall

    # Replicate model + optimizer state across devices for pmap. Saved/loaded
    # checkpoints stay un-replicated (single-device shape), so this works
    # transparently across runs with different n_devices.
    network = _replicate(network)
    smooth_network = _replicate(smooth_network)
    opt_state = _replicate(opt_state)

    # JSONL telemetry: one line per chunk. Append-mode so multiple resumes
    # accumulate. `pandas.read_json(path, lines=True)` parses post-hoc.
    metrics_jsonl_path = checkpoint_path.parent / "metrics.jsonl"
    metrics_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_fp = open(metrics_jsonl_path, "a")

    # EMA-loss tracker (host-side, β=0.99). Initialised lazily from the first
    # finite chunk loss after resume so it doesn't carry stale values.
    ema_loss = float("nan")
    ema_beta = 0.99
    spike_mult = 5.0  # flag step losses > spike_mult × last-1000-step median

    t0 = time.time()
    is_tty = sys.stdout.isatty()
    pbar = tqdm(
        total=n_steps, desc="training", unit="step", initial=start_step,
        disable=not is_tty, mininterval=2.0,
    )
    print(f"[run] starting at step {start_step}/{n_steps}  tty={is_tty}  "
          f"metrics_jsonl={metrics_jsonl_path}")

    # Plot schedule: either linear (--plot-every N) or log10 milestones
    # (1k, 2.5k, 5k, 10k, 25k, 50k, 100k, ...). Log10 milestones MUST land
    # on chunk boundaries (multiples of chunk_size) for the trigger to fire.
    if plot_log10:
        plot_milestones = sorted({
            int(m * 10**k)
            for k in range(3, 10)  # 10^3 to 10^9
            for m in (1.0, 2.5, 5.0)
            if 0 < int(m * 10**k) <= n_steps
        })
        bad = [m for m in plot_milestones if m % chunk_size != 0]
        if bad:
            raise click.UsageError(
                f"--plot-log10 milestones {bad} not divisible by chunk_size "
                f"{chunk_size}. Pick a chunk_size that divides 1000 (50, 100, 125, "
                f"200, 250, 500, 1000)."
            )
        plot_milestone_set = frozenset(plot_milestones)
        print(f"[plot] log10 milestones ({len(plot_milestones)}): {plot_milestones}")
    else:
        plot_milestone_set = frozenset()

    start_chunk = start_step // chunk_size
    for chunk_idx in range(start_chunk, n_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size

        key, sub = jax.random.split(key)
        # Split into n_devices*chunk_size keys, reshape so leading axis maps
        # across devices. With n_devices=1 this is functionally identical to
        # the previous single-GPU key stream.
        chunk_keys = jax.random.split(sub, n_devices * chunk_size)
        chunk_keys = chunk_keys.reshape((n_devices, chunk_size) + chunk_keys.shape[1:])

        step_idx = jnp.arange(start, end, dtype=jnp.float32)
        if curriculum_steps > 0:
            chunk_progress = jnp.minimum(step_idx / curriculum_steps, 1.0)
        else:
            chunk_progress = jnp.ones(chunk_size, dtype=jnp.float32)
        chunk_progress = jnp.broadcast_to(chunk_progress, (n_devices, chunk_size))

        (network, smooth_network, opt_state,
         chunk_losses, chunk_grad_norm, chunk_param_norm, chunk_retries) = train_chunk(
            network, smooth_network, opt_state, chunk_keys, chunk_progress
        )
        # pmean inside the inner step makes losses/norms identical across
        # devices; take device 0. Single D->H sync per metric per chunk.
        chunk_losses_np = np.asarray(chunk_losses[0])
        chunk_grad_np = np.asarray(chunk_grad_norm[0])
        chunk_param_np = np.asarray(chunk_param_norm[0])
        # `n_retries` is per-device per-step but the value can differ across
        # devices (each device draws its own GP batch); take the max so the
        # logged count reflects "any device retried".
        chunk_retries_np = np.asarray(chunk_retries.max(axis=0))
        losses[start:end] = chunk_losses_np
        grad_norms[start:end] = chunk_grad_np
        param_norms[start:end] = chunk_param_np
        retries_per_step[start:end] = chunk_retries_np

        # Host-side EMA-loss: skip non-finite values so a single NaN-step
        # doesn't poison the trace. First finite value seeds the EMA.
        for i, l in enumerate(chunk_losses_np):
            if np.isfinite(l):
                if not np.isfinite(ema_loss):
                    ema_loss = float(l)
                else:
                    ema_loss = ema_beta * ema_loss + (1.0 - ema_beta) * float(l)
            ema_history[start + i] = ema_loss

        # Spike detection vs last-1000-step median (computed only over the
        # finite history that actually exists pre-chunk).
        ref_window = losses[max(0, start - 1000):start]
        ref_median = float(np.nanmedian(ref_window)) if ref_window.size else float("nan")
        if np.isfinite(ref_median) and ref_median > 0:
            spike_mask = chunk_losses_np > spike_mult * ref_median
            if spike_mask.any():
                first_spike = int(np.argmax(spike_mask))
                tqdm.write(
                    f"  [spike] step {start + first_spike}  "
                    f"loss {chunk_losses_np[first_spike]:.4f} = "
                    f"{chunk_losses_np[first_spike] / ref_median:.1f}x last-1k median {ref_median:.4f}"
                )

        # Aggregated sampler-retry count for the chunk, printed once rather
        # than per occurrence.
        total_retries = int(chunk_retries_np.sum())
        if total_retries > 0:
            tqdm.write(
                f"  [sampler] {total_retries} GP-Cholesky retries in chunk {chunk_idx} "
                f"(steps {start}-{end-1})"
            )

        # NaN abort with on-disk forensic snapshot. Includes the failing
        # chunk's per-step trajectory and the 200 steps before it so the
        # ramp-up is preserved even if it predates the chunk boundary.
        nan_mask = np.isnan(chunk_losses_np) | np.isinf(chunk_losses_np)
        if nan_mask.any():
            first_local = int(np.argmax(nan_mask))
            first_step = start + first_local
            pre_window_start = max(0, start - 200)
            forensic_path = checkpoint_path.parent / "nan_forensic.npz"
            np.savez(
                str(forensic_path),
                nan_step=np.int64(first_step),
                chunk_idx=np.int64(chunk_idx),
                first_local=np.int64(first_local),
                pre_window_start=np.int64(pre_window_start),
                pre_window_loss=losses[pre_window_start:start].copy(),
                pre_window_grad_norm=grad_norms[pre_window_start:start].copy(),
                pre_window_param_norm=param_norms[pre_window_start:start].copy(),
                pre_window_ema=ema_history[pre_window_start:start].copy(),
                chunk_loss=chunk_losses_np,
                chunk_grad_norm=chunk_grad_np,
                chunk_param_norm=chunk_param_np,
                chunk_retries=chunk_retries_np,
                ref_median=np.float32(ref_median),
            )
            metrics_fp.close()
            pbar.close()
            raise SystemExit(
                f"NaN/Inf loss at step {first_step} (chunk {chunk_idx}, "
                f"local idx {first_local}, value {chunk_losses_np[first_local]}); "
                f"forensic snapshot at {forensic_path}"
            )

        # Assume uniform per-step wall-clock within a chunk (exact GPU timing
        # would require a sync per step, which defeats the point of scan).
        t_now = time.time() - t0 + wall_offset
        t_prev = float(wall_clock_s[start - 1]) if start > 0 else 0.0
        step_dt = (t_now - t_prev) / chunk_size
        wall_clock_s[start:end] = t_prev + step_dt * np.arange(1, chunk_size + 1)

        # One JSONL record per chunk. Cheap (~150 bytes/line), enables pandas
        # post-hoc analysis without parsing text logs.
        chunk_record = {
            "step": int(end),
            "chunk_idx": int(chunk_idx),
            "loss": float(chunk_losses_np[-1]),
            "loss_min": float(np.nanmin(chunk_losses_np)),
            "loss_max": float(np.nanmax(chunk_losses_np)),
            "ema_loss": float(ema_loss) if np.isfinite(ema_loss) else None,
            "grad_norm": float(chunk_grad_np[-1]),
            "grad_norm_max": float(np.nanmax(chunk_grad_np)),
            "param_norm": float(chunk_param_np[-1]),
            "retries": total_retries,
            "wall_s": float(t_now),
            "ref_median_1k": float(ref_median) if np.isfinite(ref_median) else None,
        }
        metrics_fp.write(json.dumps(chunk_record) + "\n")
        metrics_fp.flush()  # cheap; ensures `tail -f metrics.jsonl` works

        # Log if any log_every boundary fell within [start, end). Now includes
        # EMA loss, grad/param norms, throughput, ETA — the canonical signals.
        if (end // log_every) > (start // log_every):
            steps_done = end - start_step
            wall_s = max(1e-3, time.time() - t0)
            steps_per_s = steps_done / wall_s
            eta_s = (n_steps - end) / max(1e-3, steps_per_s)
            tqdm.write(
                f"step {end-1:>7d}  loss {losses[end-1]:.4f}  "
                f"ema {ema_loss:.4f}  |g| {chunk_grad_np[-1]:.3f}  "
                f"|p| {chunk_param_np[-1]:.1f}  "
                f"{steps_per_s:.1f}step/s  t {t_now/60:.1f}min  "
                f"ETA {eta_s/60:.0f}min"
            )

        # Fire a plot at log10 milestones (if enabled) or linear plot_every.
        if plot_log10:
            fire_plot = end in plot_milestone_set
            plot_step = end if fire_plot else None
        else:
            fire_plot = plot_every > 0 and (end // plot_every) > (start // plot_every)
            plot_step = ((start // plot_every) + 1) * plot_every if fire_plot else None
        if fire_plot:
            key, plot_key = jax.random.split(key)
            intermediate_plot = plot_path.with_name(
                f"{plot_path.stem}_step{plot_step:07d}{plot_path.suffix}"
            )
            make_plot(
                plot_path=intermediate_plot,
                network=_take_first_device(smooth_network),
                sampler_cfg=sampler_cfg,
                x_range=x_range_t,
                y_range=y_range_t,
                loss_history=losses[:end],
                wall_clock_s=wall_clock_s[:end],
                key=plot_key,
                gt_key=gt_key,
                resume_steps=resume_steps,
            )
            tqdm.write(f"  [plot] wrote {intermediate_plot}")

        # Save resume state at checkpoint_every boundaries. Must run AFTER any
        # plot-path key splits so the saved key continues the stream correctly.
        if checkpoint_every > 0 and (end // checkpoint_every) > (start // checkpoint_every):
            _save_resume_state(
                checkpoint_path.parent,
                _take_first_device(network),
                _take_first_device(smooth_network),
                _take_first_device(opt_state),
                key, end, losses, wall_clock_s, resume_steps,
            )
            tqdm.write(f"  [resume] saved state at step {end}")
        pbar.update(chunk_size)
    pbar.close()
    metrics_fp.close()

    elapsed = time.time() - t0
    print(f"training done in {elapsed:.1f}s, final loss {losses[-1]:.4f}")

    smooth_network_single = _take_first_device(smooth_network)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(str(checkpoint_path), smooth_network_single)
    print(f"checkpoint written: {checkpoint_path}")

    key, plot_key = jax.random.split(key)
    make_plot(
        plot_path=plot_path,
        network=smooth_network_single,
        sampler_cfg=sampler_cfg,
        x_range=x_range_t,
        y_range=y_range_t,
        loss_history=losses,
        wall_clock_s=wall_clock_s,
        key=plot_key,
        gt_key=gt_key,
        resume_steps=resume_steps,
    )
    print(f"plot written: {plot_path}")


if __name__ == "__main__":
    main()
