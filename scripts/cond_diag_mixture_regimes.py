"""Conditional-sampling diagnostic for the mixture model.

The mixture checkpoint (`RUN_DIR`) is trained on a 50/50 mix of
pure-RBF and periodic × RBF samples. This script asks: when given context
drawn from each regime, does the conditional sampler reproduce something
close to the analytic GP posterior — and to the ground-truth function?

Plots a 2x2 grid:
  - Row 1: periodic regime  (period × RBF kernel)
  - Row 2: RBF-only regime  (pure RBF kernel)
  - Col 1: extrapolation context (20 points on x ∈ [-2, 0])
  - Col 2: interpolation context (20 points spread on x ∈ [-2, 2])

Each panel overlays:
  - Ground-truth GP draw                            (black dashed)
  - Model mean ± 1σ band over 32 conditional draws  (orange band)
  - Analytic GP posterior mean ± 1σ band            (blue band)
  - Context points                                  (black x markers)

Sampling hyperparams default to γ=50, σ_y=0.001, ncorr=4, n_steps=100,
τ=0.1; the last three are overridable via NDP_N_CORR, NDP_N_STEPS and NDP_TAU.
No γ-sweep here. Writes the figure to $COND_DIAG_OUT (default
outputs/cond_diag_mixture_regimes.png).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import contextlib
import io
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from neural_diffusion_processes import train as spt
from neural_diffusion_processes.paths import default_flow_dir, FLOW_DIR_SHOW_DEFAULT
from neural_diffusion_processes.core import ConditionalFlow, ConditioningX, sample_n_conditional
from neural_diffusion_processes.core import XPoints, XYPoints
from neural_diffusion_processes.core import FlowProcess
from neural_diffusion_processes.core import VelocitySampler

# --- Config -----------------------------------------------------------------
# The flow checkpoint this diagnostic reads. Follows NDP_FLOW_DIR, the same
# knob the rest of the codebase uses, so one override moves everything.
RUN_DIR = default_flow_dir()
CFG_PATH = RUN_DIR / "config.json"
RESUME_PATH = RUN_DIR / "resume.eqx"
OUT_PATH = Path(os.environ.get(
    "COND_DIAG_OUT", "outputs/cond_diag_mixture_regimes.png"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 32
N_GRID = 256
N_CTX = 20
SEED = 42

# Sampling hyperparams.
GAMMA = 50.0
SIGMA_Y = 0.001
# Overridable so this diagnostic can be rendered at other sampler settings.
# NDP_N_CORR / NDP_TAU are the variables the rest of the codebase reads;
# NDP_N_STEPS is specific to this script.
NCORR = int(os.environ.get("NDP_N_CORR", "4"))
N_STEPS = int(os.environ.get("NDP_N_STEPS", "100"))
TAU = float(os.environ.get("NDP_TAU", "0.1"))

# Kernel hyperparams: choose mid-range values inside the training prior.
# Periodic regime: period × RBF.
PER_PERIOD = 1.0    # mid of [0.75, 1.25]
PER_LS_PER = 0.75   # mid of [0.5, 1.0]
PER_LS_RBF = 2.5    # mid of [1.0, 4.0]
# RBF-only regime: pure RBF with short lengthscale.
RBF_LS = 0.5        # mid of [0.1, 1.0]


# --- Load checkpoint --------------------------------------------------------
if not CFG_PATH.exists():
    raise SystemExit(
        f"no flow checkpoint at {RUN_DIR}.\n"
        "Train one (see the README) or point NDP_FLOW_DIR at an existing "
        "run directory containing config.json."
    )
print(f"loading config from {CFG_PATH}")
cfg = json.loads(CFG_PATH.read_text())
x_range = tuple(cfg["x_range"])
y_range = tuple(cfg["y_range"])
variance = float(cfg["variance"])
jitter = float(cfg["jitter"])

print(f"loading model from {RESUME_PATH}")
key0 = jax.random.PRNGKey(int(cfg["seed"]))
key0, net_key = jax.random.split(key0)
with contextlib.redirect_stdout(io.StringIO()):
    network_template = spt.build_network(
        key=net_key,
        hidden_dim=int(cfg["hidden_dim"]),
        n_layers=int(cfg["n_layers"]),
        num_heads=int(cfg["num_heads"]),
        intermediate_size=int(cfg["intermediate_size"]),
        xy_embedding_dim=int(cfg["xy_embedding_dim"]),
        t_sin_features=int(cfg["t_sin_features"]),
        t_mlp_hidden_dim=int(cfg["t_mlp_hidden_dim"]),
        t_embedding_dim=int(cfg["t_embedding_dim"]),
        dropout_rate=float(cfg["dropout_rate"]),
        attention_dropout_rate=float(cfg["attention_dropout_rate"]),
        use_rope=bool(cfg["use_rope"]),
        fourier_x_features=int(cfg["fourier_x_features"]),
        fourier_x_base=float(cfg["fourier_x_base"]),
    )
optim = optax.chain(
    optax.clip_by_global_norm(float(cfg["grad_clip"])),
    optax.adamw(float(cfg["lr"]), eps=1e-6),
)
opt_state_template = optim.init(eqx.filter(network_template, eqx.is_array))
key_template = jax.random.PRNGKey(0)
template = (network_template, network_template, opt_state_template, key_template)
if Path(RESUME_PATH).exists():
    (network, smooth_network, _opt, _key) = eqx.tree_deserialise_leaves(
        str(RESUME_PATH), template,
    )
    print("loaded EMA network")
else:
    # `resume.eqx` is the training state (net, smooth_net, opt_state, key); a
    # checkpoint directory may hold only the inference `run.eqx`, a different
    # tree. Only the EMA network is used below, so load it with
    # `pipeline.load_ndp_flow_network_local`.
    from neural_diffusion_processes import pipeline as ndp
    smooth_network = ndp.load_ndp_flow_network_local(
        Path(RESUME_PATH).parent,
        gamma_scaling_distributional=1.0, condition_ndp_part=True)
    network = smooth_network
    print(f"no resume.eqx; loaded EMA network from "
          f"{Path(RESUME_PATH).parent}/run.eqx via "
          f"pipeline.load_ndp_flow_network_local")

process = FlowProcess(x_range=x_range, y_range=y_range, gamma_scaling_ndp=GAMMA)


# --- Ground-truth draws and analytic GP posteriors --------------------------
x_grid = jnp.linspace(x_range[0], x_range[1], N_GRID).reshape(-1, 1)
x_grid_np = np.asarray(x_grid)[:, 0]


def kern_periodic(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    d = xa[:, None, 0] - xb[None, :, 0]
    k_per = np.exp(-2.0 * np.sin(np.pi * d / PER_PERIOD) ** 2 / (PER_LS_PER ** 2))
    k_rbf = np.exp(-0.5 * (d / PER_LS_RBF) ** 2)
    return float(variance) * k_per * k_rbf


def kern_rbf(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    d = xa[:, None, 0] - xb[None, :, 0]
    return float(variance) * np.exp(-0.5 * (d / RBF_LS) ** 2)


def gp_posterior_samples(kern_fn, ctx_x_np, ctx_y_np, n_samples, seed):
    xg = np.asarray(x_grid, dtype=np.float64)
    xc = ctx_x_np.reshape(-1, 1).astype(np.float64)
    yc = ctx_y_np.astype(np.float64)
    K_xx = kern_fn(xg, xg)
    K_xc = kern_fn(xg, xc)
    K_cc = kern_fn(xc, xc) + max(jitter, 1e-6) * np.eye(len(xc))
    alpha = np.linalg.solve(K_cc, yc)
    post_mean = K_xc @ alpha
    post_cov = K_xx - K_xc @ np.linalg.solve(K_cc, K_xc.T)
    post_cov = 0.5 * (post_cov + post_cov.T)
    w, V = np.linalg.eigh(post_cov)
    L = V * np.sqrt(np.clip(w, 0.0, None))[None, :]
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((len(xg), n_samples))
    return (post_mean[:, None] + L @ eps).T.astype(np.float32)


# Sample two GTs (one per regime) using the spt _sample_one generator so the
# context-y values match what training data looks like under each regime.
def sample_gt(regime: str, seed: int) -> np.ndarray:
    is_rbf_only = jnp.bool_(regime == "rbf")
    if regime == "periodic":
        period, ls_per, ls_rbf = PER_PERIOD, PER_LS_PER, PER_LS_RBF
    else:
        # `is_rbf_only=True` makes the periodic factor 1, ls_rbf becomes the
        # effective lengthscale; period / ls_per are unused by the kernel
        # but still required arguments.
        period, ls_per, ls_rbf = 1.0, 1.0, RBF_LS
    y_gt = spt._sample_one(
        jax.random.PRNGKey(seed),
        x_grid,
        jnp.float32(period),
        jnp.float32(ls_per),
        jnp.float32(ls_rbf),
        is_rbf_only,
        variance,
        jitter,
    )
    return np.asarray(y_gt)[:, 0]


print("sampling ground-truth draws")
y_gt_per = sample_gt("periodic", SEED)
y_gt_rbf = sample_gt("rbf", SEED + 1)


# --- Conditional flow sampling ---------------------------------------------
def model_conditional_samples(ctx_x_np, ctx_y_np, n_samples, seed):
    ctx_x = ctx_x_np.reshape(-1, 1).astype(np.float32)
    ctx_y = ctx_y_np.reshape(-1, 1).astype(np.float32)
    target = XPoints(x=x_grid, mask=jnp.zeros(N_GRID, dtype=bool))
    context = XYPoints(
        x=jnp.asarray(ctx_x), y=jnp.asarray(ctx_y),
        mask=jnp.zeros(len(ctx_x_np), dtype=bool),
    )
    norm_target = process.normalise_xs(target)
    norm_context = process.normalise_xys(context)
    handler = ConditioningX(
        norm_target, norm_context,
        likelihood_variance=SIGMA_Y ** 2,
        gamma_scaling_ndp=GAMMA,
    )
    cflow = ConditionalFlow(smooth_network, handler)
    sampler = VelocitySampler(n_steps=N_STEPS, num_corrections=NCORR, tau=TAU)
    target_xy, _, _ = sample_n_conditional(
        n_samples, jax.random.PRNGKey(seed), cflow, handler, sampler,
    )
    target_xy = process.denormalise_xys(target_xy)
    return np.asarray(target_xy.y[:, :, 0])


# --- Configure 4 panels ----------------------------------------------------
def context_indices(ctx_range, n_ctx):
    ctx_x = np.linspace(ctx_range[0], ctx_range[1], n_ctx)
    return np.array([int(np.argmin(np.abs(x_grid_np - cx))) for cx in ctx_x])


PANELS = [
    dict(
        regime="periodic", ctx_range=(-2.0, 0.0), label="periodic extrapolation",
        y_gt=y_gt_per, kern_fn=kern_periodic, seed_offset=10,
    ),
    dict(
        regime="periodic", ctx_range=(-2.0, 2.0), label="periodic interpolation",
        y_gt=y_gt_per, kern_fn=kern_periodic, seed_offset=20,
    ),
    dict(
        regime="rbf", ctx_range=(-2.0, 0.0), label="RBF-only extrapolation",
        y_gt=y_gt_rbf, kern_fn=kern_rbf, seed_offset=30,
    ),
    dict(
        regime="rbf", ctx_range=(-2.0, 2.0), label="RBF-only interpolation",
        y_gt=y_gt_rbf, kern_fn=kern_rbf, seed_offset=40,
    ),
]

results = []
for p in PANELS:
    print(f"sampling: {p['label']}")
    idx = context_indices(p["ctx_range"], N_CTX)
    ctx_x_np = x_grid_np[idx]
    ctx_y_np = p["y_gt"][idx]
    my = model_conditional_samples(
        ctx_x_np, ctx_y_np, N_SAMPLES, SEED + p["seed_offset"],
    )
    gp_y = gp_posterior_samples(
        p["kern_fn"], ctx_x_np, ctx_y_np, N_SAMPLES, SEED + p["seed_offset"] + 1,
    )
    ctx_pred = my[:, idx]
    ctx_mse = float(np.mean((ctx_pred - ctx_y_np[None, :]) ** 2))
    print(
        f"  ctx_MSE={ctx_mse:.3g}  "
        f"model_std_avg={my.std(axis=0).mean():.3f}  "
        f"GP_std_avg={gp_y.std(axis=0).mean():.3f}"
    )
    results.append({**p, "ctx_x": ctx_x_np, "ctx_y": ctx_y_np,
                    "my": my, "gp_y": gp_y, "ctx_mse": ctx_mse})


# --- Plot ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
for ax, r in zip(axes.flat, results):
    my = r["my"]
    gp_y = r["gp_y"]
    m_mu, m_sd = my.mean(axis=0), my.std(axis=0)
    g_mu, g_sd = gp_y.mean(axis=0), gp_y.std(axis=0)

    ax.fill_between(x_grid_np, g_mu - g_sd, g_mu + g_sd,
                    color="tab:blue", alpha=0.18, label="GP posterior ±1σ")
    ax.fill_between(x_grid_np, m_mu - m_sd, m_mu + m_sd,
                    color="tab:orange", alpha=0.30, label="model ±1σ")
    ax.plot(x_grid_np, g_mu, color="tab:blue", lw=1.0, alpha=0.85)
    ax.plot(x_grid_np, m_mu, color="tab:orange", lw=1.2, alpha=0.95)
    ax.plot(x_grid_np, r["y_gt"], color="black", lw=0.9, ls="--",
            alpha=0.7, label="GT")
    ax.scatter(r["ctx_x"], r["ctx_y"], marker="x", color="black",
               s=42, lw=1.6, zorder=10, label="ctx")

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title(r"$\mathbf{" + r["label"].replace(" ", r"\ ") + "}$",
                 fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, alpha=0.25)
axes[1, 0].set_xlabel("x", fontsize=13)
axes[1, 1].set_xlabel("x", fontsize=13)
axes[0, 0].set_ylabel("y", fontsize=13)
axes[1, 0].set_ylabel("y", fontsize=13)
axes[0, 0].legend(loc="upper right", fontsize=11, framealpha=0.9)

fig.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=120)
plt.savefig(OUT_PATH.with_suffix(".pdf"))
print(f"\nwrote {OUT_PATH}")
print(f"wrote {OUT_PATH.with_suffix('.pdf')}")
