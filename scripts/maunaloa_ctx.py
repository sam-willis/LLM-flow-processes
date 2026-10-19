"""Context subsets for the Mauna Loa weekly context-size sweep.

Every script that builds the sweep's contexts -- `direct_maunaloa.py`
(LLM-Direct) and `llm_score_maunaloa_ctx.py` (the LLM score grid), plus
anything that samples NDP / LLM+FP or scores the arms -- has to agree exactly,
or the arms are scored against different context points. This is the one
definition.

Two schemes:

``seed=None`` -- the default subsample: evenly-spaced *indices* into the
preset's own 20 weeks. Deterministic, one subset per n.

``seed=<int>`` -- a random subsample of the same pool, which is what lets the
sweep separate "how many points" from "which points". The evenly-spaced
scheme cannot: its subsets are NOT nested (n=5 -> n=10 drops weeks 35, 72 and
91), so each column sees a different set of weeks as well as a different number
of them.

Both schemes always keep the LAST pool point (week 128), so every column has
the same context endpoint and the same forecast region.
"""
from __future__ import annotations

import os

import numpy as np

# Which preset the context sweep describes to the LLM. Override this and
# NDP_CTX_STEM together if you need a different task, so a run's artifacts
# cannot collide with the published ones.
PRESET = os.environ.get("NDP_CTX_PRESET", "mauna_loa_co2_interp_monthly")

# Artifact stem prefix. The default deliberately keeps the name the published
# pickles were written under, so it does NOT follow PRESET's spelling; what
# matters is that a run overriding one overrides the other, or the sweep
# scores one prompt and reads another prompt's pickles.
_STEM_PREFIX = os.environ.get("NDP_CTX_STEM", "maunaloa_20rand")


def context_subset(
    x_all: np.ndarray, y_all: np.ndarray, n: int, seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_context, y_context)`` of size ``n``, sorted by x."""
    x_all = np.asarray(x_all, dtype=np.float64)
    y_all = np.asarray(y_all, dtype=np.float64)
    order = np.argsort(x_all)
    x_all, y_all = x_all[order], y_all[order]
    if n > len(x_all):
        raise ValueError(f"n={n} exceeds the pool of {len(x_all)} weeks")

    if seed is None:
        idx = np.unique(np.linspace(0, len(x_all) - 1, n).round().astype(int))
    else:
        last = len(x_all) - 1
        rng = np.random.default_rng(seed * 1000 + n)   # distinct per (n, seed)
        rest = rng.choice(last, size=n - 1, replace=False)
        idx = np.sort(np.append(rest, last))
    xc, yc = x_all[idx], y_all[idx]
    assert xc.max() == x_all.max(), "context endpoint moved; regions incomparable"
    return xc, yc


def stem(n: int, seed: int | None = None) -> str:
    """Artifact stem, e.g. ``maunaloa_20rand_ctx10`` or ``..._ctx10_d2``.

    The seedless name must stay byte-identical, or previously written pickles
    stop being found.
    """
    return f"{_STEM_PREFIX}_ctx{n}" + ("" if seed is None else f"_d{seed}")
