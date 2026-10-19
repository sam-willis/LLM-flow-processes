"""Ensemble evaluation metrics: MAE, empirical CRPS, Gaussian NLPD.

Pure numpy -- no jax, torch or matplotlib -- so notebooks and analysis scripts
can import this without paying for the full sampling stack.
"""
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Metrics: MAE + empirical CRPS for ensemble samples vs ground truth.
# ---------------------------------------------------------------------------
def mean_absolute_error(samples: np.ndarray, truth: np.ndarray) -> float:
    """MAE between the per-x ensemble *mean* and the truth.

    samples: (n_samples, n_x); truth: (n_x,). Returns a single scalar.
    """
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    return float(np.mean(np.abs(samples.mean(axis=0) - truth)))


def empirical_crps(samples: np.ndarray, truth: np.ndarray) -> float:
    """Mean empirical CRPS across x.

    For each x, ``CRPS = E|X - y| - 0.5 * E|X - X'|`` estimated from the
    ensemble. Returns the mean over x. Lower is better; coincides with MAE
    for a deterministic forecast (zero spread).
    """
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    abs_err_term = np.abs(samples - truth[None, :]).mean(axis=0)  # (n_x,)
    diff = samples[:, None, :] - samples[None, :, :]              # (n, n, n_x)
    pairwise_term = np.abs(diff).mean(axis=(0, 1))                # (n_x,)
    return float((abs_err_term - 0.5 * pairwise_term).mean())


def gaussian_nlpd(
    samples: np.ndarray,
    truth: np.ndarray,
    sigma_floor: float = 1e-3,
) -> float:
    """Mean Gaussian NLPD across x.

    For each x, fit a Gaussian to the ensemble (sample mean + unbiased std,
    floored at ``sigma_floor`` to avoid divergence on degenerate ensembles)
    and evaluate ``-log N(y_true | mu, sigma^2)``. Returns the mean over x.

    Lower is better. Heavily penalises confidently-wrong forecasts
    (small sigma + large mean offset blows up the squared-error term),
    which CRPS does not directly do.
    """
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    mu = samples.mean(axis=0)
    if samples.shape[0] > 1:
        sigma = samples.std(axis=0, ddof=1)
    else:
        sigma = np.full_like(mu, sigma_floor)
    sigma = np.maximum(sigma, sigma_floor)
    nll = 0.5 * np.log(2.0 * np.pi * sigma ** 2) + 0.5 * ((truth - mu) / sigma) ** 2
    return float(nll.mean())


def evaluate_panels_against_true_fn(
    panels: dict[str, tuple[np.ndarray, np.ndarray]],
    true_fn: Callable[[float], float],
    mask_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, dict[str, float] | None]:
    """Compute MAE, CRPS and NLPD for several panels against the same ``true_fn``.

    ``panels`` maps panel-name -> ``(samples, x_eval)``. ``mask_fn`` (when
    given) is applied per-panel to its own ``x_eval`` to restrict the
    evaluation region. Panels with empty
    samples or empty mask resolve to ``None``.
    """
    out: dict[str, dict[str, float] | None] = {}
    for name, (samples, x_eval) in panels.items():
        samples = np.asarray(samples, dtype=np.float64)
        x_eval = np.asarray(x_eval, dtype=np.float64)
        if samples.shape[0] == 0:
            out[name] = None
            continue
        mask = None
        if mask_fn is not None:
            mask = np.asarray(mask_fn(x_eval), dtype=bool)
            if not mask.any():
                out[name] = None
                continue
        out[name] = evaluate_against_true_fn(samples, x_eval, true_fn, mask=mask)
    return out


def evaluate_against_true_fn(
    samples: np.ndarray,
    x_eval: np.ndarray,
    true_fn: Callable[[float], float],
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute MAE, CRPS and NLPD for ``samples`` (n_samples, n_x) against
    ``true_fn`` evaluated at ``x_eval``.

    ``mask`` (boolean, length n_x) restricts both the samples and the truth
    to a subset of x positions before scoring. The pipeline passes the NDP
    grid points at the LLM target x's, over the whole x range.
    """
    truth = np.asarray(
        [true_fn(float(xv)) for xv in x_eval], dtype=np.float64,
    )
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        samples = samples[:, mask]
        truth = truth[mask]
    return {
        "mae": mean_absolute_error(samples, truth),
        "crps": empirical_crps(samples, truth),
        "nlpd": gaussian_nlpd(samples, truth),
    }
