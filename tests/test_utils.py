import jax
import jax.numpy as jnp
import pytest
from jax import random

from neural_diffusion_processes.core import Array, RNGKey
from neural_diffusion_processes.core import gaussian_masked_log_p, get_key_iterator, logmeanexp

SIGMA = 1
N_DATA = 10


def test_gaussian_masked_log_p__independant_unmasked():
    """
    Unmasked independant gaussian log probability test.
    """
    key_iter = get_key_iterator(0)
    # Generate mock data
    x = random.normal(next(key_iter), (N_DATA,))
    mu = random.normal(next(key_iter), (N_DATA,))

    # Expected output calculation
    expected_log_prob = jnp.sum(jax.scipy.stats.norm.logpdf(x, mu, SIGMA))
    # Calculate actual log probability
    delta = (x - mu)[..., None]
    cov = jnp.eye(N_DATA) * SIGMA**2
    actual_log_prob = gaussian_masked_log_p(delta, cov, jnp.zeros(N_DATA))

    # Assert that the computed log probability matches the expected value
    assert actual_log_prob == pytest.approx(expected_log_prob, abs=1e-5)


def test_gaussian_masked_log_p__independant_masked():
    """
    Unmasked independant gaussian log probability test.
    """
    key_iter = get_key_iterator(0)
    # Generate mock data
    x = random.normal(next(key_iter), (N_DATA,))
    mu = random.normal(next(key_iter), (N_DATA,))
    mask = random.bernoulli(next(key_iter), 0.5, (N_DATA,))

    # Expected output calculation
    log_probs = jax.scipy.stats.norm.logpdf(x, mu, SIGMA)
    expected_log_prob = jnp.sum(log_probs[mask == 0])
    # Calculate actual log probability
    delta = (x - mu)[..., None]
    cov = jnp.eye(N_DATA) * SIGMA**2
    actual_log_prob = gaussian_masked_log_p(delta, cov, mask)

    # Assert that the computed log probability matches the expected value
    assert actual_log_prob == pytest.approx(expected_log_prob, abs=1e-5)


def random_cholesky(key: RNGKey) -> Array:
    L = jnp.tril(random.normal(key, (N_DATA, N_DATA)))
    # get a square matrix with only diagonal elements of L
    diag_L = jnp.diag(jnp.diag(L))
    # if L is positive, then positive_offset is 0, otherwise it is |2L|
    positive_offset = abs(diag_L) - diag_L
    # i.e. this performs abs on the diagonal of L
    return L + positive_offset


def test_gaussian_masked_log_p__unmasked():
    """
    Unmasked gaussian log probability test.
    """
    key_iter = get_key_iterator(0)
    # Generate mock data
    x = random.normal(next(key_iter), (N_DATA,))
    mu = random.normal(next(key_iter), (N_DATA,))
    L = random_cholesky(next(key_iter))
    cov = L @ L.T
    # Expected output calculation
    expected_log_prob = jax.scipy.stats.multivariate_normal.logpdf(x, mu, cov)
    # Calculate actual log probability
    delta = (x - mu)[..., None]
    actual_log_prob = gaussian_masked_log_p(delta, cov, jnp.zeros(N_DATA))

    # Assert that the computed log probability matches the expected value
    assert actual_log_prob == pytest.approx(expected_log_prob, rel=1e-5)


def test_logmeanexp_1D():
    # Test for correctness
    key_iter = get_key_iterator(0)
    x = random.normal(next(key_iter), (N_DATA,))
    expected = jnp.log(jnp.mean(jnp.exp(x)))
    actual = logmeanexp(x)
    assert jnp.isclose(actual, expected)


def test_logmeanexp_2D():
    # Test for correctness
    key_iter = get_key_iterator(0)
    x = random.normal(next(key_iter), (N_DATA, 3))
    expected = jnp.log(jnp.mean(jnp.exp(x)))
    actual = logmeanexp(x)
    assert jnp.isclose(actual, expected)
