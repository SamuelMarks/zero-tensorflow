"""Unit tests for `test_random`."""

from __future__ import annotations

from zero_tensorflow import Tensor, random

import numpy as np


def test_random_uniform() -> None:
    """Verify `random.uniform` output shape and value bounds."""
    res = random.uniform((2, 3), minval=0.0, maxval=1.0, seed=42)
    assert isinstance(res, Tensor)
    assert res.shape == (2, 3)
    arr = np.asarray(res.numpy())
    assert np.all(arr >= 0.0) and np.all(arr <= 1.0)


def test_random_normal() -> None:
    """Verify `random.normal` output shape."""
    res = random.normal((2, 2), mean=0.0, stddev=1.0, seed=42)
    assert isinstance(res, Tensor)
    assert res.shape == (2, 2)


def test_random_truncated_normal() -> None:
    """Verify `random.truncated_normal` output shape."""
    res = random.truncated_normal((2, 2), mean=0.0, stddev=1.0, seed=42)
    assert isinstance(res, Tensor)
    assert res.shape == (2, 2)


def test_random_gamma() -> None:
    """Verify `random.gamma` and `random.gamma` with `beta` inverse scale."""
    res = random.gamma((2, 2), alpha=1.0, seed=42)
    assert isinstance(res, Tensor)
    assert res.shape == (2, 2)
    res_beta = random.gamma([2], 1.0, beta=2.0, seed=42)
    assert isinstance(res_beta, Tensor)
    assert res_beta.shape == (2,)


def test_random_poisson() -> None:
    """Verify `random.poisson` output shape and non-negative integer counts."""
    res = random.poisson((3, 4), lam=2.5, seed=42)
    assert isinstance(res, Tensor)
    assert res.shape == (3, 4)
    arr = np.asarray(res.numpy())
    assert np.all(arr >= 0.0)


def test_random_poisson_fallback() -> None:
    """Verify `random.poisson` edge cases for scalar shape, non-positive lambda, and `stateless_poisson`."""
    res_scalar_shape = random.poisson(3, lam=0.0, seed=7)
    assert isinstance(res_scalar_shape, Tensor)
    assert res_scalar_shape.shape == (3,)
    np.testing.assert_allclose(res_scalar_shape.numpy(), [0.0, 0.0, 0.0])

    res_none_shape = random.poisson(None, lam=1.5, seed=7)
    assert isinstance(res_none_shape, Tensor)
    assert res_none_shape.shape == (1,)

    res_stateless = random.stateless_poisson((2, 2), seed=[1, 2], lam=3.0)
    assert isinstance(res_stateless, Tensor)
    assert res_stateless.shape == (2, 2)


def test_random_categorical() -> None:
    """Verify `random.categorical` and `stateless_categorical`."""
    logits = Tensor([[1.0, 2.0, 3.0]])
    samples = random.categorical(logits, num_samples=4, seed=42)
    assert isinstance(samples, Tensor)
    assert samples.shape == (1, 4)

    stateless_samples = random.stateless_categorical(logits, num_samples=2, seed=[1, 2])
    assert isinstance(stateless_samples, Tensor)
    assert stateless_samples.shape == (1, 2)


def test_random_generator_and_stateless() -> None:
    """Verify `Generator`, `set_seed`, `shuffle`, `stateless_uniform`, `stateless_normal`, `stateless_gamma`, and `stateless_binomial`."""
    random.set_seed(123)
    gen = random.Generator.from_seed(123)
    u = gen.uniform((2, 2))
    n = gen.normal((2, 2))
    assert u.shape == (2, 2)
    assert n.shape == (2, 2)

    su = random.stateless_uniform((2,), seed=10)
    sn = random.stateless_normal((2,), seed=[10, 20])
    sg = random.stateless_gamma((2,), seed=[10, 20], alpha=1.0, beta=2.0)
    sb = random.stateless_binomial((2,), seed=10, counts=5, probs=0.5)
    assert su.shape == (2,)
    assert sn.shape == (2,)
    assert sg.shape == (2,)
    assert sb.shape == (2,)

    shuffled = random.shuffle(Tensor([1.0, 2.0, 3.0]), seed=42)
    assert isinstance(shuffled, Tensor)
    assert shuffled.shape == (3,)
