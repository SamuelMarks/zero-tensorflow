"""Unit tests for eager shape and array manipulation operations (`test_ops_shape_eager`)."""

from __future__ import annotations

import zero_tensorflow as tf

import numpy as np


def test_shape_eager() -> None:
    """Verify eager shape, reshape, expand_dims, squeeze, transpose, concat, stack, unstack, split, tile, broadcast_to, slice, and strided_slice operations."""
    x = tf.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    s = tf.shape(x)
    assert len(s) == 2
    assert s[0].numpy() == 2 and s[1].numpy() == 3

    reshaped = tf.reshape(x, (3, 2))
    assert isinstance(reshaped, tf.Tensor)
    assert reshaped.shape == (3, 2)
    np.testing.assert_allclose(reshaped.numpy(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    expanded = tf.expand_dims(x, axis=0)
    assert isinstance(expanded, tf.Tensor)
    assert expanded.shape == (1, 2, 3)

    squeezed = tf.squeeze(expanded, axis=0)
    assert isinstance(squeezed, tf.Tensor)
    assert squeezed.shape == (2, 3)
    np.testing.assert_allclose(squeezed.numpy(), x.numpy())

    transposed = tf.transpose(x)
    assert isinstance(transposed, tf.Tensor)
    assert transposed.shape == (3, 2)
    np.testing.assert_allclose(transposed.numpy(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    concatenated = tf.concat([x, x], axis=0)
    assert isinstance(concatenated, tf.Tensor)
    assert concatenated.shape == (4, 3)

    stacked = tf.stack([x, x], axis=0)
    assert isinstance(stacked, tf.Tensor)
    assert stacked.shape == (2, 2, 3)

    unstacked = tf.unstack(stacked, axis=0)
    assert isinstance(unstacked, (list, tuple))
    assert len(unstacked) == 2
    np.testing.assert_allclose(unstacked[0].numpy(), x.numpy())

    splits = tf.split(x, 3, axis=1)
    assert isinstance(splits, (list, tuple))
    assert len(splits) == 3
    assert splits[0].shape == (2, 1)

    tiled = tf.tile(x, [2, 1])
    assert isinstance(tiled, tf.Tensor)
    assert tiled.shape == (4, 3)

    broadcasted = tf.broadcast_to(tf.Tensor([1.0, 2.0, 3.0]), (2, 3))
    assert isinstance(broadcasted, tf.Tensor)
    assert broadcasted.shape == (2, 3)
    np.testing.assert_allclose(broadcasted.numpy(), [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    sliced = tf.slice(x, [0, 1], [2, 2])
    assert isinstance(sliced, tf.Tensor)
    assert sliced.shape == (2, 2)
    np.testing.assert_allclose(sliced.numpy(), [[2.0, 3.0], [5.0, 6.0]])

    strided = tf.strided_slice(x, [0, 0], [2, 3], [1, 2])
    assert isinstance(strided, tf.Tensor)
    assert strided.shape == (2, 2)
    np.testing.assert_allclose(strided.numpy(), [[1.0, 3.0], [4.0, 6.0]])
