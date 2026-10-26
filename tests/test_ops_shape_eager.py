"""Unit tests for eager shape and array manipulation operations (`test_ops_shape_eager`)."""

from __future__ import annotations

import zero_tensorflow as tf


def test_shape_eager() -> None:
    """Verify eager shape, reshape, expand_dims, squeeze, transpose, concat, stack, unstack, split, tile, broadcast_to, slice, and strided_slice operations."""
    x = tf.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    s = tf.shape(x)
    assert len(s) == 2
    assert s[0].numpy() == 2 and s[1].numpy() == 3

    reshaped = tf.reshape(x, (3, 2))
    assert isinstance(reshaped, tf.Tensor)
    assert reshaped.shape == (3, 2)
    tf.debugging.assert_near(reshaped, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    expanded = tf.expand_dims(x, axis=0)
    assert isinstance(expanded, tf.Tensor)
    assert expanded.shape == (1, 2, 3)

    squeezed = tf.squeeze(expanded, axis=0)
    assert isinstance(squeezed, tf.Tensor)
    assert squeezed.shape == (2, 3)
    tf.debugging.assert_near(squeezed, x)

    transposed = tf.transpose(x)
    assert isinstance(transposed, tf.Tensor)
    assert transposed.shape == (3, 2)
    tf.debugging.assert_near(transposed, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    concatenated = tf.concat([x, x], axis=0)
    assert isinstance(concatenated, tf.Tensor)
    assert concatenated.shape == (4, 3)

    stacked = tf.stack([x, x], axis=0)
    assert isinstance(stacked, tf.Tensor)
    assert stacked.shape == (2, 2, 3)

    unstacked = tf.unstack(stacked, axis=0)
    assert isinstance(unstacked, (list, tuple))
    assert len(unstacked) == 2
    tf.debugging.assert_near(unstacked[0], x)

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
    tf.debugging.assert_near(broadcasted, [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    sliced = tf.slice(x, [0, 1], [2, 2])
    assert isinstance(sliced, tf.Tensor)
    assert sliced.shape == (2, 2)
    tf.debugging.assert_near(sliced, [[2.0, 3.0], [5.0, 6.0]])

    strided = tf.strided_slice(x, [0, 0], [2, 3], [1, 2])
    assert isinstance(strided, tf.Tensor)
    assert strided.shape == (2, 2)
    tf.debugging.assert_near(strided, [[1.0, 3.0], [4.0, 6.0]])

    strided_shrink = tf.strided_slice(x, [1, 0], [2, 3], [1, 1], shrink_axis_mask=1)
    assert strided_shrink.shape == (3,)
    tf.debugging.assert_near(strided_shrink, [4.0, 5.0, 6.0])

    strided_new = tf.strided_slice(x, [0, 0], [0, 2], [1, 1], new_axis_mask=1)
    assert strided_new.shape == (1, 2, 3)

    strided_ellipsis = tf.strided_slice(x, [0, 1], [0, 3], [1, 1], ellipsis_mask=1)
    assert strided_ellipsis.shape == (2, 2)
    tf.debugging.assert_near(strided_ellipsis, [[2.0, 3.0], [5.0, 6.0]])

    strided_extra_axes = tf.strided_slice(
        x, [0, 0, 0], [2, 3, 1], [1, 1, 1], begin_mask=1, end_mask=2
    )
    assert strided_extra_axes.shape == (2, 3)
