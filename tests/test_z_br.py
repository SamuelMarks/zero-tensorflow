"""Unit tests for `keras.layers.Dense` integration (`test_z_br`)."""

from __future__ import annotations

from zero_tensorflow import keras


def test_reimport_with_units() -> None:
    """Verify `keras.layers.Dense` instantiation, building, and `units` attribute."""
    layer = keras.layers.Dense(units=4, activation="relu")
    layer.build((None, 2))
    assert layer.built is True
    assert layer.units == 4
