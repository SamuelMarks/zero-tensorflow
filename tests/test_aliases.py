import zero_tensorflow as ztf
import zero_keras as keras


def test_metrics_alias():
    assert ztf.metrics.Accuracy is keras.metrics.Accuracy


def test_losses_alias():
    assert ztf.losses.MeanSquaredError is keras.losses.MeanSquaredError


def test_optimizers_alias():
    assert ztf.optimizers.Adam is keras.optimizers.Adam


def test_initializers_alias():
    assert ztf.initializers is keras.initializers
    assert ztf.initializers.Zeros is keras.initializers.Zeros
