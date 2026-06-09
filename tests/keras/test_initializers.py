import pytest
from zero_tensorflow.keras import initializers
import numpy as np


def test_initializers_base():
    init = initializers.Initializer()
    t = init(shape=(2, 2))
    assert t.shape == (2, 2)
    assert np.all(t.numpy() == 0)


def test_constant():
    init = initializers.Constant(value=1.5)
    assert init.value == 1.5
    t = init(shape=(2, 2))
    assert np.all(t.numpy() == 1.5)

    init2 = initializers.constant(value=2.0)
    assert init2.value == 2.0


def test_zeros():
    init = initializers.Zeros()
    t = init(shape=(3,))
    assert np.all(t.numpy() == 0)
    initializers.zeros()


def test_ones():
    init = initializers.Ones()
    t = init(shape=(3,))
    assert np.all(t.numpy() == 1)
    initializers.ones()


def test_identity():
    init = initializers.Identity(gain=2.0)
    assert init.gain == 2.0
    t = init(shape=(2, 2))
    assert np.array_equal(t.numpy(), np.array([[2.0, 0.0], [0.0, 2.0]]))

    init2 = initializers.IdentityInitializer(gain=3.0)
    assert init2.gain == 3.0

    initializers.identity(gain=1.0)

    with pytest.raises(ValueError):
        init(shape=(2, 2, 2))


def test_orthogonal():
    init = initializers.Orthogonal(gain=2.0, seed=42)
    assert init.gain == 2.0
    assert init.seed == 42
    t = init(shape=(2, 2))
    assert t.shape == (2, 2)

    initializers.OrthogonalInitializer(gain=3.0, seed=1)
    initializers.orthogonal(gain=1.0)

    with pytest.raises(ValueError):
        init(shape=(2,))


def test_random_normal():
    init = initializers.RandomNormal(mean=1.0, stddev=2.0, seed=42)
    assert init.mean == 1.0
    assert init.stddev == 2.0
    assert init.seed == 42
    t = init(shape=(10, 10))
    assert t.shape == (10, 10)

    initializers.random_normal()


def test_random_uniform():
    init = initializers.RandomUniform(minval=-1.0, maxval=1.0, seed=42)
    assert init.minval == -1.0
    assert init.maxval == 1.0
    assert init.seed == 42
    t = init(shape=(10, 10))
    assert t.shape == (10, 10)

    initializers.random_uniform()


def test_truncated_normal():
    init = initializers.TruncatedNormal(mean=1.0, stddev=2.0, seed=42)
    assert init.mean == 1.0
    assert init.stddev == 2.0
    assert init.seed == 42
    t = init(shape=(10, 10))
    assert t.shape == (10, 10)

    initializers.truncated_normal()


def test_variance_scaling():
    init = initializers.VarianceScaling(
        scale=2.0, mode="fan_out", distribution="uniform", seed=42
    )
    assert init.scale == 2.0
    assert init.mode == "fan_out"
    assert init.distribution == "uniform"
    assert init.seed == 42
    t = init(shape=(10, 10))
    assert t.shape == (10, 10)

    init_untr = initializers.VarianceScaling(
        scale=2.0, mode="fan_avg", distribution="untruncated_normal", seed=42
    )
    t2 = init_untr(shape=(10, 10))
    assert t2.shape == (10, 10)

    init_default = initializers.VarianceScaling(
        scale=2.0, mode="fan_in", distribution="truncated_normal", seed=42
    )
    t3 = init_default(shape=(1,))
    assert t3.shape == (1,)

    init_default2 = initializers.VarianceScaling(
        scale=2.0, mode="fan_in", distribution="truncated_normal", seed=42
    )
    t4 = init_default2(shape=())
    assert t4.shape == ()

    initializers.variance_scaling()


def test_glorot_normal():
    init = initializers.GlorotNormal(seed=42)
    assert init.scale == 1.0
    assert init.mode == "fan_avg"
    assert init.distribution == "truncated_normal"
    assert init.seed == 42

    initializers.glorot_normal()


def test_glorot_uniform():
    init = initializers.GlorotUniform(seed=42)
    assert init.scale == 1.0
    assert init.mode == "fan_avg"
    assert init.distribution == "uniform"
    assert init.seed == 42

    initializers.glorot_uniform()


def test_he_normal():
    init = initializers.HeNormal(seed=42)
    assert init.scale == 2.0
    assert init.mode == "fan_in"
    assert init.distribution == "truncated_normal"
    assert init.seed == 42

    initializers.he_normal()


def test_he_uniform():
    init = initializers.HeUniform(seed=42)
    assert init.scale == 2.0
    assert init.mode == "fan_in"
    assert init.distribution == "uniform"
    assert init.seed == 42

    initializers.he_uniform()


def test_lecun_normal():
    init = initializers.LecunNormal(seed=42)
    assert init.scale == 1.0
    assert init.mode == "fan_in"
    assert init.distribution == "truncated_normal"
    assert init.seed == 42

    initializers.lecun_normal()


def test_lecun_uniform():
    init = initializers.LecunUniform(seed=42)
    assert init.scale == 1.0
    assert init.mode == "fan_in"
    assert init.distribution == "uniform"
    assert init.seed == 42

    initializers.lecun_uniform()


def test_stft():
    init = initializers.STFT(
        side="complex", window="hamming", scaling="spectrum", periodic=True
    )
    assert init.side == "complex"
    assert init.window == "hamming"
    assert init.scaling == "spectrum"
    assert init.periodic is True

    t = init(shape=(3, 3))
    assert t.shape == (3, 3)

    initializers.STFTInitializer()
    initializers.stft()


def test_initializers_compute_fans_nd():
    from zero_tensorflow.keras.initializers import GlorotUniform

    init = GlorotUniform()
    res = init(shape=(2, 2, 3, 4))
    assert res.shape == (2, 2, 3, 4)
