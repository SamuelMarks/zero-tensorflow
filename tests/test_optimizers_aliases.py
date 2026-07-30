import zero_keras.optimizers.legacy as legacy_mod

from zero_tensorflow.optimizers import legacy


def test_optimizers_aliases():
    assert legacy.Adagrad is legacy_mod.Adagrad
    assert legacy.Adam is legacy_mod.Adam
    assert legacy.Ftrl is legacy_mod.Ftrl
    assert legacy.Optimizer is legacy_mod.Optimizer
    assert legacy.RMSprop is legacy_mod.RMSprop
    assert legacy.SGD is legacy_mod.SGD
