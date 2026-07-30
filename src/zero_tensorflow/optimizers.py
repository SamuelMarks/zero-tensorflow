"""Optimizers module."""

import zero_keras.optimizers.legacy as legacy_mod
from zero_keras.optimizers import *


class legacy:
    """Legacy optimizers module."""

    Adagrad = legacy_mod.Adagrad
    Adam = legacy_mod.Adam
    Ftrl = legacy_mod.Ftrl
    Optimizer = legacy_mod.Optimizer
    RMSprop = legacy_mod.RMSprop
    SGD = legacy_mod.SGD
