"""TensorFlow `optimizers` module."""

from __future__ import annotations

import zero_keras.optimizers.legacy as legacy_mod
from zero_keras.optimizers import (
    SGD,
    Adadelta,
    Adafactor,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    Ftrl,
    Lamb,
    Lion,
    LossScaleOptimizer,
    Muon,
    Nadam,
    Optimizer,
    RMSprop,
    deserialize,
    get,
    schedules,
    serialize,
)


class legacy:
    """Legacy TensorFlow 1.x/2.x optimizer namespace."""

    Adagrad = legacy_mod.Adagrad
    Adam = legacy_mod.Adam
    Ftrl = legacy_mod.Ftrl
    Optimizer = legacy_mod.Optimizer
    RMSprop = legacy_mod.RMSprop
    SGD = legacy_mod.SGD


__all__ = [
    "Adadelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "Ftrl",
    "Lamb",
    "Lion",
    "LossScaleOptimizer",
    "Muon",
    "Nadam",
    "Optimizer",
    "RMSprop",
    "SGD",
    "deserialize",
    "get",
    "legacy",
    "schedules",
    "serialize",
]
