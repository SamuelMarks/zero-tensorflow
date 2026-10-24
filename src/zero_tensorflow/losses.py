"""TensorFlow `losses` module."""

from __future__ import annotations

import enum

from zero_keras.losses import (
    CTC,
    BinaryCrossentropy,
    BinaryFocalCrossentropy,
    CategoricalCrossentropy,
    CategoricalFocalCrossentropy,
    CategoricalGeneralizedCrossEntropy,
    CategoricalHinge,
    Circle,
    CosineSimilarity,
    Dice,
    Hinge,
    Huber,
    KLDivergence,
    LogCosh,
    Loss,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    MeanSquaredLogarithmicError,
    Poisson,
    SparseCategoricalCrossentropy,
    SquaredHinge,
    Tversky,
    binary_crossentropy,
    binary_focal_crossentropy,
    categorical_crossentropy,
    categorical_focal_crossentropy,
    categorical_generalized_cross_entropy,
    categorical_hinge,
    circle,
    cosine_similarity,
    ctc,
    deserialize,
    dice,
    get,
    hinge,
    huber,
    kl_divergence,
    log_cosh,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_logarithmic_error,
    poisson,
    serialize,
    sparse_categorical_crossentropy,
    squared_hinge,
    tversky,
)


class Reduction(enum.Enum):
    """Types of loss reduction."""

    AUTO = "auto"
    NONE = "none"
    SUM = "sum"
    SUM_OVER_BATCH_SIZE = "sum_over_batch_size"


kld = kl_divergence
kullback_leibler_divergence = kl_divergence
logcosh = log_cosh
mae = mean_absolute_error
mape = mean_absolute_percentage_error
mse = mean_squared_error
msle = mean_squared_logarithmic_error

KLD = kld
MAE = mae
MAPE = mape
MSE = mse
MSLE = msle

__all__ = [
    "BinaryCrossentropy",
    "BinaryFocalCrossentropy",
    "CTC",
    "CategoricalCrossentropy",
    "CategoricalFocalCrossentropy",
    "CategoricalGeneralizedCrossEntropy",
    "CategoricalHinge",
    "Circle",
    "CosineSimilarity",
    "Dice",
    "Hinge",
    "Huber",
    "KLD",
    "KLDivergence",
    "LogCosh",
    "Loss",
    "MAE",
    "MAPE",
    "MSE",
    "MSLE",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanSquaredError",
    "MeanSquaredLogarithmicError",
    "Poisson",
    "Reduction",
    "SparseCategoricalCrossentropy",
    "SquaredHinge",
    "Tversky",
    "binary_crossentropy",
    "binary_focal_crossentropy",
    "categorical_crossentropy",
    "categorical_focal_crossentropy",
    "categorical_generalized_cross_entropy",
    "categorical_hinge",
    "circle",
    "cosine_similarity",
    "ctc",
    "deserialize",
    "dice",
    "get",
    "hinge",
    "huber",
    "kl_divergence",
    "kld",
    "kullback_leibler_divergence",
    "log_cosh",
    "logcosh",
    "mae",
    "mape",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "mean_squared_logarithmic_error",
    "mse",
    "msle",
    "poisson",
    "serialize",
    "sparse_categorical_crossentropy",
    "squared_hinge",
    "tversky",
]
