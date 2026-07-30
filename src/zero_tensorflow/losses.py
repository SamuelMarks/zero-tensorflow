"""Losses module."""

import enum

from zero_keras.losses import *
from zero_keras.losses import (
    kl_divergence,
    log_cosh,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_logarithmic_error,
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
