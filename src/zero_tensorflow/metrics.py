"""Metrics module."""

from zero_keras.losses import (
    kl_divergence,
    log_cosh,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_logarithmic_error,
)
from zero_keras.metrics import *

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
