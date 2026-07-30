from zero_keras.losses import (
    kl_divergence,
    log_cosh,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_logarithmic_error,
)

from zero_tensorflow.metrics import (
    KLD,
    MAE,
    MAPE,
    MSE,
    MSLE,
    kld,
    kullback_leibler_divergence,
    logcosh,
    mae,
    mape,
    mse,
    msle,
)


def test_metrics_aliases():
    assert kld is kl_divergence
    assert kullback_leibler_divergence is kl_divergence
    assert logcosh is log_cosh
    assert mae is mean_absolute_error
    assert mape is mean_absolute_percentage_error
    assert mse is mean_squared_error
    assert msle is mean_squared_logarithmic_error

    assert KLD is kl_divergence
    assert MAE is mean_absolute_error
    assert MAPE is mean_absolute_percentage_error
    assert MSE is mean_squared_error
    assert MSLE is mean_squared_logarithmic_error
