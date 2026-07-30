from zero_keras.losses import (
    kl_divergence,
    log_cosh,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_logarithmic_error,
)

from zero_tensorflow.losses import (
    KLD,
    MAE,
    MAPE,
    MSE,
    MSLE,
    Reduction,
    kld,
    kullback_leibler_divergence,
    logcosh,
    mae,
    mape,
    mse,
    msle,
)


def test_reduction_enum():
    assert Reduction.AUTO.value == "auto"
    assert Reduction.NONE.value == "none"
    assert Reduction.SUM.value == "sum"
    assert Reduction.SUM_OVER_BATCH_SIZE.value == "sum_over_batch_size"


def test_loss_aliases():
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
