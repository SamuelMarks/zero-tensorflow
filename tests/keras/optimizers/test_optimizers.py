import pytest
import numpy as np
from zero_tensorflow.keras.optimizers import (
    Optimizer,
    SGD,
    Adam,
    RMSprop,
    Adagrad,
    Adamax,
    AdamW,
    Nadam,
    Ftrl,
    Lamb,
    Lion,
    LossScaleOptimizer,
    Muon,
    Adafactor,
    Adadelta,
)
from zero_tensorflow import Variable


def test_optimizer_base():
    opt = Optimizer()
    with pytest.raises(NotImplementedError):
        opt._apply_update(None, None)


def test_sgd():
    var = Variable(np.array([1.0, 2.0]))
    grad = np.array([0.1, 0.2])
    opt = SGD(learning_rate=0.1)
    opt.apply_gradients([(grad, var)])

    np.testing.assert_allclose(var.value, np.array([0.99, 1.98]))


def test_sgd_momentum():
    var = Variable(np.array([1.0, 2.0]))
    grad = np.array([0.1, 0.2])
    opt = SGD(learning_rate=0.1, momentum=0.9)

    # step 1
    opt.apply_gradients([(grad, var)])
    np.testing.assert_allclose(var.value, np.array([0.99, 1.98]))

    # step 2
    opt.apply_gradients([(grad, var)])
    np.testing.assert_allclose(var.value, np.array([0.971, 1.942]))


def test_adam():
    var = Variable(np.array([1.0, 2.0]))
    grad = np.array([0.1, 0.2])
    opt = Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

    opt.apply_gradients([(grad, var)])

    # manual adam step 1
    m = 0.1 * grad
    v = 0.001 * (grad**2)
    m_hat = m / (1 - 0.9)
    v_hat = v / (1 - 0.999)
    expected = np.array([1.0, 2.0]) - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-7)

    np.testing.assert_allclose(var.value, expected)


def test_rmsprop():
    var = Variable(np.array([1.0, 2.0]))
    grad = np.array([0.1, 0.2])
    opt = RMSprop(learning_rate=0.1, rho=0.9, epsilon=1e-7)

    opt.apply_gradients([(grad, var)])

    v = 0.1 * (grad**2)
    expected = np.array([1.0, 2.0]) - 0.1 * grad / (np.sqrt(v) + 1e-7)

    np.testing.assert_allclose(var.value, expected)


def test_adagrad():
    var = Variable(np.array([1.0, 2.0]))
    grad = np.array([0.1, 0.2])
    opt = Adagrad(learning_rate=0.1, initial_accumulator_value=0.1, epsilon=1e-7)

    opt.apply_gradients([(grad, var)])

    v = 0.1 + (grad**2)
    expected = np.array([1.0, 2.0]) - 0.1 * grad / (np.sqrt(v) + 1e-7)

    np.testing.assert_allclose(var.value, expected)


def test_adadelta():
    Adadelta(learning_rate=0.01)


def test_adafactor():
    Adafactor(learning_rate=0.01)


def test_adamw():
    AdamW(learning_rate=0.01)


def test_adamax():
    Adamax(learning_rate=0.01)


def test_ftrl():
    Ftrl(learning_rate=0.01)


def test_lamb():
    Lamb(learning_rate=0.01)


def test_lion():
    Lion(learning_rate=0.01)


def test_loss_scale_optimizer():
    inner = Adam()
    LossScaleOptimizer(inner_optimizer=inner)


def test_muon():
    Muon(learning_rate=0.01)


def test_nadam():
    Nadam(learning_rate=0.01)


def test_optimizer_coverage():
    from zero_tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
    from zero_tensorflow.keras.optimizers.schedules import ExponentialDecay
    from zero_tensorflow import Variable, Tensor

    v = Variable(1.0)

    # Test learning rate as schedule
    lr_schedule = ExponentialDecay(0.1, 10, 0.9)
    opt = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True, weight_decay=0.01)

    opt.apply_gradients([(Tensor(0.1), v)])
    opt.apply_gradients([(Tensor(0.1), v)])  # run twice to cover momentum path

    opt_adam = Adam(weight_decay=0.01)
    opt_adam.apply_gradients([(Tensor(0.1), v)])

    opt_none = SGD()
    opt_none.apply_gradients([(None, v)])  # coverage for grad is None

    opt_rms = RMSprop(weight_decay=0.01, momentum=0.9)
    opt_rms.apply_gradients([(Tensor(0.1), v)])
    opt_rms.apply_gradients([(Tensor(0.1), v)])

    opt_adagrad = Adagrad(weight_decay=0.01)
    opt_adagrad.apply_gradients([(Tensor(0.1), v)])
