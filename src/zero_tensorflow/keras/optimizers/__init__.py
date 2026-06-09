"""Keras optimizers module."""

import numpy as np
import zero_tensorflow
from typing import Any

from . import schedules

__all__ = [
    "schedules",
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
]


class Optimizer:
    """Abstract optimizer base class."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self.learning_rate = kwargs.get("learning_rate") or (
            args[0] if len(args) > 0 else 0.001
        )
        self.iterations = 0
        self._momentums = {}
        self._velocities = {}
        self.weight_decay = kwargs.get("weight_decay")

    def _get_lr(self):
        """_get_lr docstring."""
        if hasattr(self.learning_rate, "__call__"):
            return self.learning_rate(self.iterations)
        return self.learning_rate

    def apply_gradients(self, grads_and_vars):
        """apply_gradients docstring."""
        self.iterations += 1
        for grad, var in grads_and_vars:
            if grad is None:
                continue
            self._apply_update(grad, var)

    def _apply_update(self, grad, var):
        """_apply_update docstring."""
        raise NotImplementedError


class Adam(Optimizer):
    """Optimizer that implements the Adam algorithm."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)
        self.learning_rate = kwargs.get("learning_rate") or (
            args[0] if len(args) > 0 else 0.001
        )
        self.beta_1 = kwargs.get("beta_1", args[1] if len(args) > 1 else 0.9)
        self.beta_2 = kwargs.get("beta_2", args[2] if len(args) > 2 else 0.999)
        self.epsilon = kwargs.get("epsilon", args[3] if len(args) > 3 else 1e-7)

    def _apply_update(self, grad, var):
        """_apply_update docstring."""
        lr = self._get_lr()
        var_id = id(var)

        grad_np = (
            grad.numpy() if isinstance(grad, zero_tensorflow.Tensor) else np.array(grad)
        )
        var_np = (
            var.value.numpy()
            if isinstance(var.value, zero_tensorflow.Tensor)
            else np.array(var.value)
        )

        if self.weight_decay:
            grad_np += self.weight_decay * var_np

        if var_id not in self._momentums:
            self._momentums[var_id] = np.zeros_like(var_np)
            self._velocities[var_id] = np.zeros_like(var_np)

        m = self._momentums[var_id]
        v = self._velocities[var_id]

        # m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
        m_t = self.beta_1 * m + (1.0 - self.beta_1) * grad_np

        # v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
        v_t = self.beta_2 * v + (1.0 - self.beta_2) * (grad_np**2)

        self._momentums[var_id] = m_t
        self._velocities[var_id] = v_t

        # Bias correction
        m_hat = m_t / (1.0 - self.beta_1**self.iterations)
        v_hat = v_t / (1.0 - self.beta_2**self.iterations)

        var.assign(var_np - lr * m_hat / (np.sqrt(v_hat) + self.epsilon))


class RMSprop(Optimizer):
    """Optimizer that implements the RMSprop algorithm."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)
        self.learning_rate = kwargs.get("learning_rate") or (
            args[0] if len(args) > 0 else 0.001
        )
        self.rho = kwargs.get("rho", args[1] if len(args) > 1 else 0.9)
        self.momentum = kwargs.get("momentum", args[2] if len(args) > 2 else 0.0)
        self.epsilon = kwargs.get("epsilon", args[3] if len(args) > 3 else 1e-7)

    def _apply_update(self, grad, var):
        """_apply_update docstring."""
        lr = self._get_lr()
        var_id = id(var)

        grad_np = (
            grad.numpy() if isinstance(grad, zero_tensorflow.Tensor) else np.array(grad)
        )
        var_np = (
            var.value.numpy()
            if isinstance(var.value, zero_tensorflow.Tensor)
            else np.array(var.value)
        )

        if self.weight_decay:
            grad_np += self.weight_decay * var_np

        if var_id not in self._velocities:
            self._velocities[var_id] = np.zeros_like(var_np)
            if self.momentum > 0:
                self._momentums[var_id] = np.zeros_like(var_np)

        v = self._velocities[var_id]

        # v_t = rho * v_{t-1} + (1 - rho) * g_t^2
        v_t = self.rho * v + (1.0 - self.rho) * (grad_np**2)
        self._velocities[var_id] = v_t

        step = lr * grad_np / (np.sqrt(v_t) + self.epsilon)

        if self.momentum > 0.0:
            m = self._momentums[var_id]
            m_t = self.momentum * m + step
            self._momentums[var_id] = m_t
            var.assign(var_np - m_t)
        else:
            var.assign(var_np - step)


class Adagrad(Optimizer):
    """Optimizer that implements the Adagrad algorithm."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)
        self.learning_rate = kwargs.get("learning_rate") or (
            args[0] if len(args) > 0 else 0.001
        )
        self.initial_accumulator_value = kwargs.get(
            "initial_accumulator_value", args[1] if len(args) > 1 else 0.1
        )
        self.epsilon = kwargs.get("epsilon", args[2] if len(args) > 2 else 1e-7)

    def _apply_update(self, grad, var):
        """_apply_update docstring."""
        lr = self._get_lr()
        var_id = id(var)

        grad_np = (
            grad.numpy() if isinstance(grad, zero_tensorflow.Tensor) else np.array(grad)
        )
        var_np = (
            var.value.numpy()
            if isinstance(var.value, zero_tensorflow.Tensor)
            else np.array(var.value)
        )

        if self.weight_decay:
            grad_np += self.weight_decay * var_np

        if var_id not in self._velocities:
            self._velocities[var_id] = np.full_like(
                var_np, self.initial_accumulator_value
            )

        v = self._velocities[var_id]

        # accum_t = accumulator_{t-1} + g_t^2
        v_t = v + (grad_np**2)
        self._velocities[var_id] = v_t

        var.assign(var_np - lr * grad_np / (np.sqrt(v_t) + self.epsilon))


class Adadelta(Optimizer):
    """Optimizer that implements the Adadelta algorithm."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class Adafactor(Optimizer):
    """Optimizer that implements the Adafactor algorithm."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class AdamW(Optimizer):
    """Optimizer that implements the AdamW algorithm."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class Adamax(Optimizer):
    """Optimizer that implements the Adamax algorithm."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class Ftrl(Optimizer):
    """Optimizer that implements the FTRL algorithm."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class Lamb(Optimizer):
    """Optimizer that implements the Lamb algorithm."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class Lion(Optimizer):
    """Optimizer that implements the Lion algorithm."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class LossScaleOptimizer(Optimizer):
    """An optimizer that dynamically scales the loss to prevent underflow."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class Muon(Optimizer):
    """Optimizer that implements the Muon algorithm."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class Nadam(Optimizer):
    """Optimizer that implements the Nadam algorithm."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)


class SGD(Optimizer):
    """Gradient descent (with momentum) optimizer."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__(*args, **kwargs)
        self.learning_rate = kwargs.get("learning_rate") or (
            args[0] if len(args) > 0 else 0.01
        )
        self.momentum = kwargs.get("momentum") or (args[1] if len(args) > 1 else 0.0)
        self.nesterov = kwargs.get("nesterov") or (args[2] if len(args) > 2 else False)

    def _apply_update(self, grad, var):
        """_apply_update docstring."""
        lr = self._get_lr()

        grad_np = (
            grad.numpy() if isinstance(grad, zero_tensorflow.Tensor) else np.array(grad)
        )
        var_np = (
            var.value.numpy()
            if isinstance(var.value, zero_tensorflow.Tensor)
            else np.array(var.value)
        )

        if self.weight_decay:
            grad_np += self.weight_decay * var_np

        if self.momentum > 0.0:
            var_id = id(var)
            if var_id not in self._momentums:
                self._momentums[var_id] = np.zeros_like(var_np)

            m = self._momentums[var_id]

            # v_t = momentum * v_{t-1} - lr * g_t
            m_t = self.momentum * m - lr * grad_np
            self._momentums[var_id] = m_t

            if self.nesterov:
                # w_t = w_{t-1} + momentum * v_t - lr * g_t
                var.assign(var_np + self.momentum * m_t - lr * grad_np)
            else:
                var.assign(var_np + m_t)
        else:
            var.assign(var_np - lr * grad_np)
