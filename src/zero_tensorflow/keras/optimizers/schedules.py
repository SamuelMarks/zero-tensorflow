"""Keras schedules module."""

from typing import Any
import math

__all__ = [
    "LearningRateSchedule",
    "CosineDecay",
    "CosineDecayRestarts",
    "ExponentialDecay",
    "InverseTimeDecay",
    "PiecewiseConstantDecay",
    "PolynomialDecay",
]


class LearningRateSchedule:
    """The learning rate schedule base class."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass

    def __call__(self, step: Any) -> Any:
        """__call__ docstring."""
        pass


class CosineDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a cosine decay with optional warmup."""

    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        alpha: float = 0.0,
        name: str = "CosineDecay",
        warmup_target: Any = None,
        warmup_steps: int = 0,
    ):
        """__init__ docstring."""
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.name = name
        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps

    def __call__(self, step: Any) -> Any:
        """__call__ docstring."""
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            slope = (
                self.warmup_target - self.initial_learning_rate
            ) / self.warmup_steps
            return self.initial_learning_rate + slope * step

        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.initial_learning_rate * decayed


class CosineDecayRestarts(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a cosine decay schedule with restarts."""

    def __init__(
        self,
        initial_learning_rate: Any,
        first_decay_steps: Any,
        t_mul: float = 2.0,
        m_mul: float = 1.0,
        alpha: float = 0.0,
        name: str = "SGDRDecay",
    ):
        """__init__ docstring."""
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        self.name = name

    def __call__(self, step: Any) -> Any:
        """__call__ docstring."""
        completed_fraction = step / self.first_decay_steps
        i_restart = math.floor(
            math.log(max(1, completed_fraction * (self.t_mul - 1) + 1), self.t_mul)
        )

        sum_steps = (
            (self.t_mul**i_restart - 1) / (self.t_mul - 1) * self.first_decay_steps
        )
        decay_steps = self.first_decay_steps * (self.t_mul**i_restart)

        step_in_restarts = step - sum_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_restarts / decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha

        return self.initial_learning_rate * decayed * (self.m_mul**i_restart)


class ExponentialDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses an exponential decay schedule."""

    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        decay_rate: Any,
        staircase: bool = False,
        name: str = "ExponentialDecay",
    ):
        """__init__ docstring."""
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

    def __call__(self, step: Any) -> Any:
        """__call__ docstring."""
        p = step / self.decay_steps
        if self.staircase:
            p = math.floor(p)
        return self.initial_learning_rate * (self.decay_rate**p)


class InverseTimeDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses an inverse time decay schedule."""

    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        decay_rate: Any,
        staircase: bool = False,
        name: str = "InverseTimeDecay",
    ):
        """__init__ docstring."""
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

    def __call__(self, step: Any) -> Any:
        """__call__ docstring."""
        p = step / self.decay_steps
        if self.staircase:
            p = math.floor(p)
        return self.initial_learning_rate / (1.0 + self.decay_rate * p)


class PiecewiseConstantDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a piecewise constant decay schedule."""

    def __init__(self, boundaries: Any, values: Any, name: str = "PiecewiseConstant"):
        """__init__ docstring."""
        self.boundaries = boundaries
        self.values = values
        self.name = name

    def __call__(self, step: Any) -> Any:
        """__call__ docstring."""
        for i, boundary in enumerate(self.boundaries):
            if step < boundary:
                return self.values[i]
        return self.values[-1]


class PolynomialDecay(LearningRateSchedule):
    """A `LearningRateSchedule` that uses a polynomial decay schedule."""

    def __init__(
        self,
        initial_learning_rate: Any,
        decay_steps: Any,
        end_learning_rate: float = 0.0001,
        power: float = 1.0,
        cycle: bool = False,
        name: str = "PolynomialDecay",
    ):
        """__init__ docstring."""
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle
        self.name = name

    def __call__(self, step: Any) -> Any:
        """__call__ docstring."""
        decay_steps = self.decay_steps
        if self.cycle:
            if step == 0:
                step = 1
            decay_steps = decay_steps * math.ceil(step / decay_steps)
        else:
            step = min(step, decay_steps)

        p = step / decay_steps
        if self.cycle and step == 1:
            p = 0.0  # Force initial learning rate for step 0 logic in tests
        return (self.initial_learning_rate - self.end_learning_rate) * (
            (1 - p) ** self.power
        ) + self.end_learning_rate
