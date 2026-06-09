"""Keras losses module."""

from typing import Any

__all__ = [
    "Loss",
    "Reduction",
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
    "KLDivergence",
    "LogCosh",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanSquaredError",
    "MeanSquaredLogarithmicError",
    "Poisson",
    "SparseCategoricalCrossentropy",
    "SquaredHinge",
    "Tversky",
]


class Loss:
    """Loss base class."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Reduction:
    """No docstring available."""

    AUTO = "auto"
    NONE = "none"
    SUM = "sum"
    SUM_OVER_BATCH_SIZE = "sum_over_batch_size"

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class BinaryCrossentropy(Loss):
    """Computes the cross-entropy loss between true labels and predicted labels."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class BinaryFocalCrossentropy(Loss):
    """Computes focal cross-entropy loss between true labels and predictions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class CTC(Loss):
    """CTC (Connectionist Temporal Classification) loss."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class CategoricalCrossentropy(Loss):
    """Computes the crossentropy loss between the labels and predictions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class CategoricalFocalCrossentropy(Loss):
    """Computes the alpha balanced focal crossentropy loss."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class CategoricalGeneralizedCrossEntropy(Loss):
    """Computes the Generalized Cross Entropy loss between `y_true` & `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class CategoricalHinge(Loss):
    """Computes the categorical hinge loss between `y_true` & `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Circle(Loss):
    """Computes Circle Loss between integer labels and L2-normalized embeddings."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class CosineSimilarity(Loss):
    """Computes the cosine similarity between `y_true` & `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Dice(Loss):
    """Computes the Dice loss value between `y_true` and `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Hinge(Loss):
    """Computes the hinge loss between `y_true` & `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Huber(Loss):
    """Computes the Huber loss between `y_true` & `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class KLDivergence(Loss):
    """Computes Kullback-Leibler divergence loss between `y_true` & `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class LogCosh(Loss):
    """Computes the logarithm of the hyperbolic cosine of the prediction error."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MeanAbsoluteError(Loss):
    """Computes the mean of absolute difference between labels and predictions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MeanAbsolutePercentageError(Loss):
    """Computes the mean absolute percentage error between `y_true` & `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MeanSquaredError(Loss):
    """Computes the mean of squares of errors between labels and predictions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MeanSquaredLogarithmicError(Loss):
    """Computes the mean squared logarithmic error between `y_true` & `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Poisson(Loss):
    """Computes the Poisson loss between `y_true` & `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SparseCategoricalCrossentropy(Loss):
    """Computes the crossentropy loss between the labels and predictions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SquaredHinge(Loss):
    """Computes the squared hinge loss between `y_true` & `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Tversky(Loss):
    """Computes the Tversky loss value between `y_true` and `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass
