"""Keras metrics module."""

from typing import Any

__all__ = [
    "AUC",
    "Accuracy",
    "BinaryAccuracy",
    "BinaryCrossentropy",
    "BinaryIoU",
    "CategoricalAccuracy",
    "CategoricalCrossentropy",
    "CategoricalHinge",
    "ConcordanceCorrelation",
    "CosineSimilarity",
    "F1Score",
    "FBetaScore",
    "FalseNegatives",
    "FalsePositives",
    "Hinge",
    "IoU",
    "KLDivergence",
    "LogCoshError",
    "Mean",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanIoU",
    "MeanMetricWrapper",
    "MeanSquaredError",
    "MeanSquaredLogarithmicError",
    "Metric",
    "OneHotIoU",
    "OneHotMeanIoU",
    "PearsonCorrelation",
    "Poisson",
    "Precision",
    "PrecisionAtRecall",
    "R2Score",
    "Recall",
    "RecallAtPrecision",
    "RootMeanSquaredError",
    "SensitivityAtSpecificity",
    "SparseCategoricalAccuracy",
    "SparseCategoricalCrossentropy",
    "SparseTopKCategoricalAccuracy",
    "SpecificityAtSensitivity",
    "SquaredHinge",
    "Sum",
    "TopKCategoricalAccuracy",
    "TrueNegatives",
    "TruePositives",
]


class Metric:
    """Encapsulates metric logic and state."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class AUC(Metric):
    """Approximates the AUC (Area under the curve) of the ROC or PR curves."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Accuracy(Metric):
    """Calculates how often predictions equal labels."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class BinaryAccuracy(Metric):
    """Calculates how often predictions match binary labels."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class BinaryCrossentropy(Metric):
    """Computes the crossentropy metric between the labels and predictions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class BinaryIoU(Metric):
    """Computes the Intersection-Over-Union metric for class 0 and/or 1."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class CategoricalAccuracy(Metric):
    """Calculates how often predictions match one-hot labels."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class CategoricalCrossentropy(Metric):
    """Computes the crossentropy metric between the labels and predictions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class CategoricalHinge(Metric):
    """Computes the categorical hinge metric between `y_true` and `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class ConcordanceCorrelation(Metric):
    """Calculates the Concordance Correlation Coefficient (CCC)."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class CosineSimilarity(Metric):
    """Computes the cosine similarity between the labels and predictions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class F1Score(Metric):
    """Computes F-1 Score."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class FBetaScore(Metric):
    """Computes F-Beta score."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class FalseNegatives(Metric):
    """Calculates the number of false negatives."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class FalsePositives(Metric):
    """Calculates the number of false positives."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Hinge(Metric):
    """Computes the hinge metric between `y_true` and `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class IoU(Metric):
    """Computes the Intersection-Over-Union metric for specific target classes."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class KLDivergence(Metric):
    """Computes Kullback-Leibler divergence metric between `y_true` and `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class LogCoshError(Metric):
    """Computes the logarithm of the hyperbolic cosine of the prediction error."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Mean(Metric):
    """Compute the (weighted) mean of the given values."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MeanAbsoluteError(Metric):
    """Computes the mean absolute error between the labels and predictions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MeanAbsolutePercentageError(Metric):
    """Computes mean absolute percentage error between `y_true` and `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MeanIoU(Metric):
    """Computes the mean Intersection-Over-Union metric."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MeanMetricWrapper(Metric):
    """Wrap a stateless metric function with the `Mean` metric."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MeanSquaredError(Metric):
    """Computes the mean squared error between `y_true` and `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class MeanSquaredLogarithmicError(Metric):
    """Computes mean squared logarithmic error between `y_true` and `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class OneHotIoU(Metric):
    """Computes the Intersection-Over-Union metric for one-hot encoded labels."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class OneHotMeanIoU(Metric):
    """Computes mean Intersection-Over-Union metric for one-hot encoded labels."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class PearsonCorrelation(Metric):
    """Calculates the Pearson Correlation Coefficient (PCC)."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Poisson(Metric):
    """Computes the Poisson metric between `y_true` and `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Precision(Metric):
    """Computes the precision of the predictions with respect to the labels."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class PrecisionAtRecall(Metric):
    """Computes best precision where recall is >= specified value."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class R2Score(Metric):
    """Computes R2 score."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Recall(Metric):
    """Computes the recall of the predictions with respect to the labels."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RecallAtPrecision(Metric):
    """Computes best recall where precision is >= specified value."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class RootMeanSquaredError(Metric):
    """Computes root mean squared error metric between `y_true` and `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SensitivityAtSpecificity(Metric):
    """Computes best sensitivity where specificity is >= specified value."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SparseCategoricalAccuracy(Metric):
    """Calculates how often predictions match integer labels."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SparseCategoricalCrossentropy(Metric):
    """Computes the crossentropy metric between the labels and predictions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SparseTopKCategoricalAccuracy(Metric):
    """Computes how often integer targets are in the top `K` predictions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SpecificityAtSensitivity(Metric):
    """Computes best specificity where sensitivity is >= specified value."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class SquaredHinge(Metric):
    """Computes the hinge metric between `y_true` and `y_pred`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class Sum(Metric):
    """Compute the (weighted) sum of the given values."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class TopKCategoricalAccuracy(Metric):
    """Computes how often targets are in the top `K` predictions."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class TrueNegatives(Metric):
    """Calculates the number of true negatives."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass


class TruePositives(Metric):
    """Calculates the number of true positives."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        pass
