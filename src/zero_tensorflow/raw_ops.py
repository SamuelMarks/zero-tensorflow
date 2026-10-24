"""TensorFlow `raw_ops` low-level operation stubs module."""

from __future__ import annotations

from collections.abc import Sequence

from zero_tensorflow._typing import TensorOperand
from zero_tensorflow.core_tensor import Tensor

import zero_keras as keras  # isort: skip
from zero_keras import ops as _ops  # isort: skip

__all__ = ["raw_ops"]


class raw_ops:
    """TensorFlow `raw_ops` namespace."""

    @staticmethod
    def Abort(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Abort` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Abs(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Abs` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AccumulateNV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AccumulateNV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AccumulatorApplyGradient(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AccumulatorApplyGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AccumulatorNumAccumulated(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AccumulatorNumAccumulated` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AccumulatorSetGlobalStep(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AccumulatorSetGlobalStep` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AccumulatorTakeGradient(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AccumulatorTakeGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Acos(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Acos` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Acosh(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Acosh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Add(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Add` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return keras.layers.Add(*args, **kwargs)

    @staticmethod
    def AddManySparseToTensorsMap(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AddManySparseToTensorsMap` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AddN(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AddN` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AddSparseToTensorsMap(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AddSparseToTensorsMap` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AddV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AddV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AdjustContrast(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.AdjustContrast` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.AdjustContrast(*args, **kwargs)

    @staticmethod
    def AdjustContrastv2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AdjustContrastv2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AdjustHue(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.AdjustHue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.AdjustHue(*args, **kwargs)

    @staticmethod
    def AdjustSaturation(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.AdjustSaturation` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.AdjustSaturation(*args, **kwargs)

    @staticmethod
    def All(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.All` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AllCandidateSampler(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AllCandidateSampler` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AllToAll(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.AllToAll` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.AllToAll(*args, **kwargs)

    @staticmethod
    def Angle(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Angle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AnonymousHashTable(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AnonymousHashTable` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AnonymousIterator(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AnonymousIterator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AnonymousIteratorV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AnonymousIteratorV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AnonymousIteratorV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AnonymousIteratorV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AnonymousMemoryCache(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AnonymousMemoryCache` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AnonymousMultiDeviceIterator(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AnonymousMultiDeviceIterator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AnonymousMultiDeviceIteratorV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AnonymousMultiDeviceIteratorV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AnonymousMutableDenseHashTable(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AnonymousMutableDenseHashTable` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AnonymousMutableHashTable(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AnonymousMutableHashTable` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AnonymousMutableHashTableOfTensors(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AnonymousMutableHashTableOfTensors` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AnonymousRandomSeedGenerator(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AnonymousRandomSeedGenerator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AnonymousSeedGenerator(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AnonymousSeedGenerator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Any(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Any` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Any(*args, **kwargs)

    @staticmethod
    def ApplyAdaMax(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApplyAdaMax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyAdadelta(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApplyAdadelta` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyAdagrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApplyAdagrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyAdagradDA(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApplyAdagradDA` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyAdagradV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApplyAdagradV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyAdam(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApplyAdam` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyAddSign(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApplyAddSign` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyCenteredRMSProp(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ApplyCenteredRMSProp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyFtrl(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApplyFtrl` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyFtrlV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApplyFtrlV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyGradientDescent(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ApplyGradientDescent` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyMomentum(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApplyMomentum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyPowerSign(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApplyPowerSign` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyProximalAdagrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ApplyProximalAdagrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyProximalGradientDescent(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ApplyProximalGradientDescent` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApplyRMSProp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApplyRMSProp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApproxTopK(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApproxTopK` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ApproximateEqual(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ApproximateEqual` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ArgMax(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ArgMax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ArgMin(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ArgMin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AsString(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AsString` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Asin(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Asin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Asinh(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Asinh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Assert(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Assert` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AssertCardinalityDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AssertCardinalityDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AssertNextDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AssertNextDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AssertPrevDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AssertPrevDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Assign(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Assign` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Assign(*args, **kwargs)

    @staticmethod
    def AssignAdd(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.AssignAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.AssignAdd(*args, **kwargs)

    @staticmethod
    def AssignAddVariableOp(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AssignAddVariableOp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AssignSub(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.AssignSub` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.AssignSub(*args, **kwargs)

    @staticmethod
    def AssignSubVariableOp(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AssignSubVariableOp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AssignVariableOp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AssignVariableOp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AssignVariableXlaConcatND(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.AssignVariableXlaConcatND` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Atan(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Atan` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Atan2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Atan2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Atanh(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Atanh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AudioSpectrogram(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AudioSpectrogram` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AudioSummary(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AudioSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AudioSummaryV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AudioSummaryV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AutoShardDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AutoShardDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AvgPool(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AvgPool` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AvgPool3D(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.AvgPool3D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return keras.layers.AvgPool3D(*args, **kwargs)

    @staticmethod
    def AvgPool3DGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AvgPool3DGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AvgPoolGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.AvgPoolGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BandedTriangularSolve(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BandedTriangularSolve` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Barrier(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Barrier` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BarrierClose(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BarrierClose` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BarrierIncompleteSize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BarrierIncompleteSize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BarrierInsertMany(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BarrierInsertMany` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BarrierReadySize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BarrierReadySize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BarrierTakeMany(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BarrierTakeMany` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Batch(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Batch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchCholesky(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchCholesky` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchCholeskyGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchCholeskyGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchDatasetV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchFFT(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchFFT` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchFFT2D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchFFT2D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchFFT3D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchFFT3D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchFunction(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchFunction` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchIFFT(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchIFFT` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchIFFT2D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchIFFT2D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchIFFT3D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchIFFT3D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchMatMul(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchMatMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchMatMulV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchMatMulV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchMatMulV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchMatMulV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchMatrixBandPart(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BatchMatrixBandPart` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchMatrixDeterminant(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BatchMatrixDeterminant` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchMatrixDiag(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchMatrixDiag` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchMatrixDiagPart(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BatchMatrixDiagPart` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchMatrixInverse(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchMatrixInverse` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchMatrixSetDiag(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchMatrixSetDiag` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchMatrixSolve(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchMatrixSolve` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchMatrixSolveLs(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchMatrixSolveLs` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchMatrixTriangularSolve(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BatchMatrixTriangularSolve` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchNormWithGlobalNormalization(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BatchNormWithGlobalNormalization` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchNormWithGlobalNormalizationGrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BatchNormWithGlobalNormalizationGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchSelfAdjointEig(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BatchSelfAdjointEig` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchSelfAdjointEigV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BatchSelfAdjointEigV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchSvd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchSvd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchToSpace(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchToSpace` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BatchToSpaceND(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BatchToSpaceND` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BesselI0(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BesselI0` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BesselI0e(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BesselI0e` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BesselI1(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BesselI1` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BesselI1e(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BesselI1e` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BesselJ0(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BesselJ0` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BesselJ1(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BesselJ1` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BesselK0(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BesselK0` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BesselK0e(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BesselK0e` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BesselK1(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BesselK1` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BesselK1e(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BesselK1e` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BesselY0(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BesselY0` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BesselY1(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BesselY1` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Betainc(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Betainc` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BiasAdd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BiasAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BiasAddGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BiasAddGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BiasAddV1(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BiasAddV1` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Bincount(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Bincount` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Bitcast(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Bitcast` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BitwiseAnd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BitwiseAnd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BitwiseOr(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BitwiseOr` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BitwiseXor(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BitwiseXor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BlockLSTM(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BlockLSTM` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BlockLSTMGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BlockLSTMGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BlockLSTMGradV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BlockLSTMGradV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BlockLSTMV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BlockLSTMV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesAggregateStats(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesAggregateStats` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesBucketize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesBucketize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesCalculateBestFeatureSplit(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesCalculateBestFeatureSplit` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesCalculateBestFeatureSplitV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesCalculateBestFeatureSplitV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesCalculateBestGainsPerFeature(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesCalculateBestGainsPerFeature` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesCenterBias(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesCenterBias` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesCreateEnsemble(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesCreateEnsemble` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesCreateQuantileStreamResource(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesCreateQuantileStreamResource` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesDeserializeEnsemble(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesDeserializeEnsemble` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesEnsembleResourceHandleOp(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesEnsembleResourceHandleOp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesExampleDebugOutputs(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesExampleDebugOutputs` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesFlushQuantileSummaries(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesFlushQuantileSummaries` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesGetEnsembleStates(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesGetEnsembleStates` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesMakeQuantileSummaries(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesMakeQuantileSummaries` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesMakeStatsSummary(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesMakeStatsSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesPredict(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesPredict` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesQuantileStreamResourceAddSummaries(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesQuantileStreamResourceAddSummaries` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesQuantileStreamResourceDeserialize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesQuantileStreamResourceDeserialize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesQuantileStreamResourceFlush(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesQuantileStreamResourceFlush` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesQuantileStreamResourceGetBucketBoundaries(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesQuantileStreamResourceGetBucketBoundaries` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesQuantileStreamResourceHandleOp(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesQuantileStreamResourceHandleOp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesSerializeEnsemble(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesSerializeEnsemble` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesSparseAggregateStats(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesSparseAggregateStats` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesSparseCalculateBestFeatureSplit(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesSparseCalculateBestFeatureSplit` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesTrainingPredict(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesTrainingPredict` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesUpdateEnsemble(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesUpdateEnsemble` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BoostedTreesUpdateEnsembleV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BoostedTreesUpdateEnsembleV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BroadcastArgs(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.BroadcastArgs` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BroadcastGradientArgs(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BroadcastGradientArgs` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BroadcastTo(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.BroadcastTo` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.BroadcastTo(*args, **kwargs)

    @staticmethod
    def Bucketize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Bucketize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def BytesProducedStatsDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.BytesProducedStatsDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CSRSparseMatrixComponents(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CSRSparseMatrixComponents` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CSRSparseMatrixToDense(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CSRSparseMatrixToDense` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CSRSparseMatrixToSparseTensor(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CSRSparseMatrixToSparseTensor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CSVDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CSVDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CSVDatasetV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CSVDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CTCBeamSearchDecoder(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CTCBeamSearchDecoder` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CTCGreedyDecoder(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CTCGreedyDecoder` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CTCLoss(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CTCLoss` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CTCLossV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CTCLossV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CacheDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CacheDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CacheDatasetV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CacheDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Case(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Case` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Cast(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Cast` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Ceil(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Ceil` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CheckNumerics(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CheckNumerics` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CheckNumericsV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CheckNumericsV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Cholesky(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Cholesky` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CholeskyGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CholeskyGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ChooseFastestBranchDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ChooseFastestBranchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ChooseFastestDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ChooseFastestDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ClipByValue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ClipByValue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CloseSummaryWriter(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CloseSummaryWriter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveAllToAllV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CollectiveAllToAllV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveAllToAllV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CollectiveAllToAllV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveAssignGroupV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CollectiveAssignGroupV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveBcastRecv(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CollectiveBcastRecv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveBcastRecvV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CollectiveBcastRecvV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveBcastSend(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CollectiveBcastSend` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveBcastSendV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CollectiveBcastSendV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveGather(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CollectiveGather` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveGatherV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CollectiveGatherV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveInitializeCommunicator(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CollectiveInitializeCommunicator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectivePermute(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CollectivePermute` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveReduce(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CollectiveReduce` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveReduceScatterV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CollectiveReduceScatterV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveReduceV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CollectiveReduceV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CollectiveReduceV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CollectiveReduceV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CombinedNonMaxSuppression(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CombinedNonMaxSuppression` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Complex(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Complex` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ComplexAbs(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ComplexAbs` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CompositeTensorVariantFromComponents(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CompositeTensorVariantFromComponents` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CompositeTensorVariantToComponents(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CompositeTensorVariantToComponents` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CompressElement(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CompressElement` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ComputeAccidentalHits(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ComputeAccidentalHits` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ComputeBatchSize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ComputeBatchSize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Concat(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Concat` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ConcatOffset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ConcatOffset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ConcatV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ConcatV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ConcatenateDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ConcatenateDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ConditionalAccumulator(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ConditionalAccumulator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ConfigureDistributedTPU(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ConfigureDistributedTPU` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ConfigureTPUEmbedding(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ConfigureTPUEmbedding` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Conj(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Conj` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ConjugateTranspose(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ConjugateTranspose` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Const(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Const` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ConsumeMutexLock(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ConsumeMutexLock` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ControlTrigger(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ControlTrigger` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Conv(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Conv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Conv2D(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Conv2D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return keras.layers.Conv2D(*args, **kwargs)

    @staticmethod
    def Conv2DBackpropFilter(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.Conv2DBackpropFilter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Conv2DBackpropFilterV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.Conv2DBackpropFilterV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Conv2DBackpropInput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.Conv2DBackpropInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Conv2DBackpropInputV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.Conv2DBackpropInputV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Conv3D(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Conv3D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return keras.layers.Conv3D(*args, **kwargs)

    @staticmethod
    def Conv3DBackpropFilter(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.Conv3DBackpropFilter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Conv3DBackpropFilterV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.Conv3DBackpropFilterV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Conv3DBackpropInput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.Conv3DBackpropInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Conv3DBackpropInputV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.Conv3DBackpropInputV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ConvertToCooTensor(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ConvertToCooTensor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Copy(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Copy` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CopyHost(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CopyHost` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Cos(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Cos` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Cosh(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Cosh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CountUpTo(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CountUpTo` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CreateSummaryDbWriter(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CreateSummaryDbWriter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CreateSummaryFileWriter(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CreateSummaryFileWriter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CropAndResize(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.CropAndResize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.CropAndResize(*args, **kwargs)

    @staticmethod
    def CropAndResizeGradBoxes(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CropAndResizeGradBoxes` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CropAndResizeGradImage(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CropAndResizeGradImage` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Cross(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Cross` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CrossReplicaSum(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CrossReplicaSum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CudnnRNN(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CudnnRNN` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CudnnRNNBackprop(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CudnnRNNBackprop` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CudnnRNNBackpropV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CudnnRNNBackpropV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CudnnRNNBackpropV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CudnnRNNBackpropV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CudnnRNNCanonicalToParams(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CudnnRNNCanonicalToParams` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CudnnRNNCanonicalToParamsV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CudnnRNNCanonicalToParamsV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CudnnRNNParamsSize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CudnnRNNParamsSize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CudnnRNNParamsToCanonical(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CudnnRNNParamsToCanonical` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CudnnRNNParamsToCanonicalV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CudnnRNNParamsToCanonicalV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CudnnRNNV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CudnnRNNV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CudnnRNNV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.CudnnRNNV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Cumprod(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Cumprod` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Cumsum(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Cumsum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CumulativeLogsumexp(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.CumulativeLogsumexp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DataFormatDimMap(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DataFormatDimMap` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DataFormatVecPermute(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DataFormatVecPermute` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DataServiceDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DataServiceDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DataServiceDatasetV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DataServiceDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DataServiceDatasetV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DataServiceDatasetV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DataServiceDatasetV4(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DataServiceDatasetV4` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DatasetCardinality(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DatasetCardinality` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DatasetFingerprint(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DatasetFingerprint` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DatasetFromGraph(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DatasetFromGraph` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DatasetToGraph(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DatasetToGraph` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DatasetToGraphV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DatasetToGraphV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DatasetToSingleElement(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DatasetToSingleElement` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DatasetToTFRecord(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DatasetToTFRecord` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Dawsn(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Dawsn` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DebugGradientIdentity(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DebugGradientIdentity` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DebugGradientRefIdentity(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DebugGradientRefIdentity` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DebugIdentity(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DebugIdentity` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DebugIdentityV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DebugIdentityV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DebugIdentityV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DebugIdentityV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DebugNanCount(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DebugNanCount` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DebugNumericSummary(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DebugNumericSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DebugNumericSummaryV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DebugNumericSummaryV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodeAndCropJpeg(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodeAndCropJpeg` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodeBase64(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodeBase64` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodeBmp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodeBmp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodeCSV(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodeCSV` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodeCompressed(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodeCompressed` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodeGif(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodeGif` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodeImage(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodeImage` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodeJSONExample(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodeJSONExample` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodeJpeg(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodeJpeg` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodePaddedRaw(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodePaddedRaw` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodePng(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodePng` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodeProtoV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodeProtoV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodeRaw(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodeRaw` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DecodeWav(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DecodeWav` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DeepCopy(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DeepCopy` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DeleteIterator(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DeleteIterator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DeleteMemoryCache(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DeleteMemoryCache` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DeleteMultiDeviceIterator(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DeleteMultiDeviceIterator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DeleteRandomSeedGenerator(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DeleteRandomSeedGenerator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DeleteSeedGenerator(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DeleteSeedGenerator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DeleteSessionTensor(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DeleteSessionTensor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DenseBincount(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DenseBincount` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DenseCountSparseOutput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DenseCountSparseOutput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DenseToCSRSparseMatrix(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DenseToCSRSparseMatrix` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DenseToDenseSetOperation(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DenseToDenseSetOperation` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DenseToSparseBatchDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DenseToSparseBatchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DenseToSparseSetOperation(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DenseToSparseSetOperation` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DepthToSpace(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DepthToSpace` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DepthwiseConv2dNative(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DepthwiseConv2dNative` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DepthwiseConv2dNativeBackpropFilter(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DepthwiseConv2dNativeBackpropFilter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DepthwiseConv2dNativeBackpropInput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DepthwiseConv2dNativeBackpropInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Dequantize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Dequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DeserializeIterator(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DeserializeIterator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DeserializeManySparse(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DeserializeManySparse` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DeserializeSparse(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DeserializeSparse` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DestroyResourceOp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DestroyResourceOp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DestroyTemporaryVariable(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DestroyTemporaryVariable` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DeviceIndex(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DeviceIndex` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Diag(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Diag` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DiagPart(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DiagPart` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Digamma(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Digamma` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Dilation2D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Dilation2D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Dilation2DBackpropFilter(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.Dilation2DBackpropFilter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Dilation2DBackpropInput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.Dilation2DBackpropInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DirectedInterleaveDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DirectedInterleaveDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DisableCopyOnRead(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DisableCopyOnRead` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DistributedSave(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DistributedSave` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Div(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Div` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DivNoNan(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DivNoNan` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DrawBoundingBoxes(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.DrawBoundingBoxes` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.DrawBoundingBoxes(*args, **kwargs)

    @staticmethod
    def DrawBoundingBoxesV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DrawBoundingBoxesV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DummyIterationCounter(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DummyIterationCounter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DummyMemoryCache(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DummyMemoryCache` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DummySeedGenerator(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DummySeedGenerator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DynamicEnqueueTPUEmbeddingArbitraryTensorBatch(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DynamicEnqueueTPUEmbeddingArbitraryTensorBatch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DynamicEnqueueTPUEmbeddingRaggedTensorBatch(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.DynamicEnqueueTPUEmbeddingRaggedTensorBatch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DynamicPartition(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DynamicPartition` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DynamicStitch(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.DynamicStitch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EagerPyFunc(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.EagerPyFunc` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EditDistance(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.EditDistance` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Eig(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Eig` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Einsum(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Einsum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Einsum(*args, **kwargs)

    @staticmethod
    def Elu(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Elu` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EluGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.EluGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Empty(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Empty` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EmptyTensorList(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.EmptyTensorList` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EmptyTensorMap(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.EmptyTensorMap` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EncodeBase64(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.EncodeBase64` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EncodeJpeg(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.EncodeJpeg` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EncodeJpegVariableQuality(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.EncodeJpegVariableQuality` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EncodePng(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.EncodePng` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EncodeProto(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.EncodeProto` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EncodeWav(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.EncodeWav` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EnqueueTPUEmbeddingArbitraryTensorBatch(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.EnqueueTPUEmbeddingArbitraryTensorBatch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EnqueueTPUEmbeddingIntegerBatch(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.EnqueueTPUEmbeddingIntegerBatch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EnqueueTPUEmbeddingRaggedTensorBatch(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.EnqueueTPUEmbeddingRaggedTensorBatch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EnqueueTPUEmbeddingSparseBatch(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.EnqueueTPUEmbeddingSparseBatch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EnqueueTPUEmbeddingSparseTensorBatch(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.EnqueueTPUEmbeddingSparseTensorBatch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EnsureShape(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.EnsureShape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Enter(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Enter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Equal(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Equal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Erf(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Erf` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Erfc(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Erfc` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Erfinv(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Erfinv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def EuclideanNorm(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.EuclideanNorm` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Exit(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Exit` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Exp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Exp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExpandDims(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ExpandDims` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalAssertNextDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalAssertNextDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalAutoShardDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalAutoShardDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalBytesProducedStatsDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalBytesProducedStatsDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalCSVDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalCSVDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalChooseFastestDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalChooseFastestDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalDatasetCardinality(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalDatasetCardinality` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalDatasetToTFRecord(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalDatasetToTFRecord` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalDenseToSparseBatchDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalDenseToSparseBatchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalDirectedInterleaveDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalDirectedInterleaveDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalGroupByReducerDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalGroupByReducerDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalGroupByWindowDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalGroupByWindowDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalIgnoreErrorsDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalIgnoreErrorsDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalIteratorGetDevice(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalIteratorGetDevice` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalLMDBDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalLMDBDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalLatencyStatsDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalLatencyStatsDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalMapAndBatchDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalMapAndBatchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalMapDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalMapDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalMatchingFilesDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalMatchingFilesDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalMaxIntraOpParallelismDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalMaxIntraOpParallelismDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalNonSerializableDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalNonSerializableDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalParallelInterleaveDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalParallelInterleaveDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalParseExampleDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalParseExampleDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalPrivateThreadPoolDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalPrivateThreadPoolDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalRandomDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalRandomDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalRebatchDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalRebatchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalScanDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalScanDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalSetStatsAggregatorDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalSetStatsAggregatorDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalSleepDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalSleepDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalSlidingWindowDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalSlidingWindowDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalSqlDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalSqlDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalStatsAggregatorHandle(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalStatsAggregatorHandle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalStatsAggregatorSummary(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalStatsAggregatorSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalTakeWhileDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalTakeWhileDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalThreadPoolDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalThreadPoolDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalThreadPoolHandle(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalThreadPoolHandle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalUnbatchDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalUnbatchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExperimentalUniqueDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExperimentalUniqueDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Expint(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Expint` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Expm1(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Expm1` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExtractGlimpse(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ExtractGlimpse` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExtractGlimpseV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ExtractGlimpseV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExtractImagePatches(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExtractImagePatches` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExtractJpegShape(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ExtractJpegShape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExtractVolumePatches(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ExtractVolumePatches` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FFT(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FFT` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FFT2D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FFT2D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FFT3D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FFT3D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FFTND(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FFTND` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FIFOQueue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FIFOQueue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FIFOQueueV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FIFOQueueV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Fact(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Fact` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FakeParam(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FakeParam` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FakeQuantWithMinMaxArgs(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FakeQuantWithMinMaxArgs` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FakeQuantWithMinMaxArgsGradient(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FakeQuantWithMinMaxArgsGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FakeQuantWithMinMaxVars(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FakeQuantWithMinMaxVars` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FakeQuantWithMinMaxVarsGradient(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FakeQuantWithMinMaxVarsGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FakeQuantWithMinMaxVarsPerChannel(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FakeQuantWithMinMaxVarsPerChannel` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FakeQuantWithMinMaxVarsPerChannelGradient(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FakeQuantWithMinMaxVarsPerChannelGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FakeQueue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FakeQueue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FileSystemSetConfiguration(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FileSystemSetConfiguration` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Fill(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Fill` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FilterByLastComponentDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FilterByLastComponentDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FilterDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FilterDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FinalizeDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FinalizeDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Fingerprint(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Fingerprint` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FixedLengthRecordDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FixedLengthRecordDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FixedLengthRecordDatasetV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FixedLengthRecordDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FixedLengthRecordReader(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FixedLengthRecordReader` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FixedLengthRecordReaderV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FixedLengthRecordReaderV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FixedUnigramCandidateSampler(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FixedUnigramCandidateSampler` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FlatMapDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FlatMapDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Floor(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Floor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FloorDiv(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FloorDiv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FloorMod(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FloorMod` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FlushSummaryWriter(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FlushSummaryWriter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def For(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.For` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FractionalAvgPool(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FractionalAvgPool` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FractionalAvgPoolGrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FractionalAvgPoolGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FractionalMaxPool(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FractionalMaxPool` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FractionalMaxPoolGrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FractionalMaxPoolGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FresnelCos(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FresnelCos` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FresnelSin(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FresnelSin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FusedBatchNorm(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FusedBatchNorm` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FusedBatchNormGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FusedBatchNormGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FusedBatchNormGradV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FusedBatchNormGradV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FusedBatchNormGradV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FusedBatchNormGradV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FusedBatchNormV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FusedBatchNormV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FusedBatchNormV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FusedBatchNormV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FusedPadConv2D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.FusedPadConv2D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FusedResizeAndPadConv2D(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.FusedResizeAndPadConv2D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GRUBlockCell(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.GRUBlockCell` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GRUBlockCellGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.GRUBlockCellGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Gather(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Gather` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Gather(*args, **kwargs)

    @staticmethod
    def GatherNd(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.GatherNd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.GatherNd(*args, **kwargs)

    @staticmethod
    def GatherV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.GatherV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GenerateBoundingBoxProposals(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.GenerateBoundingBoxProposals` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GenerateVocabRemapping(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.GenerateVocabRemapping` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GeneratorDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.GeneratorDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GetElementAtIndex(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.GetElementAtIndex` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GetMinibatchSplitsWithPhysicalReplica(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.GetMinibatchSplitsWithPhysicalReplica` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GetMinibatchesInCsrWithPhysicalReplica(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.GetMinibatchesInCsrWithPhysicalReplica` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GetOptions(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.GetOptions` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GetSessionHandle(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.GetSessionHandle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GetSessionHandleV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.GetSessionHandleV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GetSessionTensor(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.GetSessionTensor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GlobalIterId(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.GlobalIterId` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Greater(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Greater` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GreaterEqual(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.GreaterEqual` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GroupByReducerDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.GroupByReducerDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GroupByWindowDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.GroupByWindowDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GuaranteeConst(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.GuaranteeConst` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def HSVToRGB(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.HSVToRGB` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def HashTable(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.HashTable` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def HashTableV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.HashTableV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def HistogramFixedWidth(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.HistogramFixedWidth` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def HistogramSummary(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.HistogramSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IFFT(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IFFT` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IFFT2D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IFFT2D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IFFT3D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IFFT3D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IFFTND(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IFFTND` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IRFFT(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IRFFT` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IRFFT2D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IRFFT2D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IRFFT3D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IRFFT3D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IRFFTND(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IRFFTND` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Identity(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Identity` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return keras.layers.Identity(*args, **kwargs)

    @staticmethod
    def IdentityN(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IdentityN` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IdentityReader(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IdentityReader` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IdentityReaderV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IdentityReaderV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def If(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.If` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Igamma(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Igamma` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IgammaGradA(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IgammaGradA` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Igammac(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Igammac` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IgnoreErrorsDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.IgnoreErrorsDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Imag(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Imag` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ImageProjectiveTransformV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ImageProjectiveTransformV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ImageProjectiveTransformV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ImageProjectiveTransformV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ImageSummary(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ImageSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ImmutableConst(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ImmutableConst` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ImportEvent(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ImportEvent` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InTopK(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InTopK` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InTopKV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InTopKV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InfeedDequeue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InfeedDequeue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InfeedDequeueTuple(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InfeedDequeueTuple` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InfeedEnqueue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InfeedEnqueue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InfeedEnqueuePrelinearizedBuffer(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.InfeedEnqueuePrelinearizedBuffer` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InfeedEnqueueTuple(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InfeedEnqueueTuple` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InitializeTable(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InitializeTable` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InitializeTableFromDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.InitializeTableFromDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InitializeTableFromTextFile(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.InitializeTableFromTextFile` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InitializeTableFromTextFileV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.InitializeTableFromTextFileV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InitializeTableV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InitializeTableV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InplaceAdd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InplaceAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InplaceSub(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InplaceSub` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InplaceUpdate(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InplaceUpdate` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InterleaveDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InterleaveDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Inv(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Inv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InvGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InvGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Invert(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Invert` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Invert(*args, **kwargs)

    @staticmethod
    def InvertPermutation(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.InvertPermutation` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IsBoostedTreesEnsembleInitialized(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.IsBoostedTreesEnsembleInitialized` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IsBoostedTreesQuantileStreamResourceInitialized(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.IsBoostedTreesQuantileStreamResourceInitialized` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IsFinite(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IsFinite` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IsInf(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IsInf` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IsNan(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IsNan` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IsTPUEmbeddingInitialized(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.IsTPUEmbeddingInitialized` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IsVariableInitialized(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.IsVariableInitialized` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IsotonicRegression(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IsotonicRegression` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Iterator(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Iterator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IteratorFromStringHandle(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.IteratorFromStringHandle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IteratorFromStringHandleV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.IteratorFromStringHandleV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IteratorGetDevice(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IteratorGetDevice` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IteratorGetNext(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IteratorGetNext` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IteratorGetNextAsOptional(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.IteratorGetNextAsOptional` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IteratorGetNextSync(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.IteratorGetNextSync` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IteratorToStringHandle(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.IteratorToStringHandle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def IteratorV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.IteratorV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def KMC2ChainInitialization(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.KMC2ChainInitialization` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def KmeansPlusPlusInitialization(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.KmeansPlusPlusInitialization` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def L2Loss(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.L2Loss` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LMDBDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LMDBDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LMDBReader(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LMDBReader` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LRN(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LRN` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LRNGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LRNGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LSTMBlockCell(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LSTMBlockCell` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LSTMBlockCellGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LSTMBlockCellGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LatencyStatsDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LatencyStatsDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LeakyRelu(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LeakyRelu` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LeakyReluGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LeakyReluGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LearnedUnigramCandidateSampler(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LearnedUnigramCandidateSampler` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LeftShift(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LeftShift` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LegacyParallelInterleaveDatasetV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LegacyParallelInterleaveDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Less(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Less` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LessEqual(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LessEqual` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Lgamma(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Lgamma` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LinSpace(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LinSpace` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ListDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ListDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ListDiff(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ListDiff` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ListSnapshotChunksDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ListSnapshotChunksDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadAndRemapMatrix(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LoadAndRemapMatrix` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LoadDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingADAMParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingADAMParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingAdadeltaParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingAdadeltaParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingAdagradMomentumParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingAdagradMomentumParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingAdagradParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingAdagradParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingCenteredRMSPropParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingCenteredRMSPropParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingFTRLParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingFTRLParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingFrequencyEstimatorParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingFrequencyEstimatorParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingMDLAdagradLightParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingMDLAdagradLightParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingMomentumParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingMomentumParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingProximalAdagradParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingProximalAdagradParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingProximalYogiParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingProximalYogiParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingRMSPropParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingRMSPropParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoadTPUEmbeddingStochasticGradientDescentParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LoadTPUEmbeddingStochasticGradientDescentParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Log(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Log` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Log1p(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Log1p` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LogMatrixDeterminant(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LogMatrixDeterminant` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LogSoftmax(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LogSoftmax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LogUniformCandidateSampler(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LogUniformCandidateSampler` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LogicalAnd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LogicalAnd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LogicalNot(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LogicalNot` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LogicalOr(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LogicalOr` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LookupTableExport(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LookupTableExport` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LookupTableExportV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LookupTableExportV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LookupTableFind(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LookupTableFind` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LookupTableFindV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LookupTableFindV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LookupTableImport(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LookupTableImport` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LookupTableImportV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LookupTableImportV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LookupTableInsert(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LookupTableInsert` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LookupTableInsertV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LookupTableInsertV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LookupTableRemoveV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.LookupTableRemoveV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LookupTableSize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LookupTableSize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LookupTableSizeV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LookupTableSizeV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LoopCond(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LoopCond` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def LowerBound(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.LowerBound` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Lu(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Lu` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MakeIterator(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MakeIterator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MapAndBatchDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MapAndBatchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MapClear(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MapClear` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MapDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MapDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MapDefun(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MapDefun` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MapIncompleteSize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MapIncompleteSize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MapPeek(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MapPeek` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MapSize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MapSize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MapStage(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MapStage` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MapUnstage(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MapUnstage` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MapUnstageNoKey(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MapUnstageNoKey` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatMul(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatchingFiles(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatchingFiles` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatchingFilesDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MatchingFilesDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixBandPart(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixBandPart` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixDeterminant(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixDeterminant` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixDiag(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixDiag` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixDiagPart(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixDiagPart` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixDiagPartV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixDiagPartV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixDiagPartV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixDiagPartV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixDiagV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixDiagV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixDiagV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixDiagV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixExponential(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixExponential` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixInverse(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixInverse` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixLogarithm(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixLogarithm` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixSetDiag(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixSetDiag` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixSetDiagV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixSetDiagV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixSetDiagV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixSetDiagV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixSolve(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixSolve` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixSolveLs(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixSolveLs` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixSquareRoot(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MatrixSquareRoot` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MatrixTriangularSolve(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MatrixTriangularSolve` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Max(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Max` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MaxIntraOpParallelismDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MaxIntraOpParallelismDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MaxPool(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MaxPool` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MaxPool3D(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.MaxPool3D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return keras.layers.MaxPool3D(*args, **kwargs)

    @staticmethod
    def MaxPool3DGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MaxPool3DGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MaxPool3DGradGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MaxPool3DGradGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MaxPoolGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MaxPoolGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MaxPoolGradGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MaxPoolGradGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MaxPoolGradGradV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MaxPoolGradGradV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MaxPoolGradGradWithArgmax(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MaxPoolGradGradWithArgmax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MaxPoolGradV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MaxPoolGradV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MaxPoolGradWithArgmax(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MaxPoolGradWithArgmax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MaxPoolV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MaxPoolV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MaxPoolWithArgmax(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MaxPoolWithArgmax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Maximum(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Maximum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return keras.layers.Maximum(*args, **kwargs)

    @staticmethod
    def Mean(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Mean` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return keras.metrics.Mean(*args, **kwargs)

    @staticmethod
    def Merge(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Merge` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MergeSummary(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MergeSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MergeV2Checkpoints(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MergeV2Checkpoints` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Mfcc(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Mfcc` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Min(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Min` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Minimum(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Minimum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return keras.layers.Minimum(*args, **kwargs)

    @staticmethod
    def MirrorPad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MirrorPad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MirrorPadGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MirrorPadGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Mod(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Mod` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ModelDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ModelDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Mul(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Mul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MulNoNan(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MulNoNan` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MultiDeviceIterator(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MultiDeviceIterator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MultiDeviceIteratorFromStringHandle(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MultiDeviceIteratorFromStringHandle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MultiDeviceIteratorGetNextFromShard(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MultiDeviceIteratorGetNextFromShard` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MultiDeviceIteratorInit(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MultiDeviceIteratorInit` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MultiDeviceIteratorToStringHandle(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MultiDeviceIteratorToStringHandle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Multinomial(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Multinomial` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MutableDenseHashTable(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MutableDenseHashTable` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MutableDenseHashTableV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MutableDenseHashTableV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MutableHashTable(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MutableHashTable` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MutableHashTableOfTensors(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MutableHashTableOfTensors` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MutableHashTableOfTensorsV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.MutableHashTableOfTensorsV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MutableHashTableV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MutableHashTableV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MutexLock(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MutexLock` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MutexV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.MutexV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NcclAllReduce(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.NcclAllReduce` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NcclBroadcast(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.NcclBroadcast` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NcclReduce(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.NcclReduce` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Ndtri(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Ndtri` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NearestNeighbors(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.NearestNeighbors` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Neg(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Neg` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NextAfter(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.NextAfter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NextIteration(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.NextIteration` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NoOp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.NoOp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NonDeterministicInts(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.NonDeterministicInts` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NonMaxSuppression(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.NonMaxSuppression` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.NonMaxSuppression(*args, **kwargs)

    @staticmethod
    def NonMaxSuppressionV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.NonMaxSuppressionV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NonMaxSuppressionV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.NonMaxSuppressionV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NonMaxSuppressionV4(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.NonMaxSuppressionV4` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NonMaxSuppressionV5(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.NonMaxSuppressionV5` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NonMaxSuppressionWithOverlaps(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.NonMaxSuppressionWithOverlaps` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NonSerializableDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.NonSerializableDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NotEqual(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.NotEqual` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NthElement(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.NthElement` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OneHot(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OneHot` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OneShotIterator(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OneShotIterator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OnesLike(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OnesLike` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OptimizeDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OptimizeDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OptimizeDatasetV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OptimizeDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OptionalFromValue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OptionalFromValue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OptionalGetValue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OptionalGetValue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OptionalHasValue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OptionalHasValue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OptionalNone(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OptionalNone` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OptionsDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OptionsDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OrderedMapClear(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OrderedMapClear` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OrderedMapIncompleteSize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.OrderedMapIncompleteSize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OrderedMapPeek(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OrderedMapPeek` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OrderedMapSize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OrderedMapSize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OrderedMapStage(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OrderedMapStage` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OrderedMapUnstage(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OrderedMapUnstage` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OrderedMapUnstageNoKey(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.OrderedMapUnstageNoKey` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OutfeedDequeue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OutfeedDequeue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OutfeedDequeueTuple(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.OutfeedDequeueTuple` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OutfeedDequeueTupleV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.OutfeedDequeueTupleV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OutfeedDequeueV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OutfeedDequeueV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OutfeedEnqueue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.OutfeedEnqueue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OutfeedEnqueueTuple(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.OutfeedEnqueueTuple` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Pack(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Pack` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Pad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Pad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PadV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PadV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PaddedBatchDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PaddedBatchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PaddedBatchDatasetV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.PaddedBatchDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PaddingFIFOQueue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PaddingFIFOQueue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PaddingFIFOQueueV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PaddingFIFOQueueV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParallelBatchDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParallelBatchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParallelConcat(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ParallelConcat` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParallelDynamicStitch(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParallelDynamicStitch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParallelFilterDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParallelFilterDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParallelInterleaveDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParallelInterleaveDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParallelInterleaveDatasetV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParallelInterleaveDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParallelInterleaveDatasetV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParallelInterleaveDatasetV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParallelInterleaveDatasetV4(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParallelInterleaveDatasetV4` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParallelMapDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ParallelMapDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParallelMapDatasetV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParallelMapDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParameterizedTruncatedNormal(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParameterizedTruncatedNormal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParseExample(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ParseExample` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParseExampleDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParseExampleDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParseExampleDatasetV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParseExampleDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParseExampleV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ParseExampleV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParseSequenceExample(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParseSequenceExample` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParseSequenceExampleV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParseSequenceExampleV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParseSingleExample(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ParseSingleExample` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParseSingleSequenceExample(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ParseSingleSequenceExample` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ParseTensor(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ParseTensor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PartitionedCall(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PartitionedCall` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Placeholder(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Placeholder` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PlaceholderV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PlaceholderV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PlaceholderWithDefault(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.PlaceholderWithDefault` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Polygamma(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Polygamma` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PopulationCount(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.PopulationCount` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.PopulationCount(*args, **kwargs)

    @staticmethod
    def Pow(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Pow` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PrefetchDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PrefetchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Prelinearize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Prelinearize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PrelinearizeTuple(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PrelinearizeTuple` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PreventGradient(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PreventGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Print(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Print` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PrintV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PrintV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PriorityQueue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PriorityQueue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PriorityQueueV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PriorityQueueV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PrivateThreadPoolDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.PrivateThreadPoolDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Prod(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Prod` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PyFunc(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PyFunc` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PyFuncStateless(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.PyFuncStateless` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Qr(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Qr` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizeAndDequantize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizeAndDequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizeAndDequantizeV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizeAndDequantizeV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizeAndDequantizeV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizeAndDequantizeV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizeAndDequantizeV4(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizeAndDequantizeV4` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizeAndDequantizeV4Grad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizeAndDequantizeV4Grad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizeDownAndShrinkRange(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizeDownAndShrinkRange` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizeV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizeV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedAdd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizedAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedAvgPool(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizedAvgPool` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedBatchNormWithGlobalNormalization(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedBatchNormWithGlobalNormalization` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedBiasAdd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizedBiasAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConcat(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizedConcat` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConv2D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizedConv2D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConv2DAndRelu(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedConv2DAndRelu` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConv2DAndReluAndRequantize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedConv2DAndReluAndRequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConv2DAndRequantize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedConv2DAndRequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConv2DPerChannel(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedConv2DPerChannel` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConv2DWithBias(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedConv2DWithBias` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConv2DWithBiasAndRelu(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedConv2DWithBiasAndRelu` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConv2DWithBiasAndReluAndRequantize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedConv2DWithBiasAndReluAndRequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConv2DWithBiasAndRequantize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedConv2DWithBiasAndRequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConv2DWithBiasSignedSumAndReluAndRequantize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedConv2DWithBiasSignedSumAndReluAndRequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConv2DWithBiasSumAndRelu(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedConv2DWithBiasSumAndRelu` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedConv2DWithBiasSumAndReluAndRequantize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedConv2DWithBiasSumAndReluAndRequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedDepthwiseConv2D(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedDepthwiseConv2D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedDepthwiseConv2DWithBias(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedDepthwiseConv2DWithBias` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedDepthwiseConv2DWithBiasAndRelu(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedDepthwiseConv2DWithBiasAndRelu` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedInstanceNorm(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedInstanceNorm` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedMatMul(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizedMatMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedMatMulWithBias(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedMatMulWithBias` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedMatMulWithBiasAndDequantize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedMatMulWithBiasAndDequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedMatMulWithBiasAndRelu(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedMatMulWithBiasAndRelu` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedMatMulWithBiasAndReluAndRequantize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedMatMulWithBiasAndReluAndRequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedMatMulWithBiasAndRequantize(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedMatMulWithBiasAndRequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedMaxPool(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizedMaxPool` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedMul(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizedMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedRelu(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizedRelu` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedRelu6(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizedRelu6` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedReluX(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizedReluX` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedReshape(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QuantizedReshape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QuantizedResizeBilinear(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.QuantizedResizeBilinear` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueClose(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueClose` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueCloseV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueCloseV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueDequeue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueDequeue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueDequeueMany(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueDequeueMany` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueDequeueManyV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueDequeueManyV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueDequeueUpTo(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueDequeueUpTo` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueDequeueUpToV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueDequeueUpToV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueDequeueV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueDequeueV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueEnqueue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueEnqueue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueEnqueueMany(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueEnqueueMany` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueEnqueueManyV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueEnqueueManyV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueEnqueueV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueEnqueueV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueIsClosed(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueIsClosed` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueIsClosedV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueIsClosedV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueSize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueSize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueSizeV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.QueueSizeV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RFFT(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RFFT` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RFFT2D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RFFT2D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RFFT3D(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RFFT3D` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RFFTND(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RFFTND` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RGBToHSV(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RGBToHSV` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RaggedBincount(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RaggedBincount` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RaggedCountSparseOutput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RaggedCountSparseOutput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RaggedCross(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RaggedCross` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RaggedFillEmptyRows(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RaggedFillEmptyRows` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RaggedFillEmptyRowsGrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RaggedFillEmptyRowsGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RaggedGather(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.RaggedGather` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.RaggedGather(*args, **kwargs)

    @staticmethod
    def RaggedRange(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.RaggedRange` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.RaggedRange(*args, **kwargs)

    @staticmethod
    def RaggedTensorFromVariant(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RaggedTensorFromVariant` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RaggedTensorToSparse(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RaggedTensorToSparse` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RaggedTensorToTensor(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RaggedTensorToTensor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RaggedTensorToVariant(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RaggedTensorToVariant` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RaggedTensorToVariantGradient(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RaggedTensorToVariantGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomCrop(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.RandomCrop` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return keras.layers.RandomCrop(*args, **kwargs)

    @staticmethod
    def RandomDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RandomDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomDatasetV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RandomDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomGamma(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RandomGamma` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomGammaGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RandomGammaGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomIndexShuffle(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RandomIndexShuffle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomPoisson(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RandomPoisson` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomPoissonV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RandomPoissonV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomShuffle(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RandomShuffle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomShuffleQueue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RandomShuffleQueue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomShuffleQueueV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RandomShuffleQueueV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomStandardNormal(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RandomStandardNormal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomUniform(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RandomUniform` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomUniformInt(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RandomUniformInt` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Range(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Range` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RangeDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RangeDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Rank(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Rank` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReadFile(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReadFile` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReadVariableOp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReadVariableOp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReadVariableXlaSplitND(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ReadVariableXlaSplitND` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderNumRecordsProduced(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ReaderNumRecordsProduced` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderNumRecordsProducedV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ReaderNumRecordsProducedV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderNumWorkUnitsCompleted(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ReaderNumWorkUnitsCompleted` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderNumWorkUnitsCompletedV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ReaderNumWorkUnitsCompletedV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderRead(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReaderRead` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderReadUpTo(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReaderReadUpTo` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderReadUpToV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReaderReadUpToV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderReadV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReaderReadV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderReset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReaderReset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderResetV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReaderResetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderRestoreState(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReaderRestoreState` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderRestoreStateV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ReaderRestoreStateV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderSerializeState(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ReaderSerializeState` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReaderSerializeStateV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ReaderSerializeStateV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Real(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Real` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RealDiv(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RealDiv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RebatchDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RebatchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RebatchDatasetV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RebatchDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Reciprocal(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Reciprocal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReciprocalGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReciprocalGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RecordInput(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RecordInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Recv(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Recv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RecvTPUEmbeddingActivations(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RecvTPUEmbeddingActivations` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReduceDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReduceDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReduceJoin(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReduceJoin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RefEnter(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RefEnter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RefExit(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RefExit` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RefIdentity(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RefIdentity` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RefMerge(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RefMerge` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RefNextIteration(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RefNextIteration` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RefSelect(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RefSelect` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RefSwitch(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RefSwitch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RegexFullMatch(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RegexFullMatch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RegexReplace(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RegexReplace` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RegisterDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RegisterDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RegisterDatasetV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RegisterDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Relu(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Relu` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Relu6(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Relu6` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Relu6Grad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Relu6Grad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReluGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReluGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RemoteCall(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RemoteCall` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RepeatDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RepeatDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RequantizationRange(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RequantizationRange` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RequantizationRangePerChannel(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RequantizationRangePerChannel` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Requantize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Requantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RequantizePerChannel(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RequantizePerChannel` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Reshape(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Reshape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Reshape(*args, **kwargs)

    @staticmethod
    def ResizeArea(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResizeArea` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResizeBicubic(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.ResizeBicubic` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.ResizeBicubic(*args, **kwargs)

    @staticmethod
    def ResizeBicubicGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResizeBicubicGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResizeBilinear(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.ResizeBilinear` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.ResizeBilinear(*args, **kwargs)

    @staticmethod
    def ResizeBilinearGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResizeBilinearGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResizeNearestNeighbor(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResizeNearestNeighbor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResizeNearestNeighborGrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResizeNearestNeighborGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceAccumulatorApplyGradient(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceAccumulatorApplyGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceAccumulatorNumAccumulated(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceAccumulatorNumAccumulated` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceAccumulatorSetGlobalStep(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceAccumulatorSetGlobalStep` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceAccumulatorTakeGradient(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceAccumulatorTakeGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyAdaMax(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyAdaMax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyAdadelta(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyAdadelta` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyAdagrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyAdagrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyAdagradDA(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyAdagradDA` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyAdagradV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyAdagradV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyAdam(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResourceApplyAdam` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyAdamWithAmsgrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyAdamWithAmsgrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyAddSign(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyAddSign` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyCenteredRMSProp(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyCenteredRMSProp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyFtrl(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResourceApplyFtrl` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyFtrlV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyFtrlV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyGradientDescent(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyGradientDescent` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyKerasMomentum(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyKerasMomentum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyMomentum(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyMomentum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyPowerSign(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyPowerSign` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyProximalAdagrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyProximalAdagrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyProximalGradientDescent(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyProximalGradientDescent` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceApplyRMSProp(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceApplyRMSProp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceConditionalAccumulator(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceConditionalAccumulator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceCountUpTo(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResourceCountUpTo` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceGather(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResourceGather` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceGatherNd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResourceGatherNd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceScatterAdd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResourceScatterAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceScatterDiv(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResourceScatterDiv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceScatterMax(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResourceScatterMax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceScatterMin(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResourceScatterMin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceScatterMul(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResourceScatterMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceScatterNdAdd(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceScatterNdAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceScatterNdMax(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceScatterNdMax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceScatterNdMin(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceScatterNdMin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceScatterNdSub(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceScatterNdSub` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceScatterNdUpdate(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceScatterNdUpdate` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceScatterSub(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ResourceScatterSub` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceScatterUpdate(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceScatterUpdate` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceSparseApplyAdadelta(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceSparseApplyAdadelta` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceSparseApplyAdagrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceSparseApplyAdagrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceSparseApplyAdagradDA(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceSparseApplyAdagradDA` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceSparseApplyAdagradV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceSparseApplyAdagradV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceSparseApplyCenteredRMSProp(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceSparseApplyCenteredRMSProp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceSparseApplyFtrl(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceSparseApplyFtrl` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceSparseApplyFtrlV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceSparseApplyFtrlV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceSparseApplyKerasMomentum(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceSparseApplyKerasMomentum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceSparseApplyMomentum(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceSparseApplyMomentum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceSparseApplyProximalAdagrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceSparseApplyProximalAdagrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceSparseApplyProximalGradientDescent(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceSparseApplyProximalGradientDescent` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceSparseApplyRMSProp(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceSparseApplyRMSProp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceStridedSliceAssign(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ResourceStridedSliceAssign` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Restore(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Restore` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RestoreSlice(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RestoreSlice` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RestoreV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RestoreV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingADAMParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingADAMParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingAdadeltaParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingAdadeltaParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingAdagradMomentumParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingAdagradMomentumParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingAdagradParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingAdagradParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingCenteredRMSPropParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingCenteredRMSPropParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingFTRLParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingFTRLParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingFrequencyEstimatorParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingFrequencyEstimatorParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingMDLAdagradLightParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingMDLAdagradLightParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingMomentumParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingMomentumParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingProximalAdagradParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingProximalAdagradParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingProximalYogiParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingProximalYogiParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingRMSPropParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingRMSPropParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RetrieveTPUEmbeddingStochasticGradientDescentParameters(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.RetrieveTPUEmbeddingStochasticGradientDescentParameters` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Reverse(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Reverse` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReverseSequence(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReverseSequence` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ReverseV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ReverseV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RewriteDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RewriteDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RightShift(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RightShift` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Rint(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Rint` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RngReadAndSkip(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RngReadAndSkip` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RngSkip(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RngSkip` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Roll(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Roll` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Roll(*args, **kwargs)

    @staticmethod
    def Round(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Round` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Rsqrt(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Rsqrt` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RsqrtGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.RsqrtGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SampleDistortedBoundingBox(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SampleDistortedBoundingBox` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SampleDistortedBoundingBoxV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SampleDistortedBoundingBoxV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SamplingDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SamplingDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Save(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Save` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SaveDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SaveDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SaveDatasetV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SaveDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SaveSlices(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SaveSlices` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SaveV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SaveV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScalarSummary(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ScalarSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScaleAndTranslate(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ScaleAndTranslate` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScaleAndTranslateGrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ScaleAndTranslateGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScanDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ScanDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScatterAdd(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.ScatterAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.ScatterAdd(*args, **kwargs)

    @staticmethod
    def ScatterDiv(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ScatterDiv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScatterMax(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.ScatterMax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.ScatterMax(*args, **kwargs)

    @staticmethod
    def ScatterMin(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.ScatterMin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.ScatterMin(*args, **kwargs)

    @staticmethod
    def ScatterMul(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.ScatterMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.ScatterMul(*args, **kwargs)

    @staticmethod
    def ScatterNd(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.ScatterNd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.ScatterNd(*args, **kwargs)

    @staticmethod
    def ScatterNdAdd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ScatterNdAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScatterNdMax(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ScatterNdMax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScatterNdMin(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ScatterNdMin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScatterNdNonAliasingAdd(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ScatterNdNonAliasingAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScatterNdSub(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ScatterNdSub` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScatterNdUpdate(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ScatterNdUpdate` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScatterSub(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ScatterSub` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ScatterUpdate(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ScatterUpdate` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SdcaFprint(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SdcaFprint` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SdcaOptimizer(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SdcaOptimizer` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SdcaOptimizerV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SdcaOptimizerV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SdcaShrinkL1(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SdcaShrinkL1` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SegmentMax(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SegmentMax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SegmentMaxV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SegmentMaxV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SegmentMean(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SegmentMean` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SegmentMin(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SegmentMin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SegmentMinV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SegmentMinV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SegmentProd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SegmentProd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SegmentProdV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SegmentProdV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SegmentSum(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SegmentSum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SegmentSumV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SegmentSumV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Select(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Select` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Select(*args, **kwargs)

    @staticmethod
    def SelectV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SelectV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SelfAdjointEig(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SelfAdjointEig` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SelfAdjointEigV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SelfAdjointEigV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Selu(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Selu` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SeluGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SeluGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Send(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Send` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SendTPUEmbeddingGradients(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SendTPUEmbeddingGradients` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SerializeIterator(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SerializeIterator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SerializeManySparse(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SerializeManySparse` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SerializeSparse(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SerializeSparse` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SerializeTensor(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SerializeTensor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SetSize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SetSize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SetStatsAggregatorDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SetStatsAggregatorDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Shape(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Shape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ShapeN(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ShapeN` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ShardDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ShardDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ShardedFilename(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ShardedFilename` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ShardedFilespec(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ShardedFilespec` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ShuffleAndRepeatDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ShuffleAndRepeatDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ShuffleAndRepeatDatasetV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ShuffleAndRepeatDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ShuffleDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ShuffleDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ShuffleDatasetV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ShuffleDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ShuffleDatasetV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ShuffleDatasetV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ShutdownDistributedTPU(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ShutdownDistributedTPU` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Sigmoid(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Sigmoid` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SigmoidGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SigmoidGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Sign(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Sign` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Sin(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Sin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Sinh(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Sinh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Size(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Size` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SkipDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SkipDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SleepDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SleepDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Slice(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Slice` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Slice(*args, **kwargs)

    @staticmethod
    def SlidingWindowDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SlidingWindowDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Snapshot(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Snapshot` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SnapshotChunkDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SnapshotChunkDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SnapshotDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SnapshotDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SnapshotDatasetReader(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SnapshotDatasetReader` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SnapshotDatasetV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SnapshotDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SnapshotNestedDatasetReader(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SnapshotNestedDatasetReader` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SobolSample(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SobolSample` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SobolSample(*args, **kwargs)

    @staticmethod
    def Softmax(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Softmax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return keras.layers.Softmax(*args, **kwargs)

    @staticmethod
    def SoftmaxCrossEntropyWithLogits(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SoftmaxCrossEntropyWithLogits` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Softplus(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Softplus` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SoftplusGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SoftplusGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Softsign(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Softsign` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SoftsignGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SoftsignGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SpaceToBatch(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SpaceToBatch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SpaceToBatchND(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SpaceToBatchND` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SpaceToDepth(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SpaceToDepth` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseAccumulatorApplyGradient(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseAccumulatorApplyGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseAccumulatorTakeGradient(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseAccumulatorTakeGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseAdd(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseAdd(*args, **kwargs)

    @staticmethod
    def SparseAddGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseAddGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseApplyAdadelta(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseApplyAdadelta` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseApplyAdagrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseApplyAdagrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseApplyAdagradDA(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseApplyAdagradDA` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseApplyAdagradV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseApplyAdagradV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseApplyCenteredRMSProp(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseApplyCenteredRMSProp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseApplyFtrl(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseApplyFtrl` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseApplyFtrlV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseApplyFtrlV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseApplyMomentum(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseApplyMomentum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseApplyProximalAdagrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseApplyProximalAdagrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseApplyProximalGradientDescent(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseApplyProximalGradientDescent` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseApplyRMSProp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseApplyRMSProp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseBincount(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseBincount` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseBincount(*args, **kwargs)

    @staticmethod
    def SparseConcat(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseConcat` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseConditionalAccumulator(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseConditionalAccumulator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseCountSparseOutput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseCountSparseOutput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseCross(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseCross` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseCrossHashed(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseCrossHashed` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseCrossHashed(*args, **kwargs)

    @staticmethod
    def SparseCrossV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseCrossV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseDenseCwiseAdd(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseDenseCwiseAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseDenseCwiseDiv(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseDenseCwiseDiv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseDenseCwiseMul(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseDenseCwiseMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseFillEmptyRows(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseFillEmptyRows` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseFillEmptyRows(*args, **kwargs)

    @staticmethod
    def SparseFillEmptyRowsGrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseFillEmptyRowsGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseMatMul(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseMatMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseMatrixAdd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseMatrixAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseMatrixMatMul(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseMatrixMatMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseMatrixMul(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseMatrixMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseMatrixNNZ(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseMatrixNNZ` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseMatrixOrderingAMD(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseMatrixOrderingAMD` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseMatrixSoftmax(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseMatrixSoftmax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseMatrixSoftmaxGrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseMatrixSoftmaxGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseMatrixSparseCholesky(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseMatrixSparseCholesky` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseMatrixSparseMatMul(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseMatrixSparseMatMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseMatrixTranspose(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseMatrixTranspose` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseMatrixZeros(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseMatrixZeros` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseReduceMax(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseReduceMax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseReduceMax(*args, **kwargs)

    @staticmethod
    def SparseReduceMaxSparse(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseReduceMaxSparse` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseReduceSum(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseReduceSum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseReduceSum(*args, **kwargs)

    @staticmethod
    def SparseReduceSumSparse(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseReduceSumSparse` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseReorder(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseReorder` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseReorder(*args, **kwargs)

    @staticmethod
    def SparseReshape(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseReshape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseReshape(*args, **kwargs)

    @staticmethod
    def SparseSegmentMean(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseSegmentMean` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseSegmentMean(*args, **kwargs)

    @staticmethod
    def SparseSegmentMeanGrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseSegmentMeanGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSegmentMeanGradV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseSegmentMeanGradV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSegmentMeanWithNumSegments(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseSegmentMeanWithNumSegments` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSegmentSqrtN(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseSegmentSqrtN` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseSegmentSqrtN(*args, **kwargs)

    @staticmethod
    def SparseSegmentSqrtNGrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseSegmentSqrtNGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSegmentSqrtNGradV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseSegmentSqrtNGradV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSegmentSqrtNWithNumSegments(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseSegmentSqrtNWithNumSegments` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSegmentSum(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseSegmentSum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseSegmentSum(*args, **kwargs)

    @staticmethod
    def SparseSegmentSumGrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseSegmentSumGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSegmentSumGradV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseSegmentSumGradV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSegmentSumWithNumSegments(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseSegmentSumWithNumSegments` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSlice(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseSlice` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseSlice(*args, **kwargs)

    @staticmethod
    def SparseSliceGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseSliceGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSoftmax(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.SparseSoftmax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.SparseSoftmax(*args, **kwargs)

    @staticmethod
    def SparseSoftmaxCrossEntropyWithLogits(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseSoftmaxCrossEntropyWithLogits` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSparseMaximum(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseSparseMaximum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSparseMinimum(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseSparseMinimum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseSplit(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseSplit` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseTensorDenseAdd(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseTensorDenseAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseTensorDenseMatMul(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseTensorDenseMatMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseTensorSliceDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseTensorSliceDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseTensorToCSRSparseMatrix(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseTensorToCSRSparseMatrix` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseToDense(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SparseToDense` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SparseToSparseSetOperation(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.SparseToSparseSetOperation` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Spence(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Spence` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Split(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Split` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Split(*args, **kwargs)

    @staticmethod
    def SplitV(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SplitV` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SqlDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SqlDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Sqrt(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Sqrt` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SqrtGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SqrtGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Square(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Square` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SquaredDifference(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SquaredDifference` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Squeeze(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Squeeze` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Squeeze(*args, **kwargs)

    @staticmethod
    def Stack(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Stack` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Stack(*args, **kwargs)

    @staticmethod
    def StackClose(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StackClose` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StackCloseV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StackCloseV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StackPop(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StackPop` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StackPopV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StackPopV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StackPush(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StackPush` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StackPushV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StackPushV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StackV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StackV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Stage(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Stage` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StageClear(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StageClear` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StagePeek(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StagePeek` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StageSize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StageSize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatefulPartitionedCall(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatefulPartitionedCall` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatefulRandomBinomial(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatefulRandomBinomial` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatefulStandardNormal(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatefulStandardNormal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatefulStandardNormalV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatefulStandardNormalV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatefulTruncatedNormal(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatefulTruncatedNormal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatefulUniform(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StatefulUniform` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatefulUniformFullInt(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatefulUniformFullInt` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatefulUniformInt(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StatefulUniformInt` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessCase(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StatelessCase` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessIf(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StatelessIf` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessMultinomial(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessMultinomial` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessParameterizedTruncatedNormal(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessParameterizedTruncatedNormal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomBinomial(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomBinomial` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomGammaV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomGammaV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomGammaV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomGammaV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomGetAlg(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomGetAlg` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomGetKeyCounter(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomGetKeyCounter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomGetKeyCounterAlg(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomGetKeyCounterAlg` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomNormal(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomNormal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomNormalV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomNormalV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomPoisson(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomPoisson` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomUniform(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomUniform` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomUniformFullInt(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomUniformFullInt` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomUniformFullIntV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomUniformFullIntV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomUniformInt(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomUniformInt` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomUniformIntV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomUniformIntV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessRandomUniformV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessRandomUniformV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessSampleDistortedBoundingBox(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessSampleDistortedBoundingBox` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessShuffle(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StatelessShuffle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessTruncatedNormal(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessTruncatedNormal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessTruncatedNormalV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatelessTruncatedNormalV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatelessWhile(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StatelessWhile` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StaticRegexFullMatch(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StaticRegexFullMatch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StaticRegexReplace(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StaticRegexReplace` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatsAggregatorHandle(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatsAggregatorHandle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatsAggregatorHandleV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatsAggregatorHandleV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatsAggregatorSetSummaryWriter(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatsAggregatorSetSummaryWriter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StatsAggregatorSummary(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StatsAggregatorSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StopGradient(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StopGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StoreMinibatchStatisticsInFdo(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StoreMinibatchStatisticsInFdo` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StridedSlice(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.StridedSlice` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.StridedSlice(*args, **kwargs)

    @staticmethod
    def StridedSliceAssign(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StridedSliceAssign` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StridedSliceGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StridedSliceGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringFormat(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StringFormat` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringJoin(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StringJoin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringLength(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StringLength` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringLower(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StringLower` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringNGrams(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StringNGrams` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringSplit(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StringSplit` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringSplitV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StringSplitV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringStrip(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StringStrip` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringToHashBucket(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StringToHashBucket` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringToHashBucketFast(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StringToHashBucketFast` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringToHashBucketStrong(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.StringToHashBucketStrong` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringToNumber(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StringToNumber` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StringUpper(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.StringUpper` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Sub(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Sub` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Substr(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Substr` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Sum(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Sum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return keras.metrics.Sum(*args, **kwargs)

    @staticmethod
    def SummaryWriter(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SummaryWriter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Svd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Svd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Switch(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Switch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SymbolicGradient(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SymbolicGradient` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def SyncDevice(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.SyncDevice` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TFRecordDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TFRecordDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TFRecordDatasetV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TFRecordDatasetV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TFRecordReader(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TFRecordReader` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TFRecordReaderV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TFRecordReaderV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUAnnotateTensorsWithDynamicShape(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TPUAnnotateTensorsWithDynamicShape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUCompilationResult(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TPUCompilationResult` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUCopyWithDynamicShape(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TPUCopyWithDynamicShape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUEmbeddingActivations(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TPUEmbeddingActivations` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUOrdinalSelector(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TPUOrdinalSelector` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUPartitionedCall(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TPUPartitionedCall` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUPartitionedInput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TPUPartitionedInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUPartitionedInputV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TPUPartitionedInputV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUPartitionedOutput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TPUPartitionedOutput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUPartitionedOutputV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TPUPartitionedOutputV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUReplicateMetadata(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TPUReplicateMetadata` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUReplicatedInput(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TPUReplicatedInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TPUReplicatedOutput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TPUReplicatedOutput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TakeDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TakeDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TakeManySparseFromTensorsMap(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TakeManySparseFromTensorsMap` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TakeWhileDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TakeWhileDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Tan(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Tan` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Tanh(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Tanh` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TanhGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TanhGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TemporaryVariable(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TemporaryVariable` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArray(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArray` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayClose(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayClose` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayCloseV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayCloseV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayCloseV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayCloseV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayConcat(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayConcat` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayConcatV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorArrayConcatV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayConcatV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorArrayConcatV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayGather(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayGather` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayGatherV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorArrayGatherV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayGatherV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorArrayGatherV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayGradV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayGradV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayGradV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayGradV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayGradWithShape(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorArrayGradWithShape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayPack(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayPack` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayRead(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.TensorArrayRead` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.TensorArrayRead(*args, **kwargs)

    @staticmethod
    def TensorArrayReadV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayReadV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayReadV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayReadV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayScatter(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayScatter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayScatterV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorArrayScatterV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayScatterV3(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorArrayScatterV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArraySize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArraySize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArraySizeV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArraySizeV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArraySizeV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArraySizeV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArraySplit(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArraySplit` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArraySplitV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArraySplitV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArraySplitV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArraySplitV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayUnpack(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayUnpack` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayWrite(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.TensorArrayWrite` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.TensorArrayWrite(*args, **kwargs)

    @staticmethod
    def TensorArrayWriteV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayWriteV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorArrayWriteV3(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorArrayWriteV3` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListConcat(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListConcat` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListConcatLists(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorListConcatLists` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListConcatV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListConcatV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListElementShape(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorListElementShape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListFromTensor(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorListFromTensor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListGather(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListGather` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListGetItem(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListGetItem` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListLength(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListLength` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListPopBack(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListPopBack` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListPushBack(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListPushBack` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListPushBackBatch(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorListPushBackBatch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListReserve(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListReserve` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListResize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListResize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListScatter(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListScatter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListScatterIntoExistingList(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorListScatterIntoExistingList` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListScatterV2(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorListScatterV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListSetItem(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListSetItem` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListSplit(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListSplit` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorListStack(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorListStack` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorMapErase(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorMapErase` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorMapHasKey(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorMapHasKey` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorMapInsert(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorMapInsert` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorMapLookup(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorMapLookup` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorMapSize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorMapSize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorMapStackKeys(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorMapStackKeys` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorScatterAdd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorScatterAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorScatterMax(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorScatterMax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorScatterMin(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorScatterMin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorScatterSub(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorScatterSub` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorScatterUpdate(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.TensorScatterUpdate` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.TensorScatterUpdate(*args, **kwargs)

    @staticmethod
    def TensorSliceDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorSliceDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorStridedSliceUpdate(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.TensorStridedSliceUpdate` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorSummary(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TensorSummaryV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TensorSummaryV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TextLineDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TextLineDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TextLineReader(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TextLineReader` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TextLineReaderV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TextLineReaderV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ThreadPoolDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ThreadPoolDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ThreadPoolHandle(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ThreadPoolHandle` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ThreadUnsafeUnigramCandidateSampler(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.ThreadUnsafeUnigramCandidateSampler` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Tile(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Tile` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Tile(*args, **kwargs)

    @staticmethod
    def TileGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TileGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Timestamp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Timestamp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ToBool(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ToBool` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TopK(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.TopK` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.TopK(*args, **kwargs)

    @staticmethod
    def TopKV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TopKV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Transpose(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Transpose` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Transpose(*args, **kwargs)

    @staticmethod
    def TridiagonalMatMul(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TridiagonalMatMul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TridiagonalSolve(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TridiagonalSolve` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TruncateDiv(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TruncateDiv` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TruncateMod(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TruncateMod` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TruncatedNormal(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.TruncatedNormal` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Unbatch(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Unbatch` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnbatchDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UnbatchDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnbatchGrad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UnbatchGrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UncompressElement(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UncompressElement` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnicodeDecode(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UnicodeDecode` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnicodeDecodeWithOffsets(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.UnicodeDecodeWithOffsets` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnicodeEncode(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UnicodeEncode` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnicodeScript(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UnicodeScript` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnicodeTranscode(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UnicodeTranscode` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniformCandidateSampler(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.UniformCandidateSampler` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniformDequantize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UniformDequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniformQuantize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UniformQuantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniformQuantizedAdd(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.UniformQuantizedAdd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniformQuantizedClipByValue(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.UniformQuantizedClipByValue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniformQuantizedConvolution(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.UniformQuantizedConvolution` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniformQuantizedConvolutionHybrid(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.UniformQuantizedConvolutionHybrid` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniformQuantizedDot(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.UniformQuantizedDot` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniformQuantizedDotHybrid(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.UniformQuantizedDotHybrid` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniformRequantize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UniformRequantize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Unique(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Unique` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniqueDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UniqueDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniqueV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UniqueV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniqueWithCounts(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UniqueWithCounts` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UniqueWithCountsV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UniqueWithCountsV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Unpack(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Unpack` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnravelIndex(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UnravelIndex` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnsortedSegmentJoin(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.UnsortedSegmentJoin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnsortedSegmentMax(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UnsortedSegmentMax` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnsortedSegmentMin(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UnsortedSegmentMin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnsortedSegmentProd(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.UnsortedSegmentProd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnsortedSegmentSum(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UnsortedSegmentSum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Unstage(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Unstage` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnwrapDatasetVariant(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.UnwrapDatasetVariant` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UpperBound(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.UpperBound` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def VarHandleOp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.VarHandleOp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def VarIsInitializedOp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.VarIsInitializedOp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Variable(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Variable` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def VariableShape(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.VariableShape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def VariableV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.VariableV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Where(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `raw_ops.Where` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _ops.Where(*args, **kwargs)

    @staticmethod
    def While(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.While` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WholeFileReader(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.WholeFileReader` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WholeFileReaderV2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.WholeFileReaderV2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WindowDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.WindowDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WindowOp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.WindowOp` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WorkerHeartbeat(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.WorkerHeartbeat` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WrapDatasetVariant(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.WrapDatasetVariant` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WriteAudioSummary(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.WriteAudioSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WriteFile(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.WriteFile` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WriteGraphSummary(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.WriteGraphSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WriteHistogramSummary(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.WriteHistogramSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WriteImageSummary(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.WriteImageSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WriteRawProtoSummary(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.WriteRawProtoSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WriteScalarSummary(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.WriteScalarSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def WriteSummary(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.WriteSummary` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Xdivy(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Xdivy` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaConcatND(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.XlaConcatND` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSparseCoreAdagrad(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.XlaSparseCoreAdagrad` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSparseCoreAdagradMomentum(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.XlaSparseCoreAdagradMomentum` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSparseCoreAdam(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.XlaSparseCoreAdam` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSparseCoreFtrl(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.XlaSparseCoreFtrl` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSparseCoreSgd(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.XlaSparseCoreSgd` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSparseDenseMatmul(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.XlaSparseDenseMatmul` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSparseDenseMatmulGradWithAdagradAndCsrInput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.XlaSparseDenseMatmulGradWithAdagradAndCsrInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSparseDenseMatmulGradWithAdamAndCsrInput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.XlaSparseDenseMatmulGradWithAdamAndCsrInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSparseDenseMatmulGradWithFtrlAndCsrInput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.XlaSparseDenseMatmulGradWithFtrlAndCsrInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSparseDenseMatmulGradWithSgdAndCsrInput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.XlaSparseDenseMatmulGradWithSgdAndCsrInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSparseDenseMatmulWithCsrInput(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `raw_ops.XlaSparseDenseMatmulWithCsrInput` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def XlaSplitND(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.XlaSplitND` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Xlog1py(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Xlog1py` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Xlogy(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Xlogy` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ZerosLike(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ZerosLike` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Zeta(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.Zeta` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ZipDataset(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `raw_ops.ZipDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None
