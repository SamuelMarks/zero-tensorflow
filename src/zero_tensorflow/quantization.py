"""TensorFlow model quantization module."""

from __future__ import annotations

from zero_tensorflow._typing import TensorOperand

__all__ = ["experimental"]


class experimental:
    """Experimental quantization specifications and options."""

    class QuantizationComponentSpec:
        """Specification for quantizing a specific model component."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `QuantizationComponentSpec`.

            Args:
                *args: Positional component specification arguments.
                **kwargs: Keyword component specification arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class QuantizationMethod:
        """Quantization scheme configuration."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `QuantizationMethod`.

            Args:
                *args: Positional quantization method arguments.
                **kwargs: Keyword quantization method arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class QuantizationOptions:
        """Configuration options for SavedModel quantization."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `QuantizationOptions`.

            Args:
                *args: Positional quantization option arguments.
                **kwargs: Keyword quantization option arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class TfRecordRepresentativeDatasetSaver:
        """Saves representative calibration datasets in TFRecord format."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `TfRecordRepresentativeDatasetSaver`.

            Args:
                *args: Positional dataset saver arguments.
                **kwargs: Keyword dataset saver arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class UnitWiseQuantizationSpec:
        """Per-unit or per-channel quantization specification."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `UnitWiseQuantizationSpec`.

            Args:
                *args: Positional unit specification arguments.
                **kwargs: Keyword unit specification arguments.
            """
            self.args = args
            self.kwargs = kwargs

    @staticmethod
    def quantize_saved_model(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Quantize a SavedModel according to provided quantization options.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        return
