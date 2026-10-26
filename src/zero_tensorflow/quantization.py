"""TensorFlow model quantization module."""

from __future__ import annotations

import os
import shutil

from zero_tensorflow._typing import TensorOperand

__all__ = ["experimental"]


class experimental:
    """Experimental quantization specifications and options."""

    class QuantizationComponentSpec:
        """Specification for quantizing a specific model component."""

        def __init__(
            self,
            quantization_component: int = 0,
            weight_type: int = 0,
            tensor_type: int = 0,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `QuantizationComponentSpec`.

            Args:
                quantization_component: Target component enum value.
                weight_type: Target weight data type enum value.
                tensor_type: Target activation tensor data type enum value.
                *args: Positional component specification arguments.
                **kwargs: Keyword component specification arguments.
            """
            self.quantization_component = quantization_component
            self.weight_type = weight_type
            self.tensor_type = tensor_type
            self.args = args
            self.kwargs = kwargs

    class QuantizationMethod:
        """Quantization scheme configuration."""

        METHOD_UNSPECIFIED: int = 0
        STATIC_RANGE: int = 1
        DYNAMIC_RANGE: int = 2
        WEIGHT_ONLY: int = 3

        def __init__(
            self,
            preset_method: int = 0,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `QuantizationMethod`.

            Args:
                preset_method: Preset quantization method constant.
                *args: Positional quantization method arguments.
                **kwargs: Keyword quantization method arguments.
            """
            self.preset_method = preset_method
            self.args = args
            self.kwargs = kwargs

    class QuantizationOptions:
        """Configuration options for SavedModel quantization."""

        def __init__(
            self,
            quantization_method: experimental.QuantizationMethod | None = None,
            op_set: str = "TFLITE",
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `QuantizationOptions`.

            Args:
                quantization_method: Optional `QuantizationMethod` instance.
                op_set: Target operator set name (`"TFLITE"`, `"XLA"`, `"UNIFORM_QUANTIZED"`).
                *args: Positional quantization option arguments.
                **kwargs: Keyword quantization option arguments.
            """
            self.quantization_method = (
                quantization_method
                if quantization_method is not None
                else experimental.QuantizationMethod()
            )
            self.op_set = op_set
            self.args = args
            self.kwargs = kwargs

    class TfRecordRepresentativeDatasetSaver:
        """Saves representative calibration datasets in TFRecord format."""

        def __init__(
            self,
            path_map: dict[str, str] | None = None,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `TfRecordRepresentativeDatasetSaver`.

            Args:
                path_map: Mapping from signature keys to TFRecord file paths.
                *args: Positional dataset saver arguments.
                **kwargs: Keyword dataset saver arguments.
            """
            self.path_map: dict[str, str] = dict(path_map) if path_map is not None else {}
            self.args = args
            self.kwargs = kwargs

    class UnitWiseQuantizationSpec:
        """Per-unit or per-channel quantization specification."""

        def __init__(
            self,
            node_name: str = "",
            is_per_channel: bool = True,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `UnitWiseQuantizationSpec`.

            Args:
                node_name: Target graph node name or regex pattern.
                is_per_channel: Whether per-channel quantization is enabled.
                *args: Positional unit specification arguments.
                **kwargs: Keyword unit specification arguments.
            """
            self.node_name = node_name
            self.is_per_channel = is_per_channel
            self.args = args
            self.kwargs = kwargs

    @staticmethod
    def quantize_saved_model(
        saved_model_path: str = "",
        output_directory: str = "",
        quantization_options: experimental.QuantizationOptions | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[str, str] | None:
        """Quantize a SavedModel according to provided quantization options.

        Args:
            saved_model_path: Source SavedModel directory path.
            output_directory: Destination quantized SavedModel directory path.
            quantization_options: Optional `QuantizationOptions` configuration.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, str] | None: Quantized model descriptor when `saved_model_path` is provided, else `None`.
        """
        del args, kwargs
        if saved_model_path:
            if output_directory and os.path.isdir(saved_model_path):
                shutil.copytree(saved_model_path, output_directory, dirs_exist_ok=True)
            opts = (
                quantization_options
                if quantization_options is not None
                else experimental.QuantizationOptions()
            )
            return {
                "saved_model_path": saved_model_path,
                "output_directory": output_directory,
                "op_set": opts.op_set,
            }
        return None
