"""TensorFlow `saved_model` serialization and export module."""

from __future__ import annotations

from zero_tensorflow._typing import CallableOperandProtocol, TensorOperand

__all__ = [
    "ASSETS_DIRECTORY",
    "ASSETS_KEY",
    "Asset",
    "CLASSIFY_INPUTS",
    "CLASSIFY_METHOD_NAME",
    "CLASSIFY_OUTPUT_CLASSES",
    "CLASSIFY_OUTPUT_SCORES",
    "DEBUG_DIRECTORY",
    "DEBUG_INFO_FILENAME_PB",
    "DEFAULT_SERVING_SIGNATURE_DEF_KEY",
    "GPU",
    "LoadOptions",
    "PREDICT_INPUTS",
    "PREDICT_METHOD_NAME",
    "PREDICT_OUTPUTS",
    "REGRESS_INPUTS",
    "REGRESS_METHOD_NAME",
    "REGRESS_OUTPUTS",
    "SAVED_MODEL_FILENAME_PB",
    "SAVED_MODEL_FILENAME_PBTXT",
    "SAVED_MODEL_SCHEMA_VERSION",
    "SERVING",
    "SaveOptions",
    "TPU",
    "TRAINING",
    "VARIABLES_DIRECTORY",
    "VARIABLES_FILENAME",
    "contains_saved_model",
    "experimental",
    "load",
    "save",
]


def save(
    obj: CallableOperandProtocol | TensorOperand,
    export_dir: str,
    signatures: TensorOperand = None,
    options: SaveOptions | TensorOperand = None,
) -> None:
    """Export a trackable object or callable to `SavedModel` format.

    Args:
        obj: Trackable object or callable to export.
        export_dir: Directory path in which to write the SavedModel.
        signatures: Optional function signature(s) to export.
        options: Optional `tf.saved_model.SaveOptions` configuration.
    """
    del export_dir, signatures, options
    if callable(obj):
        obj()


def load(
    export_dir: str,
    tags: TensorOperand = None,
    options: LoadOptions | TensorOperand = None,
) -> None:
    """Load a `SavedModel` from `export_dir`.

    Args:
        export_dir: Directory path containing the SavedModel.
        tags: Optional set of string tags identifying the MetaGraphDef.
        options: Optional `tf.saved_model.LoadOptions` configuration.

    Returns:
        None: Stub loaded model handle.
    """
    del export_dir, tags, options
    return None


ASSETS_DIRECTORY: int = 0
"""Subdirectory name containing extra assets."""

ASSETS_KEY: int = 0
"""Collection key for SavedModel assets."""


class Asset:
    """Represents a file asset to be hermetically bundled into a SavedModel."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `Asset`.

        Args:
            *args: Positional asset path arguments.
            **kwargs: Keyword asset arguments.
        """
        self.args = args
        self.kwargs = kwargs


CLASSIFY_INPUTS: int = 0
"""Classification signature input key."""

CLASSIFY_METHOD_NAME: int = 0
"""Classification signature method name."""

CLASSIFY_OUTPUT_CLASSES: int = 0
"""Classification signature output classes key."""

CLASSIFY_OUTPUT_SCORES: int = 0
"""Classification signature output scores key."""

DEBUG_DIRECTORY: int = 0
"""Subdirectory name containing debug information."""

DEBUG_INFO_FILENAME_PB: int = 0
"""Filename for serialized GraphDebugInfo protobuf."""

DEFAULT_SERVING_SIGNATURE_DEF_KEY: int = 0
"""Default signature definition key used for serving."""

GPU: int = 0
"""GPU device tag constant."""


class LoadOptions:
    """Options for loading a `SavedModel`."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `LoadOptions`.

        Args:
            *args: Positional load option arguments.
            **kwargs: Keyword load option arguments.
        """
        self.args = args
        self.kwargs = kwargs


PREDICT_INPUTS: int = 0
"""Prediction signature input key."""

PREDICT_METHOD_NAME: int = 0
"""Prediction signature method name."""

PREDICT_OUTPUTS: int = 0
"""Prediction signature output key."""

REGRESS_INPUTS: int = 0
"""Regression signature input key."""

REGRESS_METHOD_NAME: int = 0
"""Regression signature method name."""

REGRESS_OUTPUTS: int = 0
"""Regression signature output key."""

SAVED_MODEL_FILENAME_PB: int = 0
"""Binary SavedModel protobuf filename."""

SAVED_MODEL_FILENAME_PBTXT: int = 0
"""Text-format SavedModel protobuf filename."""

SAVED_MODEL_SCHEMA_VERSION: int = 0
"""Schema version integer for SavedModel protobufs."""

SERVING: int = 0
"""Serving graph tag constant."""


class SaveOptions:
    """Options for saving a `SavedModel`."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `SaveOptions`.

        Args:
            *args: Positional save option arguments.
            **kwargs: Keyword save option arguments.
        """
        self.args = args
        self.kwargs = kwargs


TPU: int = 0
"""TPU device tag constant."""

TRAINING: int = 0
"""Training graph tag constant."""

VARIABLES_DIRECTORY: int = 0
"""Subdirectory name containing checkpoint variables."""

VARIABLES_FILENAME: int = 0
"""Filename prefix for saved variables."""


def contains_saved_model(*args: TensorOperand, **kwargs: TensorOperand) -> None:
    """Check whether the provided directory contains a `SavedModel`.

    Args:
        *args: Positional directory arguments.
        **kwargs: Keyword arguments.
    """
    del args, kwargs
    return


class experimental:
    """Experimental `saved_model` APIs."""

    class Fingerprint:
        """SavedModel fingerprint metadata."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `Fingerprint`.

            Args:
                *args: Positional fingerprint arguments.
                **kwargs: Keyword fingerprint arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class TrackableResource:
        """Base class for trackable resources saved in a SavedModel."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `TrackableResource`.

            Args:
                *args: Positional resource arguments.
                **kwargs: Keyword resource arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class VariablePolicy:
        """Policy governing how variables are handled when saving."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `VariablePolicy`.

            Args:
                *args: Positional policy arguments.
                **kwargs: Keyword policy arguments.
            """
            self.args = args
            self.kwargs = kwargs

    @staticmethod
    def read_fingerprint(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Read the `Fingerprint` of a SavedModel directory.

        Args:
            *args: Positional export directory arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return
