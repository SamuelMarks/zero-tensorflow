"""TensorFlow saved_model module."""

from typing import Any, Optional

__all__ = [
    "save",
    "load",
]


def save(
    obj: Any,
    export_dir: str,
    signatures: Optional[Any] = None,
    options: Optional[Any] = None,
) -> None:
    """
    Export a tf.Module (and subclasses) obj to SavedModel format.

    Args:
        obj: A trackable object (e.g., tf.Module, tf.keras.Model) to export.
        export_dir: A directory in which to write the SavedModel.
        signatures: Optional, one of three options:
        options: Optional, tf.saved_model.SaveOptions object.
    """
    raise NotImplementedError("Not implemented: tf.saved_model.save")


def load(
    export_dir: str, tags: Optional[Any] = None, options: Optional[Any] = None
) -> Any:
    """
    Load a SavedModel from export_dir.

    Args:
        export_dir: The SavedModel directory to load from.
        tags: A tag or sequence of tags identifying the MetaGraph to load.
        options: Optional, tf.saved_model.LoadOptions object.

    Returns:
        A trackable object that represents the SavedModel.
    """
    raise NotImplementedError("Not implemented: tf.saved_model.load")


# Stubs from TODO_PLAN.md
from typing import Any

ASSETS_DIRECTORY: int = 0
"""Stub for ASSETS_DIRECTORY."""

ASSETS_KEY: int = 0
"""Stub for ASSETS_KEY."""


class Asset:
    """Stub for Asset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: Asset")


CLASSIFY_INPUTS: int = 0
"""Stub for CLASSIFY_INPUTS."""

CLASSIFY_METHOD_NAME: int = 0
"""Stub for CLASSIFY_METHOD_NAME."""

CLASSIFY_OUTPUT_CLASSES: int = 0
"""Stub for CLASSIFY_OUTPUT_CLASSES."""

CLASSIFY_OUTPUT_SCORES: int = 0
"""Stub for CLASSIFY_OUTPUT_SCORES."""

DEBUG_DIRECTORY: int = 0
"""Stub for DEBUG_DIRECTORY."""

DEBUG_INFO_FILENAME_PB: int = 0
"""Stub for DEBUG_INFO_FILENAME_PB."""

DEFAULT_SERVING_SIGNATURE_DEF_KEY: int = 0
"""Stub for DEFAULT_SERVING_SIGNATURE_DEF_KEY."""

GPU: int = 0
"""Stub for GPU."""


class LoadOptions:
    """Stub for LoadOptions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: LoadOptions")


PREDICT_INPUTS: int = 0
"""Stub for PREDICT_INPUTS."""

PREDICT_METHOD_NAME: int = 0
"""Stub for PREDICT_METHOD_NAME."""

PREDICT_OUTPUTS: int = 0
"""Stub for PREDICT_OUTPUTS."""

REGRESS_INPUTS: int = 0
"""Stub for REGRESS_INPUTS."""

REGRESS_METHOD_NAME: int = 0
"""Stub for REGRESS_METHOD_NAME."""

REGRESS_OUTPUTS: int = 0
"""Stub for REGRESS_OUTPUTS."""

SAVED_MODEL_FILENAME_PB: int = 0
"""Stub for SAVED_MODEL_FILENAME_PB."""

SAVED_MODEL_FILENAME_PBTXT: int = 0
"""Stub for SAVED_MODEL_FILENAME_PBTXT."""

SAVED_MODEL_SCHEMA_VERSION: int = 0
"""Stub for SAVED_MODEL_SCHEMA_VERSION."""

SERVING: int = 0
"""Stub for SERVING."""


class SaveOptions:
    """Stub for SaveOptions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: SaveOptions")


TPU: int = 0
"""Stub for TPU."""

TRAINING: int = 0
"""Stub for TRAINING."""

VARIABLES_DIRECTORY: int = 0
"""Stub for VARIABLES_DIRECTORY."""

VARIABLES_FILENAME: int = 0
"""Stub for VARIABLES_FILENAME."""


def contains_saved_model(*args: Any, **kwargs: Any) -> None:
    """Stub for contains_saved_model."""
    raise NotImplementedError("Not implemented: contains_saved_model")


class experimental:
    """Stub for experimental module."""

    class Fingerprint:
        """Stub for Fingerprint."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: Fingerprint")

    class TrackableResource:
        """Stub for TrackableResource."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: TrackableResource")

    class VariablePolicy:
        """Stub for VariablePolicy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: VariablePolicy")

    @staticmethod
    def read_fingerprint(*args: Any, **kwargs: Any) -> None:
        """Stub for read_fingerprint."""
        raise NotImplementedError("Not implemented: read_fingerprint")
