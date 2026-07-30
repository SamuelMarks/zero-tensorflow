"""TensorFlow saved_model module."""

from __future__ import annotations

from typing import Any

__all__ = [
    "load",
    "save",
]


def save(
    obj: Any,
    export_dir: str,
    signatures: Any | None = None,
    options: Any | None = None,
) -> None:
    try:
        from ml_switcheroo_compiler.export.graph import export_logical_graph
        from ml_switcheroo_compiler.tracing.tracer import trace

        # Just a placeholder for tracing
        # In reality we would trace the signatures or the __call__ method
        if callable(obj):
            graph = trace(obj)
            export_logical_graph(graph, export_dir + "/saved_model.json")
    except ImportError:
        pass


def load(export_dir: str, tags=None, options=None) -> Any:
    return None


# Stubs from TODO_PLAN.md

ASSETS_DIRECTORY: int = 0
"""Stub for ASSETS_DIRECTORY."""

ASSETS_KEY: int = 0
"""Stub for ASSETS_KEY."""


class Asset:
    """Stub for Asset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


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
        pass


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
        pass


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
    return


class experimental:
    """Stub for experimental module."""

    class Fingerprint:
        """Stub for Fingerprint."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

    class TrackableResource:
        """Stub for TrackableResource."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class VariablePolicy:
        """Stub for VariablePolicy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    @staticmethod
    def read_fingerprint(*args: Any, **kwargs: Any) -> None:
        """Stub for read_fingerprint."""
        return
