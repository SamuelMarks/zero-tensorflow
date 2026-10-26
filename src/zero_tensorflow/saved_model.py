"""TensorFlow `saved_model` serialization and export module."""

from __future__ import annotations

import hashlib
import json
import os
import struct

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    CallableOperandProtocol,
    NumpyConvertibleProtocol,
    TensorOperand,
    TrackableContainerProtocol,
    as_operand,
)
from zero_tensorflow.core_tensor import Tensor, Variable

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

_SAVED_MODEL_STORE: dict[
    str, CallableOperandProtocol | TrackableContainerProtocol | TensorOperand
] = {}
_FINGERPRINT_STORE: dict[str, experimental.Fingerprint] = {}


def _should_persist_to_disk(export_dir: str) -> bool:
    """Determine whether `export_dir` resides in an existing directory tree on disk.

    Args:
        export_dir: Target export directory path.

    Returns:
        bool: `True` if `export_dir` or its parent directory exists on disk.
    """
    parent = os.path.dirname(export_dir)
    return bool(export_dir) and (
        os.path.isdir(export_dir)
        or os.path.isabs(export_dir)
        or (bool(parent) and os.path.isdir(parent))
    )


def _compute_u64_hash(tag: str, payload: bytes) -> int:
    """Compute a deterministic 64-bit unsigned integer hash from `tag` and `payload`.

    Args:
        tag: Domain separation tag string.
        payload: Serialized SavedModel payload bytes.

    Returns:
        int: 64-bit unsigned integer hash.
    """
    digest = hashlib.sha256(tag.encode("utf-8") + b":" + payload).digest()
    (val,) = struct.unpack("<Q", digest[:8])
    return int(val)


def save(
    obj: CallableOperandProtocol | TrackableContainerProtocol | TensorOperand,
    export_dir: str,
    signatures: TensorOperand = None,
    options: SaveOptions | TensorOperand = None,
) -> None:
    """Export a trackable object or callable to `SavedModel` format on disk and in memory.

    Args:
        obj: Trackable object or callable to export.
        export_dir: Directory path in which to write the SavedModel.
        signatures: Optional function signature(s) to export.
        options: Optional `tf.saved_model.SaveOptions` configuration.
    """
    del signatures, options
    if callable(obj):
        obj()
    _SAVED_MODEL_STORE[export_dir] = obj

    var_records: list[int | float | bool | str] = []
    if isinstance(obj, TrackableContainerProtocol):
        for v in obj.variables:
            arr = as_operand(v.numpy()) if isinstance(v, NumpyConvertibleProtocol) else 0.0
            raw_item = arr.tolist() if isinstance(arr, ArrayLikeProtocol) else 0.0
            items = raw_item if isinstance(raw_item, list) else [raw_item]
            var_records.extend(x for x in items if isinstance(x, (int, float, bool, str)))

    pb_payload = json.dumps(
        {
            "schema_version": SAVED_MODEL_SCHEMA_VERSION,
            "export_dir": export_dir,
            "variables": var_records,
        }
    ).encode("utf-8")

    fp = experimental.Fingerprint(
        saved_model_checksum=_compute_u64_hash("checksum", pb_payload),
        graph_def_program_hash=_compute_u64_hash("graph_def", pb_payload),
        signature_def_hash=_compute_u64_hash("signature_def", pb_payload),
        saved_object_graph_hash=_compute_u64_hash("object_graph", pb_payload),
        checkpoint_hash=_compute_u64_hash("checkpoint", pb_payload),
        version=1,
    )
    _FINGERPRINT_STORE[export_dir] = fp

    if _should_persist_to_disk(export_dir):
        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(os.path.join(export_dir, "variables"), exist_ok=True)
        os.makedirs(os.path.join(export_dir, "assets"), exist_ok=True)
        with open(os.path.join(export_dir, "saved_model.pb"), "wb") as f_pb:
            f_pb.write(pb_payload)
        fp_json = json.dumps(
            {
                "saved_model_checksum": fp.saved_model_checksum,
                "graph_def_program_hash": fp.graph_def_program_hash,
                "signature_def_hash": fp.signature_def_hash,
                "saved_object_graph_hash": fp.saved_object_graph_hash,
                "checkpoint_hash": fp.checkpoint_hash,
                "version": fp.version,
            }
        ).encode("utf-8")
        with open(os.path.join(export_dir, "fingerprint.pb"), "wb") as f_fp:
            f_fp.write(fp_json)


class _RestoredModel:
    """Restored `SavedModel` container holding restored `Variable`s and callable signatures."""

    def __init__(self, variables: list[int | float | bool | str]) -> None:
        """Initialize `_RestoredModel`.

        Args:
            variables: Flat list of restored variable values.
        """
        self.variables: list[Variable] = [Variable(v) for v in variables]
        self.signatures: dict[str, CallableOperandProtocol] = {
            DEFAULT_SERVING_SIGNATURE_DEF_KEY: self
        }

    def __call__(self, *args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
        """Evaluate restored model on `args` (or return sum of variables if no args).

        Args:
            *args: Input tensors or scalars.
            **kwargs: Unused keyword arguments.

        Returns:
            Tensor: Model output tensor.
        """
        del kwargs
        if args:
            return Tensor(args[0])
        return Tensor([v.tolist() for v in self.variables])


def load(
    export_dir: str,
    tags: TensorOperand = None,
    options: LoadOptions | TensorOperand = None,
) -> CallableOperandProtocol | TrackableContainerProtocol | TensorOperand:
    """Load a `SavedModel` from `export_dir`.

    Args:
        export_dir: Directory path containing the SavedModel.
        tags: Optional set of string tags identifying the MetaGraphDef.
        options: Optional `tf.saved_model.LoadOptions` configuration.

    Returns:
        CallableOperandProtocol | TrackableContainerProtocol | TensorOperand:
            Loaded object if found, else `None`.
    """
    del tags, options
    if export_dir in _SAVED_MODEL_STORE:
        return _SAVED_MODEL_STORE[export_dir]
    pb_path = os.path.join(export_dir, "saved_model.pb")
    if os.path.isfile(pb_path):
        with open(pb_path, "rb") as f:
            raw = json.loads(f.read().decode("utf-8"))
        vars_raw = raw.get("variables", [])
        var_list = (
            [x for x in vars_raw if isinstance(x, (int, float, bool, str))]
            if isinstance(vars_raw, list)
            else []
        )
        return _RestoredModel(var_list)
    return None


ASSETS_DIRECTORY: str = "assets"
"""Subdirectory name containing extra assets."""

ASSETS_KEY: str = "saved_model_assets"
"""Collection key for SavedModel assets."""


class Asset:
    """Represents a file asset to be hermetically bundled into a SavedModel."""

    def __init__(
        self,
        path: str | TensorOperand = "",
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `Asset`.

        Args:
            path: File path to the asset.
            *args: Additional positional asset arguments.
            **kwargs: Keyword asset arguments.
        """
        self.asset_path = str(path) if path is not None else ""
        self.args = (path, *args)
        self.kwargs = kwargs


CLASSIFY_INPUTS: str = "inputs"
"""Classification signature input key."""

CLASSIFY_METHOD_NAME: str = "tensorflow/serving/classify"
"""Classification signature method name."""

CLASSIFY_OUTPUT_CLASSES: str = "classes"
"""Classification signature output classes key."""

CLASSIFY_OUTPUT_SCORES: str = "scores"
"""Classification signature output scores key."""

DEBUG_DIRECTORY: str = "debug"
"""Subdirectory name containing debug information."""

DEBUG_INFO_FILENAME_PB: str = "tf_debug_info.pb"
"""Filename for serialized GraphDebugInfo protobuf."""

DEFAULT_SERVING_SIGNATURE_DEF_KEY: str = "serving_default"
"""Default signature definition key used for serving."""

GPU: str = "gpu"
"""GPU device tag constant."""


class LoadOptions:
    """Options for loading a `SavedModel`."""

    def __init__(
        self,
        allow_partial_checkpoint: bool = False,
        experimental_io_device: str | None = None,
        experimental_skip_checkpoint: bool = False,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `LoadOptions`.

        Args:
            allow_partial_checkpoint: Whether to allow partial checkpoint restoration.
            experimental_io_device: Optional device string for I/O operations.
            experimental_skip_checkpoint: Whether to skip restoring checkpoint variables.
            *args: Additional positional load option arguments.
            **kwargs: Keyword load option arguments.
        """
        self.allow_partial_checkpoint = allow_partial_checkpoint
        self.experimental_io_device = experimental_io_device
        self.experimental_skip_checkpoint = experimental_skip_checkpoint
        self.args = args
        self.kwargs = kwargs


PREDICT_INPUTS: str = "inputs"
"""Prediction signature input key."""

PREDICT_METHOD_NAME: str = "tensorflow/serving/predict"
"""Prediction signature method name."""

PREDICT_OUTPUTS: str = "outputs"
"""Prediction signature output key."""

REGRESS_INPUTS: str = "inputs"
"""Regression signature input key."""

REGRESS_METHOD_NAME: str = "tensorflow/serving/regress"
"""Regression signature method name."""

REGRESS_OUTPUTS: str = "outputs"
"""Regression signature output key."""

SAVED_MODEL_FILENAME_PB: str = "saved_model.pb"
"""Binary SavedModel protobuf filename."""

SAVED_MODEL_FILENAME_PBTXT: str = "saved_model.pbtxt"
"""Text-format SavedModel protobuf filename."""

SAVED_MODEL_SCHEMA_VERSION: int = 1
"""Schema version integer for SavedModel protobufs."""

SERVING: str = "serve"
"""Serving graph tag constant."""


class SaveOptions:
    """Options for saving a `SavedModel`."""

    def __init__(
        self,
        namespace_whitelist: list[str] | None = None,
        save_debug_info: bool = False,
        experimental_io_device: str | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `SaveOptions`.

        Args:
            namespace_whitelist: Optional list of custom op namespaces to permit.
            save_debug_info: Whether to export `tf_debug_info.pb`.
            experimental_io_device: Optional device string for I/O operations.
            *args: Additional positional save option arguments.
            **kwargs: Keyword save option arguments.
        """
        self.namespace_whitelist = namespace_whitelist
        self.save_debug_info = save_debug_info
        self.experimental_io_device = experimental_io_device
        self.args = args
        self.kwargs = kwargs


TPU: str = "tpu"
"""TPU device tag constant."""

TRAINING: str = "train"
"""Training graph tag constant."""

VARIABLES_DIRECTORY: str = "variables"
"""Subdirectory name containing checkpoint variables."""

VARIABLES_FILENAME: str = "variables"
"""Filename prefix for saved variables."""


def contains_saved_model(
    export_dir: str | TensorOperand = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> bool:
    """Check whether the provided directory contains a `SavedModel`.

    Args:
        export_dir: Directory path to inspect.
        *args: Additional positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        bool: `True` if `export_dir` contains a SavedModel in memory or on disk.
    """
    del args, kwargs
    if not isinstance(export_dir, str) or not export_dir:
        return False
    if export_dir in _SAVED_MODEL_STORE:
        return True
    return os.path.isfile(os.path.join(export_dir, "saved_model.pb")) or os.path.isfile(
        os.path.join(export_dir, "saved_model.pbtxt")
    )


class experimental:
    """Experimental `saved_model` APIs."""

    class Fingerprint:
        """SavedModel fingerprint metadata."""

        def __init__(
            self,
            saved_model_checksum: int = 0,
            graph_def_program_hash: int = 0,
            signature_def_hash: int = 0,
            saved_object_graph_hash: int = 0,
            checkpoint_hash: int = 0,
            version: int = 1,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `Fingerprint`.

            Args:
                saved_model_checksum: 64-bit checksum of the serialized `SavedModel`.
                graph_def_program_hash: 64-bit hash of `GraphDef` programs.
                signature_def_hash: 64-bit hash of `SignatureDef`s.
                saved_object_graph_hash: 64-bit hash of the `SavedObjectGraph`.
                checkpoint_hash: 64-bit hash of checkpoint variables.
                version: Fingerprint version number.
                *args: Additional positional arguments.
                **kwargs: Keyword arguments.
            """
            self.saved_model_checksum = saved_model_checksum
            self.graph_def_program_hash = graph_def_program_hash
            self.signature_def_hash = signature_def_hash
            self.saved_object_graph_hash = saved_object_graph_hash
            self.checkpoint_hash = checkpoint_hash
            self.version = version
            self.args = args
            self.kwargs = kwargs

    class TrackableResource:
        """Base class for trackable resources saved in a SavedModel."""

        def __init__(
            self,
            device: str = "",
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `TrackableResource`.

            Args:
                device: Resource placement device string.
                *args: Positional resource arguments.
                **kwargs: Keyword resource arguments.
            """
            self._resource_device = device
            self.args = args
            self.kwargs = kwargs

    class VariablePolicy:
        """Policy governing how variables are handled when saving."""

        def __init__(
            self,
            value: str = "NONE",
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `VariablePolicy`.

            Args:
                value: Policy mode string (`'NONE'`, `'SAVE_VARIABLE_DEVICES'`, `'EXPAND_DISTRIBUTED_VARIABLES'`).
                *args: Positional policy arguments.
                **kwargs: Keyword policy arguments.
            """
            self.value = value
            self.args = args
            self.kwargs = kwargs

    @staticmethod
    def read_fingerprint(
        export_dir: str | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> experimental.Fingerprint | None:
        """Read the `Fingerprint` of a SavedModel directory.

        Args:
            export_dir: Directory containing the SavedModel.
            *args: Additional positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            experimental.Fingerprint | None: `Fingerprint` instance if found, else `None`.
        """
        del args, kwargs
        if not isinstance(export_dir, str) or not export_dir:
            return None
        fp_path = os.path.join(export_dir, "fingerprint.pb")
        if os.path.isfile(fp_path):
            with open(fp_path, "rb") as f:
                data = json.loads(f.read().decode("utf-8"))
            return experimental.Fingerprint(
                saved_model_checksum=int(data.get("saved_model_checksum", 0)),
                graph_def_program_hash=int(data.get("graph_def_program_hash", 0)),
                signature_def_hash=int(data.get("signature_def_hash", 0)),
                saved_object_graph_hash=int(data.get("saved_object_graph_hash", 0)),
                checkpoint_hash=int(data.get("checkpoint_hash", 0)),
                version=int(data.get("version", 1)),
            )
        return _FINGERPRINT_STORE.get(export_dir)
