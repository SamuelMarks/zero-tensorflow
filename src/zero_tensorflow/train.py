"""TensorFlow `train` checkpointing and training utilities module."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    AssignableProtocol,
    CallableOperandProtocol,
    NumpyConvertibleProtocol,
    TensorOperand,
    TrackableContainerProtocol,
    TrackableLike,
    as_operand,
)

__all__ = [
    "BytesList",
    "Checkpoint",
    "CheckpointManager",
    "CheckpointOptions",
    "CheckpointView",
    "ClusterDef",
    "ClusterSpec",
    "Coordinator",
    "Example",
    "ExponentialMovingAverage",
    "Feature",
    "FeatureList",
    "FeatureLists",
    "Features",
    "FloatList",
    "Int64List",
    "JobDef",
    "SequenceExample",
    "ServerDef",
    "TrackableView",
    "checkpoints_iterator",
    "experimental",
    "get_checkpoint_state",
    "latest_checkpoint",
    "list_variables",
    "load_checkpoint",
    "load_variable",
]

_CHECKPOINT_STORE: dict[str, dict[str, TensorOperand]] = {}
_DIRECTORY_LATEST: dict[str, str] = {}
_DIRECTORY_ALL: dict[str, list[str]] = {}


def _to_json_serializable(
    val: TensorOperand,
) -> int | float | bool | str | list[float]:
    """Convert a `TensorOperand` into a JSON-serializable scalar or list.

    Args:
        val: Input operand value.

    Returns:
        int | float | bool | str | list[float]: JSON-compatible value.
    """
    raw_list = val.tolist() if isinstance(val, ArrayLikeProtocol) else val
    return raw_list if isinstance(raw_list, (int, float, bool, str, list)) else 0.0


def _write_checkpoint_to_disk(
    file_prefix: str,
    state_dict: dict[str, TensorOperand],
) -> None:
    """Persist checkpoint JSON and TensorFlow `checkpoint` tracker file if directory exists.

    Args:
        file_prefix: Checkpoint prefix path.
        state_dict: Flattened variable state dictionary.
    """
    dir_name = os.path.dirname(file_prefix)
    if not dir_name or not os.path.isdir(dir_name):
        return
    serialized: dict[str, dict[str, list[int] | int | float | bool | str | list[float]]] = {}
    for k, v in state_dict.items():
        shape_list = list(v.shape) if isinstance(v, ArrayLikeProtocol) else []
        serialized[k] = {
            "shape": shape_list,
            "value": _to_json_serializable(v),
        }
    ckpt_file = f"{file_prefix}.ckpt.json"
    with open(ckpt_file, "w", encoding="utf-8") as f:
        json.dump(serialized, f)

    all_paths = _DIRECTORY_ALL.get(dir_name, [file_prefix])
    tracker_path = os.path.join(dir_name, "checkpoint")
    lines = [f'model_checkpoint_path: "{file_prefix}"']
    for p in all_paths:
        lines.append(f'all_model_checkpoint_paths: "{p}"')
    nl = chr(10)
    with open(tracker_path, "w", encoding="utf-8") as f:
        f.write(nl.join(lines) + nl)


def _load_checkpoint_from_disk(
    file_prefix: str,
) -> dict[str, TensorOperand] | None:
    """Load a checkpoint state dictionary from `<file_prefix>.ckpt.json` if present on disk.

    Args:
        file_prefix: Checkpoint prefix path.

    Returns:
        dict[str, TensorOperand] | None: Loaded state dictionary or `None`.
    """
    ckpt_file = f"{file_prefix}.ckpt.json"
    if not os.path.isfile(ckpt_file):
        return None
    with open(ckpt_file, encoding="utf-8") as f:
        raw_data = json.load(f)
    return {
        str(k): as_operand(
            entry["value"] if isinstance(entry, Mapping) and "value" in entry else entry
        )
        for k, entry in raw_data.items()
    }


class Status:
    """Status object returned by `Checkpoint.restore` to verify restoration state."""

    def assert_consumed(self) -> Status:
        """Assert that all objects in the checkpoint have been matched and consumed.

        Returns:
            Status: Self reference for method chaining.
        """
        return self

    def assert_existing_objects_matched(self) -> Status:
        """Assert that all existing Python objects in the `Checkpoint` were matched.

        Returns:
            Status: Self reference for method chaining.
        """
        return self

    def run_restore_ops(self) -> None:
        """Execute restore operations in graph mode (no-op in eager mode)."""
        return None


class _CheckpointState:
    """Protocol buffer-like container holding the latest and all checkpoint paths."""

    def __init__(
        self,
        model_checkpoint_path: str = "",
        all_model_checkpoint_paths: Sequence[str] | None = None,
    ) -> None:
        """Initialize `_CheckpointState`.

        Args:
            model_checkpoint_path: Path prefix to the latest checkpoint.
            all_model_checkpoint_paths: Sequence of all known checkpoint path prefixes.
        """
        self.model_checkpoint_path = model_checkpoint_path
        self.all_model_checkpoint_paths: list[str] = (
            list(all_model_checkpoint_paths)
            if all_model_checkpoint_paths is not None
            else ([model_checkpoint_path] if model_checkpoint_path else [])
        )


def _extract_trackable_state(
    tracked: dict[str, TrackableLike | TensorOperand],
) -> dict[str, TensorOperand]:
    """Extract a flat dictionary of variable values from tracked objects.

    Args:
        tracked: Dictionary mapping attribute names to tracked objects.

    Returns:
        dict[str, TensorOperand]: Flattened state dictionary.
    """
    state_dict: dict[str, TensorOperand] = {}
    for k, v in tracked.items():
        if isinstance(v, TrackableContainerProtocol):
            for i, var in enumerate(v.variables):
                if isinstance(var, NumpyConvertibleProtocol):
                    state_dict[f"{k}/{i}"] = as_operand(var.numpy())
        elif isinstance(v, NumpyConvertibleProtocol):
            state_dict[k] = as_operand(v.numpy())
    return state_dict


class Checkpoint:
    """Groups trackable objects, saving and restoring their state."""

    def __init__(self, **kwargs: TrackableLike | TensorOperand) -> None:
        """Initialize a `Checkpoint` tracking the given keyword objects.

        Args:
            **kwargs: Trackable objects to include in the checkpoint.
        """
        self._kwargs: dict[str, TrackableLike | TensorOperand] = dict(kwargs)

    def save(self, file_prefix: str, **kwargs: TensorOperand) -> str:
        """Save a training checkpoint to `file_prefix`.

        Args:
            file_prefix: Prefix path for the saved checkpoint file.
            **kwargs: Optional save options.

        Returns:
            str: The saved checkpoint `file_prefix`.
        """
        del kwargs
        state_dict = _extract_trackable_state(self._kwargs)
        _CHECKPOINT_STORE[file_prefix] = state_dict
        dir_name = os.path.dirname(file_prefix)
        if dir_name:
            _DIRECTORY_LATEST[dir_name] = file_prefix
            history = _DIRECTORY_ALL.setdefault(dir_name, [])
            if file_prefix not in history:
                history.append(file_prefix)
        _write_checkpoint_to_disk(file_prefix, state_dict)
        return file_prefix

    def restore(
        self,
        save_path: str | None,
        options: CheckpointOptions | TensorOperand = None,
    ) -> Status | None:
        """Restore a training checkpoint from `save_path`.

        Args:
            save_path: Path prefix to restore from, or None.
            options: Optional `CheckpointOptions` configuration.

        Returns:
            Status | None: `Status` object if `save_path` is found, else `None`.
        """
        del options
        if save_path is None:
            return None
        state_dict = _CHECKPOINT_STORE.get(save_path)
        if state_dict is None:
            state_dict = _load_checkpoint_from_disk(save_path)
        if state_dict is None:
            return None

        for k, v in self._kwargs.items():
            if isinstance(v, TrackableContainerProtocol):
                for i, var in enumerate(v.variables):
                    key = f"{k}/{i}"
                    if key in state_dict:
                        var.assign(state_dict[key])
            elif isinstance(v, AssignableProtocol) and k in state_dict:
                v.assign(state_dict[k])

        return Status()


class CheckpointManager:
    """Manages multiple checkpoints by keeping some and deleting unneeded ones."""

    def __init__(
        self,
        checkpoint: Checkpoint | None,
        directory: str | None,
        max_to_keep: int | None,
        keep_checkpoint_every_n_hours: int | None = None,
        checkpoint_name: str = "ckpt",
        step_counter: TensorOperand = None,
        checkpoint_interval: int | None = None,
        init_fn: CallableOperandProtocol | None = None,
    ) -> None:
        """Initialize a `CheckpointManager`.

        Args:
            checkpoint: The `tf.train.Checkpoint` instance to manage.
            directory: Path to the directory in which to write checkpoints.
            max_to_keep: Maximum number of recent checkpoints to keep.
            keep_checkpoint_every_n_hours: Optional interval in hours to preserve checkpoints.
            checkpoint_name: Base name for checkpoint files.
            step_counter: Optional `tf.Variable` used for counting steps.
            checkpoint_interval: Optional step interval between checkpoints.
            init_fn: Optional callable executed if no checkpoints exist in `directory`.
        """
        del keep_checkpoint_every_n_hours, step_counter, checkpoint_interval, init_fn
        self._checkpoint = checkpoint
        self._directory = directory or ""
        self._max_to_keep = max_to_keep
        self._checkpoint_name = checkpoint_name
        self._step: int = 0
        self._checkpoints: list[str] = []
        self._kwargs: dict[str, TrackableLike | TensorOperand] = (
            dict(checkpoint._kwargs) if isinstance(checkpoint, Checkpoint) else {}
        )

    def save(
        self,
        checkpoint_number: int | str | None = None,
        check_interval: bool = True,
        options: CheckpointOptions | TensorOperand = None,
        **kwargs: TensorOperand,
    ) -> str:
        """Create a new managed checkpoint and prune older checkpoints if needed.

        Args:
            checkpoint_number: Optional step number or explicit prefix string.
            check_interval: Whether to respect `checkpoint_interval`.
            options: Optional `CheckpointOptions` instance.
            **kwargs: Additional save keyword options.

        Returns:
            str: Path prefix to the saved checkpoint.
        """
        del check_interval, options, kwargs
        self._step += 1
        if isinstance(checkpoint_number, str):
            prefix_str = checkpoint_number
        else:
            step_num = checkpoint_number if isinstance(checkpoint_number, int) else self._step
            base = f"{self._checkpoint_name}-{step_num}"
            prefix_str = os.path.join(self._directory, base) if self._directory else base

        if prefix_str not in self._checkpoints:
            self._checkpoints.append(prefix_str)

        if self._directory:
            _DIRECTORY_LATEST[self._directory] = prefix_str
            _DIRECTORY_ALL[self._directory] = list(self._checkpoints)

        if isinstance(self._checkpoint, Checkpoint):
            self._checkpoint.save(prefix_str)
        else:
            state_dict = _extract_trackable_state(self._kwargs)
            _CHECKPOINT_STORE[prefix_str] = state_dict
            _write_checkpoint_to_disk(prefix_str, state_dict)

        if isinstance(self._max_to_keep, int) and self._max_to_keep > 0:
            while len(self._checkpoints) > self._max_to_keep:
                evicted = self._checkpoints.pop(0)
                _CHECKPOINT_STORE.pop(evicted, None)
                evicted_file = f"{evicted}.ckpt.json"
                if os.path.isfile(evicted_file):
                    os.remove(evicted_file)
            _DIRECTORY_ALL[self._directory] = list(self._checkpoints)
            _write_checkpoint_to_disk(
                prefix_str,
                _CHECKPOINT_STORE.get(prefix_str, {}),
            )

        return prefix_str

    @property
    def latest_checkpoint(self) -> str | None:
        """Return the prefix of the most recent checkpoint in `directory`.

        Returns:
            str | None: The prefix of the most recent checkpoint, or `None`.
        """
        return self._checkpoints[-1] if self._checkpoints else None

    @property
    def checkpoints(self) -> list[str]:
        """Return a list of managed checkpoint prefixes.

        Returns:
            list[str]: Managed checkpoint paths.
        """
        return list(self._checkpoints)


class BytesList:
    """Protocol buffer container for a list of byte strings."""

    def __init__(
        self,
        value: Sequence[bytes | str] | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `BytesList`.

        Args:
            value: Optional sequence of byte strings.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            self.value: list[bytes] = [
                v if isinstance(v, bytes) else str(v).encode("utf-8") for v in value
            ]
        else:
            self.value = []
        self.args = (value, *args)
        self.kwargs = kwargs


class FloatList:
    """Protocol buffer container for a list of 32-bit floating-point values."""

    def __init__(
        self,
        value: Sequence[float | int] | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `FloatList`.

        Args:
            value: Optional sequence of floats.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            self.value: list[float] = [float(v) for v in value if isinstance(v, (int, float))]
        else:
            self.value = []
        self.args = (value, *args)
        self.kwargs = kwargs


class Int64List:
    """Protocol buffer container for a list of 64-bit integer values."""

    def __init__(
        self,
        value: Sequence[int] | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `Int64List`.

        Args:
            value: Optional sequence of integers.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            self.value: list[int] = [int(v) for v in value if isinstance(v, (int, float))]
        else:
            self.value = []
        self.args = (value, *args)
        self.kwargs = kwargs


class Feature:
    """Protocol buffer message representing a single feature value."""

    def __init__(
        self,
        bytes_list: BytesList | None = None,
        float_list: FloatList | None = None,
        int64_list: Int64List | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `Feature`.

        Args:
            bytes_list: Optional `BytesList` value.
            float_list: Optional `FloatList` value.
            int64_list: Optional `Int64List` value.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.bytes_list = bytes_list if isinstance(bytes_list, BytesList) else BytesList()
        self.float_list = float_list if isinstance(float_list, FloatList) else FloatList()
        self.int64_list = int64_list if isinstance(int64_list, Int64List) else Int64List()
        self.args = args
        self.kwargs = kwargs


class Features:
    """Protocol buffer message mapping feature names to `Feature` instances."""

    def __init__(
        self,
        feature: Mapping[str, Feature] | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `Features`.

        Args:
            feature: Mapping from feature key strings to `Feature` objects.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.feature: dict[str, Feature] = dict(feature) if isinstance(feature, Mapping) else {}
        self.args = args
        self.kwargs = kwargs


class FeatureList:
    """Protocol buffer message representing a sequence of `Feature` messages."""

    def __init__(
        self,
        feature: Sequence[Feature] | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `FeatureList`.

        Args:
            feature: Sequence of `Feature` instances.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.feature: list[Feature] = list(feature) if feature is not None else []
        self.args = args
        self.kwargs = kwargs


class FeatureLists:
    """Protocol buffer message mapping feature names to `FeatureList` instances."""

    def __init__(
        self,
        feature_list: Mapping[str, FeatureList] | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `FeatureLists`.

        Args:
            feature_list: Mapping from feature names to `FeatureList` instances.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.feature_list: dict[str, FeatureList] = (
            dict(feature_list) if isinstance(feature_list, Mapping) else {}
        )
        self.args = args
        self.kwargs = kwargs


class Example:
    """Protocol buffer message representing a single training or inference example."""

    def __init__(
        self,
        features: Features | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `Example`.

        Args:
            features: Optional `Features` container.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.features = features if isinstance(features, Features) else Features()
        self.args = args
        self.kwargs = kwargs

    def SerializeToString(self) -> bytes:
        """Serialize the `Example` to JSON-encoded bytes.

        Returns:
            bytes: Serialized `Example` bytes.
        """
        payload: dict[str, dict[str, list[str] | list[float] | list[int]]] = {}
        for k, feat in self.features.feature.items():
            payload[k] = {
                "bytes_list": [b.decode("latin1") for b in feat.bytes_list.value],
                "float_list": list(feat.float_list.value),
                "int64_list": list(feat.int64_list.value),
            }
        return json.dumps(payload).encode("utf-8")


class SequenceExample:
    """Protocol buffer message representing context features and sequential feature lists."""

    def __init__(
        self,
        context: Features | None = None,
        feature_lists: FeatureLists | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `SequenceExample`.

        Args:
            context: Context `Features` instance.
            feature_lists: Sequential `FeatureLists` instance.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.context = context if isinstance(context, Features) else Features()
        self.feature_lists = (
            feature_lists if isinstance(feature_lists, FeatureLists) else FeatureLists()
        )
        self.args = args
        self.kwargs = kwargs


class CheckpointOptions:
    """Options for constructing a `Checkpoint` save or restore operation."""

    def __init__(
        self,
        experimental_io_device: str | None = None,
        experimental_enable_async_checkpoint: bool = False,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `CheckpointOptions`.

        Args:
            experimental_io_device: Optional device string for checkpoint I/O.
            experimental_enable_async_checkpoint: Whether async checkpointing is enabled.
            *args: Additional positional option arguments.
            **kwargs: Keyword option arguments.
        """
        self.experimental_io_device = experimental_io_device
        self.experimental_enable_async_checkpoint = experimental_enable_async_checkpoint
        self.args = args
        self.kwargs = kwargs


class CheckpointView:
    """View for inspecting trackable objects serialized in a checkpoint."""

    def __init__(
        self,
        save_path: str | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `CheckpointView`.

        Args:
            save_path: Checkpoint prefix path.
            *args: Positional checkpoint view arguments.
            **kwargs: Keyword checkpoint view arguments.
        """
        self.save_path = save_path if isinstance(save_path, str) else ""
        self.args = args
        self.kwargs = kwargs

    def children(self, node_id: int = 0) -> dict[str, int]:
        """Return child node mappings for `node_id` in the checkpoint.

        Args:
            node_id: Graph node identifier.

        Returns:
            dict[str, int]: Mapping from attribute names to child node IDs.
        """
        del node_id
        state = _resolve_checkpoint_dict(self.save_path)
        if state is None:
            return {}
        return {name: idx + 1 for idx, name in enumerate(sorted(state))}


class JobDef:
    """Protocol buffer message defining a single job in a `ClusterDef`."""

    def __init__(
        self,
        name: str = "",
        tasks: Mapping[int, str] | Sequence[str] | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `JobDef`.

        Args:
            name: Job name (e.g., `'worker'`, `'ps'`).
            tasks: Mapping or sequence of task network addresses.
            *args: Positional job definition arguments.
            **kwargs: Keyword job definition arguments.
        """
        self.name = name
        if isinstance(tasks, Mapping):
            self.tasks: dict[int, str] = {int(k): str(v) for k, v in tasks.items()}
        elif isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes)):
            self.tasks = {i: str(v) for i, v in enumerate(tasks)}
        else:
            self.tasks = {}
        self.args = args
        self.kwargs = kwargs


class ClusterDef:
    """Protocol buffer message defining a distributed TensorFlow cluster."""

    def __init__(
        self,
        job: Sequence[JobDef] | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `ClusterDef`.

        Args:
            job: Sequence of `JobDef` instances.
            *args: Positional cluster definition arguments.
            **kwargs: Keyword cluster definition arguments.
        """
        self.job: list[JobDef] = list(job) if job is not None else []
        self.args = args
        self.kwargs = kwargs


class ClusterSpec:
    """Represents a cluster as a set of tasks organized into jobs."""

    def __init__(
        self,
        cluster: Mapping[str, Sequence[str] | Mapping[int, str]]
        | ClusterSpec
        | ClusterDef
        | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `ClusterSpec`.

        Args:
            cluster: Dictionary mapping job names to task address lists, `ClusterSpec`, or `ClusterDef`.
            *args: Additional positional arguments.
            **kwargs: Keyword cluster specification arguments.
        """
        self._cluster_dict: dict[str, list[str]] = {}
        if isinstance(cluster, ClusterSpec):
            self._cluster_dict = cluster.as_dict()
        elif isinstance(cluster, ClusterDef):
            for j in cluster.job:
                self._cluster_dict[j.name] = [j.tasks[k] for k in sorted(j.tasks)]
        elif isinstance(cluster, Mapping):
            for job_name, tasks in cluster.items():
                if isinstance(tasks, Mapping):
                    self._cluster_dict[str(job_name)] = [str(tasks[k]) for k in sorted(tasks)]
                else:
                    self._cluster_dict[str(job_name)] = [str(t) for t in tasks]
        self.args = args
        self.kwargs = kwargs

    def as_dict(self) -> dict[str, list[str]]:
        """Return a dictionary from job names to lists of task network addresses.

        Returns:
            dict[str, list[str]]: Copy of the cluster specification dictionary.
        """
        return {k: list(v) for k, v in self._cluster_dict.items()}

    @property
    def jobs(self) -> list[str]:
        """Return a list of job names in the cluster.

        Returns:
            list[str]: Sorted job names.
        """
        return sorted(self._cluster_dict)

    def num_tasks(self, job_name: str) -> int:
        """Return the number of tasks defined in `job_name`.

        Args:
            job_name: Name of the job.

        Returns:
            int: Number of tasks in `job_name`.
        """
        return len(self._cluster_dict.get(job_name, []))

    def task_address(self, job_name: str, task_index: int) -> str:
        """Return the network address of `task_index` in `job_name`.

        Args:
            job_name: Name of the job.
            task_index: Zero-based task index.

        Returns:
            str: Task network address string.
        """
        return self._cluster_dict[job_name][task_index]


class Coordinator:
    """Coordinator for threads in a training loop."""

    def __init__(
        self,
        clean_stop_exception_types: Sequence[type[BaseException]] | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `Coordinator`.

        Args:
            clean_stop_exception_types: Exception types treated as clean thread exits.
            *args: Positional coordinator arguments.
            **kwargs: Keyword coordinator arguments.
        """
        self.clean_stop_exception_types = (
            tuple(clean_stop_exception_types)
            if clean_stop_exception_types is not None
            else (StopIteration,)
        )
        self._stopped: bool = False
        self.args = args
        self.kwargs = kwargs

    def request_stop(self, ex: BaseException | None = None) -> None:
        """Request that the coordinated threads stop.

        Args:
            ex: Optional exception that triggered the stop.
        """
        del ex
        self._stopped = True

    def should_stop(self) -> bool:
        """Check whether a stop has been requested.

        Returns:
            bool: `True` if `request_stop` was called, else `False`.
        """
        return self._stopped

    def clear_stop(self) -> None:
        """Clear the stop flag so the coordinator can be reused."""
        self._stopped = False

    def join(
        self,
        threads: Sequence[TensorOperand] | None = None,
        stop_grace_period_secs: float = 120.0,
        ignore_live_threads: bool = False,
    ) -> None:
        """Wait for coordinated threads to terminate.

        Args:
            threads: Sequence of thread-like objects.
            stop_grace_period_secs: Grace period in seconds after `request_stop()`.
            ignore_live_threads: Whether to ignore threads still alive after timeout.
        """
        del threads, stop_grace_period_secs, ignore_live_threads
        return None


class ServerDef:
    """Protocol buffer message defining the configuration of a distributed server."""

    def __init__(
        self,
        cluster: ClusterDef | None = None,
        job_name: str = "",
        task_index: int = 0,
        protocol: str = "grpc",
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `ServerDef`.

        Args:
            cluster: `ClusterDef` describing the cluster membership.
            job_name: Job name to which this server belongs.
            task_index: Task index within `job_name`.
            protocol: Communication protocol (e.g., `'grpc'`).
            *args: Positional server definition arguments.
            **kwargs: Keyword server definition arguments.
        """
        self.cluster = cluster if isinstance(cluster, ClusterDef) else ClusterDef()
        self.job_name = job_name
        self.task_index = task_index
        self.protocol = protocol
        self.args = args
        self.kwargs = kwargs


class TrackableView:
    """View for inspecting the object graph of a trackable root object."""

    def __init__(
        self,
        root: TrackableLike | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `TrackableView`.

        Args:
            root: Root trackable object.
            *args: Positional trackable root arguments.
            **kwargs: Keyword arguments.
        """
        self.root = root
        self.args = args
        self.kwargs = kwargs

    def children(
        self,
        obj: TrackableLike | TensorOperand = None,
    ) -> dict[str, TensorOperand]:
        """Return a dictionary of child references for `obj` (defaults to `self.root`).

        Args:
            obj: Trackable object to inspect.

        Returns:
            dict[str, TensorOperand]: Child name to object mapping.
        """
        target = self.root if obj is None else obj
        if isinstance(target, TrackableContainerProtocol):
            return {
                f"var_{i}": as_operand(v.numpy())
                if isinstance(v, NumpyConvertibleProtocol)
                else None
                for i, v in enumerate(target.variables)
            }
        return {}


class ExponentialMovingAverage:
    """Maintains moving averages of variables by employing an exponential decay."""

    def __init__(
        self,
        decay: float = 0.999,
        num_updates: int | None = None,
        zero_debias: bool = False,
        name: str = "ExponentialMovingAverage",
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `ExponentialMovingAverage`.

        Args:
            decay: Decay rate float in `[0.0, 1.0]`.
            num_updates: Optional step count used to dynamically bound decay.
            zero_debias: Whether to apply zero-debiasing (`1 / (1 - decay^step)`) to moving averages.
            name: Optional name prefix for the moving average operations.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.decay = decay
        self.num_updates = num_updates
        self.zero_debias = zero_debias
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self._averages: dict[int, TensorOperand] = {}
        self._biased_accum: dict[int, float] = {}
        self._steps: dict[int, int] = {}

    def apply(self, var_list: Sequence[TensorOperand] | None = None) -> None:
        """Maintain moving averages of the variables in `var_list`.

        Args:
            var_list: Sequence of `Variable` or `Tensor` objects to track.
        """
        if var_list is None:
            return
        eff_decay = self.decay
        if self.num_updates is not None:
            eff_decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for v in var_list:
            vid = id(v)
            raw_val: TensorOperand = (
                as_operand(v.numpy()) if isinstance(v, NumpyConvertibleProtocol) else v
            )
            curr_val: TensorOperand = (
                raw_val.item()
                if isinstance(raw_val, ArrayLikeProtocol) and raw_val.size == 1
                else raw_val
            )
            if self.zero_debias and isinstance(curr_val, (int, float)):
                step = self._steps.get(vid, 0) + 1
                self._steps[vid] = step
                prev_biased = self._biased_accum.get(vid, 0.0)
                biased = eff_decay * prev_biased + (1.0 - eff_decay) * float(curr_val)
                self._biased_accum[vid] = biased
                denom = 1.0 - (eff_decay**step)
                self._averages[vid] = biased / denom if denom > 0.0 else float(curr_val)
            elif vid not in self._averages:
                self._averages[vid] = curr_val
            else:
                prev_val = self._averages[vid]
                if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                    self._averages[vid] = eff_decay * float(prev_val) + (1.0 - eff_decay) * float(
                        curr_val
                    )
                else:
                    self._averages[vid] = curr_val

    def average(self, var: TensorOperand) -> TensorOperand:
        """Return the `Variable` or value holding the average of `var`.

        Args:
            var: A variable previously passed to `apply()`.

        Returns:
            TensorOperand: Moving average value if tracked, else `None`.
        """
        return self._averages.get(id(var))

    def average_name(self, var: TensorOperand) -> str:
        """Return the name of the `Variable` holding the average for `var`.

        Args:
            var: A variable to compute the average name for.

        Returns:
            str: Shadow variable name string.
        """
        del var
        return f"Variable/{self.name}"


class experimental:
    """Experimental training and checkpoint sharding APIs."""

    class MaxShardSizePolicy:
        """Policy that splits tensors into shards capped by a maximum byte size."""

        def __init__(
            self,
            max_shard_size: int = 0,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `MaxShardSizePolicy`.

            Args:
                max_shard_size: Maximum byte size per shard.
                *args: Positional shard size arguments.
                **kwargs: Keyword arguments.
            """
            self.max_shard_size = max_shard_size
            self.args = args
            self.kwargs = kwargs

    class PythonState:
        """Mixin for trackable Python objects with custom serialization state."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `PythonState`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self._state_str: str = ""
            self.args = args
            self.kwargs = kwargs

        def serialize(self) -> str:
            """Serialize the custom Python state to a string.

            Returns:
                str: Serialized state string.
            """
            return self._state_str

        def deserialize(self, string_value: str) -> None:
            """Restore the custom Python state from `string_value`.

            Args:
                string_value: Previously serialized state string.
            """
            self._state_str = string_value

    class ShardByTaskPolicy:
        """Policy that shards checkpoint variables across cluster tasks."""

        def __init__(
            self,
            num_shards: int = 1,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `ShardByTaskPolicy`.

            Args:
                num_shards: Number of task shards.
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.num_shards = num_shards
            self.args = args
            self.kwargs = kwargs

    class ShardableTensor:
        """Specification of a tensor slice eligible for checkpoint sharding."""

        def __init__(
            self,
            tensor: TensorOperand = None,
            shape: Sequence[int] | None = None,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `ShardableTensor`.

            Args:
                tensor: Underlying tensor value.
                shape: Optional tensor shape tuple.
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.tensor = tensor
            self.shape: tuple[int, ...] = tuple(shape) if shape is not None else ()
            self.args = args
            self.kwargs = kwargs

    class ShardingCallback:
        """Base class for custom checkpoint sharding callbacks."""

        def __init__(
            self,
            description: str = "",
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `ShardingCallback`.

            Args:
                description: Human-readable callback description.
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.description = description
            self.args = args
            self.kwargs = kwargs


def _read_tracker_file(tracker_path: str) -> tuple[str, list[str]]:
    """Parse `model_checkpoint_path` and `all_model_checkpoint_paths` from `tracker_path`.

    Args:
        tracker_path: Path to the TensorFlow `'checkpoint'` state file.

    Returns:
        tuple[str, list[str]]: `(latest_path, all_paths)`.
    """
    latest = ""
    all_paths: list[str] = []
    with open(tracker_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split('"')
            if line.startswith("model_checkpoint_path:"):
                latest = parts[1]
            else:
                all_paths.append(parts[1])
    return (latest, all_paths)


def latest_checkpoint(
    checkpoint_dir: str | TensorOperand = None,
    latest_filename: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> str | None:
    """Find the filename of the latest saved checkpoint file in a directory.

    Args:
        checkpoint_dir: Directory where checkpoints were saved.
        latest_filename: Optional name of the checkpoint tracker file.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        str | None: Latest checkpoint prefix in `checkpoint_dir` if found, else `None`.
    """
    del args, kwargs
    if isinstance(checkpoint_dir, str) and checkpoint_dir:
        if checkpoint_dir in _DIRECTORY_LATEST:
            return _DIRECTORY_LATEST[checkpoint_dir]
        tracker_path = os.path.join(checkpoint_dir, latest_filename or "checkpoint")
        if os.path.isfile(tracker_path):
            latest_disk, _ = _read_tracker_file(tracker_path)
            return latest_disk
        matches = [
            k
            for k in _CHECKPOINT_STORE
            if k == checkpoint_dir or k.startswith(checkpoint_dir.rstrip("/") + "/")
        ]
        if matches:
            return matches[-1]
    return None


def get_checkpoint_state(
    checkpoint_dir: str | TensorOperand = None,
    latest_filename: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> _CheckpointState | None:
    """Return `_CheckpointState` from the `'checkpoint'` file or active store in `checkpoint_dir`.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        latest_filename: Optional tracker filename (defaults to `'checkpoint'`).
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        _CheckpointState | None: Populated `_CheckpointState` if found, else `None`.
    """
    del args, kwargs
    if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
        return None
    tracker_path = os.path.join(checkpoint_dir, latest_filename or "checkpoint")
    if os.path.isfile(tracker_path):
        latest_disk, all_paths = _read_tracker_file(tracker_path)
        return _CheckpointState(latest_disk, all_paths)

    latest_mem = latest_checkpoint(checkpoint_dir)
    if latest_mem is not None:
        all_mem = _DIRECTORY_ALL.get(checkpoint_dir, [latest_mem])
        return _CheckpointState(latest_mem, all_mem)
    return None


def checkpoints_iterator(
    checkpoint_dir: str | TensorOperand = None,
    min_interval_secs: float = 0.0,
    timeout: float | None = None,
    timeout_fn: CallableOperandProtocol | None = None,
) -> list[str]:
    """Return an iterable list of checkpoint prefixes found in `checkpoint_dir`.

    Args:
        checkpoint_dir: Directory in which checkpoints are saved.
        min_interval_secs: Minimum interval between polling checks.
        timeout: Maximum wait time in seconds.
        timeout_fn: Optional callable invoked on timeout.

    Returns:
        list[str]: Ordered list of checkpoint path prefixes in `checkpoint_dir`.
    """
    del min_interval_secs, timeout, timeout_fn
    state = get_checkpoint_state(checkpoint_dir)
    if state is not None:
        return list(state.all_model_checkpoint_paths)
    return []


def _resolve_checkpoint_dict(
    ckpt_dir_or_file: str | TensorOperand,
) -> dict[str, TensorOperand] | None:
    """Resolve the stored checkpoint dictionary for `ckpt_dir_or_file`.

    Args:
        ckpt_dir_or_file: Checkpoint prefix or directory path.

    Returns:
        dict[str, TensorOperand] | None: Stored variable dictionary or `None`.
    """
    if not isinstance(ckpt_dir_or_file, str) or not ckpt_dir_or_file:
        return None
    if ckpt_dir_or_file in _CHECKPOINT_STORE:
        return _CHECKPOINT_STORE[ckpt_dir_or_file]
    disk_direct = _load_checkpoint_from_disk(ckpt_dir_or_file)
    if disk_direct is not None:
        return disk_direct
    latest = latest_checkpoint(ckpt_dir_or_file)
    if latest is not None:
        if latest in _CHECKPOINT_STORE:
            return _CHECKPOINT_STORE[latest]
        return _load_checkpoint_from_disk(latest)
    return None


def list_variables(
    ckpt_dir_or_file: str | TensorOperand = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> list[tuple[str, list[int]]]:
    """Return list of `(name, shape)` tuples for all variables in a checkpoint.

    Args:
        ckpt_dir_or_file: Directory or checkpoint prefix path.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        list[tuple[str, list[int]]]: Sorted list of `(variable_name, shape_list)` pairs.
    """
    del args, kwargs
    state = _resolve_checkpoint_dict(ckpt_dir_or_file)
    if state is None:
        return []
    result: list[tuple[str, list[int]]] = []
    for name in sorted(state):
        val = state[name]
        if isinstance(val, ArrayLikeProtocol):
            shape_list = list(val.shape)
        elif isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            shape_list = [len(val)]
        else:
            shape_list = []
        result.append((name, shape_list))
    return result


def load_checkpoint(
    ckpt_dir_or_file: str | TensorOperand = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> dict[str, TensorOperand] | None:
    """Return a checkpoint dictionary for checkpoints found at the given path.

    Args:
        ckpt_dir_or_file: Directory or checkpoint prefix path.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        dict[str, TensorOperand] | None: Checkpoint state dictionary if found, else `None`.
    """
    del args, kwargs
    state = _resolve_checkpoint_dict(ckpt_dir_or_file)
    return dict(state) if state is not None else None


def load_variable(
    ckpt_dir_or_file: str | TensorOperand = None,
    name: str | TensorOperand = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> TensorOperand:
    """Return the tensor value of the given variable in a checkpoint.

    Args:
        ckpt_dir_or_file: Directory or checkpoint prefix path.
        name: Name of the variable to load.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        TensorOperand: Variable value if found, else `None`.
    """
    del args, kwargs
    state = _resolve_checkpoint_dict(ckpt_dir_or_file)
    if state is not None and isinstance(name, str) and name in state:
        return state[name]
    return None
