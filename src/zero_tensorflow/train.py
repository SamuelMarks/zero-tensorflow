"""TensorFlow `train` checkpointing and training utilities module."""

from __future__ import annotations

import os
from collections.abc import Sequence

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
        _CHECKPOINT_STORE[file_prefix] = _extract_trackable_state(self._kwargs)
        dir_name = os.path.dirname(file_prefix)
        if dir_name:
            _DIRECTORY_LATEST[dir_name] = file_prefix
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
        if save_path is None or save_path not in _CHECKPOINT_STORE:
            return None

        state_dict = _CHECKPOINT_STORE[save_path]
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

        if isinstance(self._checkpoint, Checkpoint):
            self._checkpoint.save(prefix_str)
        else:
            _CHECKPOINT_STORE[prefix_str] = _extract_trackable_state(self._kwargs)

        if self._directory:
            _DIRECTORY_LATEST[self._directory] = prefix_str

        if prefix_str not in self._checkpoints:
            self._checkpoints.append(prefix_str)

        if isinstance(self._max_to_keep, int) and self._max_to_keep > 0:
            while len(self._checkpoints) > self._max_to_keep:
                evicted = self._checkpoints.pop(0)
                _CHECKPOINT_STORE.pop(evicted, None)

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

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `BytesList`.

        Args:
            *args: Positional byte list arguments.
            **kwargs: Keyword byte list arguments.
        """
        self.args = args
        self.kwargs = kwargs


class CheckpointOptions:
    """Options for constructing a `Checkpoint` save or restore operation."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `CheckpointOptions`.

        Args:
            *args: Positional option arguments.
            **kwargs: Keyword option arguments.
        """
        self.args = args
        self.kwargs = kwargs


class CheckpointView:
    """View for inspecting trackable objects serialized in a checkpoint."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `CheckpointView`.

        Args:
            *args: Positional checkpoint view arguments.
            **kwargs: Keyword checkpoint view arguments.
        """
        self.args = args
        self.kwargs = kwargs


class ClusterDef:
    """Protocol buffer message defining a distributed TensorFlow cluster."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `ClusterDef`.

        Args:
            *args: Positional cluster definition arguments.
            **kwargs: Keyword cluster definition arguments.
        """
        self.args = args
        self.kwargs = kwargs


class ClusterSpec:
    """Represents a cluster as a set of tasks organized into jobs."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `ClusterSpec`.

        Args:
            *args: Positional cluster specification arguments.
            **kwargs: Keyword cluster specification arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Coordinator:
    """Coordinator for threads in a training loop."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `Coordinator`.

        Args:
            *args: Positional coordinator arguments.
            **kwargs: Keyword coordinator arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Example:
    """Protocol buffer message representing a single training or inference example."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `Example`.

        Args:
            *args: Positional example arguments.
            **kwargs: Keyword example arguments.
        """
        self.args = args
        self.kwargs = kwargs


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
            zero_debias: Whether to apply zero-debiasing to moving averages.
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
            if vid not in self._averages:
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


class Feature:
    """Protocol buffer message representing a single feature value."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `Feature`.

        Args:
            *args: Positional feature arguments.
            **kwargs: Keyword feature arguments.
        """
        self.args = args
        self.kwargs = kwargs


class FeatureList:
    """Protocol buffer message representing a sequence of `Feature` messages."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `FeatureList`.

        Args:
            *args: Positional feature list arguments.
            **kwargs: Keyword feature list arguments.
        """
        self.args = args
        self.kwargs = kwargs


class FeatureLists:
    """Protocol buffer message mapping feature names to `FeatureList` instances."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `FeatureLists`.

        Args:
            *args: Positional feature lists arguments.
            **kwargs: Keyword feature lists arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Features:
    """Protocol buffer message mapping feature names to `Feature` instances."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `Features`.

        Args:
            *args: Positional features map arguments.
            **kwargs: Keyword features map arguments.
        """
        self.args = args
        self.kwargs = kwargs


class FloatList:
    """Protocol buffer container for a list of 32-bit floating-point values."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `FloatList`.

        Args:
            *args: Positional float list arguments.
            **kwargs: Keyword float list arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Int64List:
    """Protocol buffer container for a list of 64-bit integer values."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `Int64List`.

        Args:
            *args: Positional int64 list arguments.
            **kwargs: Keyword int64 list arguments.
        """
        self.args = args
        self.kwargs = kwargs


class JobDef:
    """Protocol buffer message defining a single job in a `ClusterDef`."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `JobDef`.

        Args:
            *args: Positional job definition arguments.
            **kwargs: Keyword job definition arguments.
        """
        self.args = args
        self.kwargs = kwargs


class SequenceExample:
    """Protocol buffer message representing context features and sequential feature lists."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `SequenceExample`.

        Args:
            *args: Positional sequence example arguments.
            **kwargs: Keyword sequence example arguments.
        """
        self.args = args
        self.kwargs = kwargs


class ServerDef:
    """Protocol buffer message defining the configuration of a distributed server."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `ServerDef`.

        Args:
            *args: Positional server definition arguments.
            **kwargs: Keyword server definition arguments.
        """
        self.args = args
        self.kwargs = kwargs


class TrackableView:
    """View for inspecting the object graph of a trackable root object."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `TrackableView`.

        Args:
            *args: Positional trackable root arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


def checkpoints_iterator(*args: TensorOperand, **kwargs: TensorOperand) -> None:
    """Continuously yield new checkpoint files as they appear in a directory.

    Args:
        *args: Positional directory arguments.
        **kwargs: Keyword iterator arguments.
    """
    del args, kwargs
    return


class experimental:
    """Experimental training and checkpoint sharding APIs."""

    class MaxShardSizePolicy:
        """Policy that splits tensors into shards capped by a maximum byte size."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `MaxShardSizePolicy`.

            Args:
                *args: Positional shard size arguments.
                **kwargs: Keyword arguments.
            """
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
            self.args = args
            self.kwargs = kwargs

    class ShardByTaskPolicy:
        """Policy that shards checkpoint variables across cluster tasks."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `ShardByTaskPolicy`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class ShardableTensor:
        """Specification of a tensor slice eligible for checkpoint sharding."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `ShardableTensor`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class ShardingCallback:
        """Base class for custom checkpoint sharding callbacks."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `ShardingCallback`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs


def get_checkpoint_state(*args: TensorOperand, **kwargs: TensorOperand) -> None:
    """Return `CheckpointState` proto from the `'checkpoint'` file in a directory.

    Args:
        *args: Positional directory arguments.
        **kwargs: Keyword arguments.
    """
    del args, kwargs
    return


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
    del latest_filename, args, kwargs
    if isinstance(checkpoint_dir, str):
        if checkpoint_dir in _DIRECTORY_LATEST:
            return _DIRECTORY_LATEST[checkpoint_dir]
        matches = [
            k
            for k in _CHECKPOINT_STORE
            if k == checkpoint_dir or k.startswith(checkpoint_dir.rstrip("/") + "/")
        ]
        if matches:
            return matches[-1]
    return None


def _resolve_checkpoint_dict(
    ckpt_dir_or_file: str | TensorOperand,
) -> dict[str, TensorOperand] | None:
    """Resolve the stored checkpoint dictionary for `ckpt_dir_or_file`.

    Args:
        ckpt_dir_or_file: Checkpoint prefix or directory path.

    Returns:
        dict[str, TensorOperand] | None: Stored variable dictionary or `None`.
    """
    if not isinstance(ckpt_dir_or_file, str):
        return None
    if ckpt_dir_or_file in _CHECKPOINT_STORE:
        return _CHECKPOINT_STORE[ckpt_dir_or_file]
    latest = latest_checkpoint(ckpt_dir_or_file)
    if latest is not None and latest in _CHECKPOINT_STORE:
        return _CHECKPOINT_STORE[latest]
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
