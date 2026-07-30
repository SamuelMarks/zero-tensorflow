"""TensorFlow train module."""

from __future__ import annotations

from typing import Any

__all__ = [
    "Checkpoint",
    "CheckpointManager",
]


class Checkpoint:
    """Groups trackable objects, saving and restoring them."""

    def __init__(self, **kwargs: Any):
        """
        Initialize the object.

        Args:
            **kwargs: Trackable objects to include in the checkpoint.
        """
        self._kwargs = kwargs

    def save(self, file_prefix: str, **kwargs: Any) -> str:
        from ml_switcheroo_compiler.state.checkpoint import save_state_dict

        # Extract state variables
        state_dict = {}
        for k, v in getattr(self, "_kwargs", {}).items():
            if hasattr(v, "variables"):
                for i, var in enumerate(v.variables):
                    state_dict[f"{k}/{i}"] = var.numpy()
            elif hasattr(v, "numpy"):
                state_dict[k] = v.numpy()

        save_state_dict(state_dict, file_prefix + ".safetensors")
        return file_prefix

    def restore(self, save_path: str, options: Any | None = None) -> Any:
        from ml_switcheroo_compiler.state.checkpoint import load_state_dict

        try:
            state_dict = load_state_dict(save_path + ".safetensors")
        except Exception:  # noqa: BLE001
            return None

        for k, v in getattr(self, "_kwargs", {}).items():
            if hasattr(v, "variables"):
                for i, var in enumerate(v.variables):
                    key = f"{k}/{i}"
                    if key in state_dict:
                        var.assign(state_dict[key])
            elif hasattr(v, "assign") and k in state_dict:
                v.assign(state_dict[k])

        class Status:
            def assert_consumed(self):
                return self

            def assert_existing_objects_matched(self):
                return self

            def run_restore_ops(self):
                pass

        return Status()


class CheckpointManager:
    """Deletes old checkpoints."""

    def __init__(
        self,
        checkpoint: Checkpoint,
        directory: str,
        max_to_keep: int,
        keep_checkpoint_every_n_hours: int | None = None,
        checkpoint_name: str = "ckpt",
        step_counter: Any | None = None,
        checkpoint_interval: int | None = None,
        init_fn: Any | None = None,
    ):
        """
        Initialize the object.

        Args:
            checkpoint: The tf.train.Checkpoint instance to save and manage checkpoints for.
            directory: The path to a directory in which to write checkpoints.
            max_to_keep: An integer, the number of checkpoints to keep.
            keep_checkpoint_every_n_hours: (Optional) An integer, the number of hours between each checkpoint to be saved.
            checkpoint_name: (Optional) Custom name for the checkpoint file.
            step_counter: (Optional) A tf.Variable instance for checking the current step.
            checkpoint_interval: (Optional) An integer, the number of steps between each checkpoint to be saved.
            init_fn: (Optional) Callable. A function to do customized initialization if no checkpoints are in the directory.
        """
        self._checkpoint = checkpoint
        self._directory = directory
        self._max_to_keep = max_to_keep
        self._checkpoint_name = checkpoint_name

    def save(self, file_prefix: str, **kwargs: Any) -> str:
        from ml_switcheroo_compiler.state.checkpoint import save_state_dict

        # Extract state variables
        state_dict = {}
        for k, v in getattr(self, "_kwargs", {}).items():
            if hasattr(v, "variables"):
                for i, var in enumerate(v.variables):
                    state_dict[f"{k}/{i}"] = var.numpy()
            elif hasattr(v, "numpy"):
                state_dict[k] = v.numpy()

        save_state_dict(state_dict, file_prefix + ".safetensors")
        return file_prefix

    @property
    def latest_checkpoint(self) -> str | None:
        """
        The prefix of the most recent checkpoint in directory.

        Returns:
            Optional[str]: The prefix of the most recent checkpoint.
        """
        return None

    @property
    def checkpoints(self) -> list[str]:
        """
        A list of managed checkpoints.

        Returns:
            List[str]: A list of managed checkpoints.
        """
        return []


# Stubs from TODO_PLAN.md


class BytesList:
    """Stub for BytesList."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class CheckpointOptions:
    """Stub for CheckpointOptions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class CheckpointView:
    """Stub for CheckpointView."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class ClusterDef:
    """Stub for ClusterDef."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class ClusterSpec:
    """Stub for ClusterSpec."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class Coordinator:
    """Stub for Coordinator."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class Example:
    """Stub for Example."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class ExponentialMovingAverage:
    """Stub for ExponentialMovingAverage."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class Feature:
    """Stub for Feature."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class FeatureList:
    """Stub for FeatureList."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class FeatureLists:
    """Stub for FeatureLists."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class Features:
    """Stub for Features."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class FloatList:
    """Stub for FloatList."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class Int64List:
    """Stub for Int64List."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class JobDef:
    """Stub for JobDef."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class SequenceExample:
    """Stub for SequenceExample."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class ServerDef:
    """Stub for ServerDef."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class TrackableView:
    """Stub for TrackableView."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def checkpoints_iterator(*args: Any, **kwargs: Any) -> None:
    """Stub for checkpoints_iterator."""
    return


class experimental:
    """Stub for experimental module."""

    class MaxShardSizePolicy:
        """Stub for MaxShardSizePolicy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class PythonState:
        """Stub for PythonState."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class ShardByTaskPolicy:
        """Stub for ShardByTaskPolicy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class ShardableTensor:
        """Stub for ShardableTensor."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class ShardingCallback:
        """Stub for ShardingCallback."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


def get_checkpoint_state(*args: Any, **kwargs: Any) -> None:
    """Stub for get_checkpoint_state."""
    return


def latest_checkpoint(*args: Any, **kwargs: Any) -> None:
    """Stub for latest_checkpoint."""
    return


def list_variables(*args: Any, **kwargs: Any) -> None:
    """Stub for list_variables."""
    return


def load_checkpoint(*args: Any, **kwargs: Any) -> None:
    """Stub for load_checkpoint."""
    return


def load_variable(*args: Any, **kwargs: Any) -> None:
    """Stub for load_variable."""
    return
