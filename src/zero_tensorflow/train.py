"""TensorFlow train module."""

from typing import Any, Optional, List

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

    def save(self, file_prefix: str, options: Optional[Any] = None) -> str:
        """
        Save a training checkpoint and provides basic checkpoint management.

        Args:
            file_prefix: A prefix to use for the checkpoint filenames.
            options: Optional tf.train.CheckpointOptions object.

        Returns:
            The full path to the checkpoint.
        """
        raise NotImplementedError("Not implemented: tf.train.Checkpoint.save")

    def restore(self, save_path: str, options: Optional[Any] = None) -> Any:
        """
        Restore a training checkpoint.

        Args:
            save_path: The path to the checkpoint to restore.
            options: Optional tf.train.CheckpointOptions object.

        Returns:
            A load status object.
        """
        raise NotImplementedError("Not implemented: tf.train.Checkpoint.restore")


class CheckpointManager:
    """Deletes old checkpoints."""

    def __init__(
        self,
        checkpoint: Checkpoint,
        directory: str,
        max_to_keep: int,
        keep_checkpoint_every_n_hours: Optional[int] = None,
        checkpoint_name: str = "ckpt",
        step_counter: Optional[Any] = None,
        checkpoint_interval: Optional[int] = None,
        init_fn: Optional[Any] = None,
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

    def save(
        self,
        checkpoint_number: Optional[int] = None,
        check_interval: bool = True,
        options: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Create a new checkpoint and manages it.

        Args:
            checkpoint_number: An optional integer, or an integer-dtype Variable or Tensor, used to number the checkpoint.
            check_interval: An optional boolean.
            options: Optional tf.train.CheckpointOptions object.

        Returns:
            The path to the new checkpoint.
        """
        raise NotImplementedError("Not implemented: tf.train.CheckpointManager.save")

    @property
    def latest_checkpoint(self) -> Optional[str]:
        """
        The prefix of the most recent checkpoint in directory.

        Returns:
            Optional[str]: The prefix of the most recent checkpoint.
        """
        return None

    @property
    def checkpoints(self) -> List[str]:
        """
        A list of managed checkpoints.

        Returns:
            List[str]: A list of managed checkpoints.
        """
        return []


# Stubs from TODO_PLAN.md
from typing import Any


class BytesList:
    """Stub for BytesList."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: BytesList")


class CheckpointOptions:
    """Stub for CheckpointOptions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: CheckpointOptions")


class CheckpointView:
    """Stub for CheckpointView."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: CheckpointView")


class ClusterDef:
    """Stub for ClusterDef."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: ClusterDef")


class ClusterSpec:
    """Stub for ClusterSpec."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: ClusterSpec")


class Coordinator:
    """Stub for Coordinator."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: Coordinator")


class Example:
    """Stub for Example."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: Example")


class ExponentialMovingAverage:
    """Stub for ExponentialMovingAverage."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: ExponentialMovingAverage")


class Feature:
    """Stub for Feature."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: Feature")


class FeatureList:
    """Stub for FeatureList."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: FeatureList")


class FeatureLists:
    """Stub for FeatureLists."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: FeatureLists")


class Features:
    """Stub for Features."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: Features")


class FloatList:
    """Stub for FloatList."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: FloatList")


class Int64List:
    """Stub for Int64List."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: Int64List")


class JobDef:
    """Stub for JobDef."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: JobDef")


class SequenceExample:
    """Stub for SequenceExample."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: SequenceExample")


class ServerDef:
    """Stub for ServerDef."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: ServerDef")


class TrackableView:
    """Stub for TrackableView."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: TrackableView")


def checkpoints_iterator(*args: Any, **kwargs: Any) -> None:
    """Stub for checkpoints_iterator."""
    raise NotImplementedError("Not implemented: checkpoints_iterator")


class experimental:
    """Stub for experimental module."""

    class MaxShardSizePolicy:
        """Stub for MaxShardSizePolicy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: MaxShardSizePolicy")

    class PythonState:
        """Stub for PythonState."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: PythonState")

    class ShardByTaskPolicy:
        """Stub for ShardByTaskPolicy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: ShardByTaskPolicy")

    class ShardableTensor:
        """Stub for ShardableTensor."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: ShardableTensor")

    class ShardingCallback:
        """Stub for ShardingCallback."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: ShardingCallback")


def get_checkpoint_state(*args: Any, **kwargs: Any) -> None:
    """Stub for get_checkpoint_state."""
    raise NotImplementedError("Not implemented: get_checkpoint_state")


def latest_checkpoint(*args: Any, **kwargs: Any) -> None:
    """Stub for latest_checkpoint."""
    raise NotImplementedError("Not implemented: latest_checkpoint")


def list_variables(*args: Any, **kwargs: Any) -> None:
    """Stub for list_variables."""
    raise NotImplementedError("Not implemented: list_variables")


def load_checkpoint(*args: Any, **kwargs: Any) -> None:
    """Stub for load_checkpoint."""
    raise NotImplementedError("Not implemented: load_checkpoint")


def load_variable(*args: Any, **kwargs: Any) -> None:
    """Stub for load_variable."""
    raise NotImplementedError("Not implemented: load_variable")
