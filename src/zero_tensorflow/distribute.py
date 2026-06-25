"""TensorFlow distribute module."""

from typing import Any, Callable, Optional, Sequence
import threading

__all__ = [
    "Strategy",
    "MirroredStrategy",
    "MultiWorkerMirroredStrategy",
    "OneDeviceStrategy",
    "TPUStrategy",
    "ReduceOp",
]


class ReduceOp:
    """Reduction operations."""

    SUM = "SUM"
    MEAN = "MEAN"


class Strategy:
    """Base class for distribution strategies."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the object."""
        pass

    def scope(self) -> Any:
        """
        Return a context manager that selects this strategy.

        Returns:
            Any: The context manager.
        """
        return _StrategyScope(self)

    def run(
        self,
        fn: Callable[..., Any],
        args: tuple = (),
        kwargs: Optional[dict] = None,
        options: Any = None,
    ) -> Any:
        """
        Run the computation defined by `fn` on each replica.

        Args:
            fn (Callable): The function to run.
            args (tuple): Arguments to pass to `fn`.
            kwargs (dict): Keyword arguments to pass to `fn`.
            options (Any): Options for the run.

        Returns:
            Any: The result of running `fn`.
        """
        if kwargs is None:
            kwargs = {}
        return fn(*args, **kwargs)

    def reduce(self, reduce_op: str, value: Any, axis: Optional[int] = None) -> Any:
        """
        Reduce `value` across replicas.

        Args:
            reduce_op (str): The reduction operation (e.g., 'SUM', 'MEAN').
            value (Any): The value to reduce.
            axis (Optional[int]): The axis to reduce along.

        Returns:
            Any: The reduced value.
        """
        return value


class _StrategyScope:
    """Context manager for setting the current strategy."""

    def __init__(self, strategy: Strategy):
        """Initialize the object."""
        self._strategy = strategy
        self._previous_strategy = None

    def __enter__(self) -> Strategy:
        """Enter the scope."""
        self._previous_strategy = getattr(_strategy_state, "current", None)
        _strategy_state.current = self._strategy
        return self._strategy

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the scope."""
        _strategy_state.current = self._previous_strategy


_strategy_state = threading.local()


class MirroredStrategy(Strategy):
    """Synchronous training across multiple replicas on one machine."""

    def __init__(
        self, devices: Optional[Sequence[str]] = None, cross_device_ops: Any = None
    ) -> None:
        """
        Initialize the object.

        Args:
            devices (Optional[Sequence[str]]): List of devices to use.
            cross_device_ops (Any): Cross-device ops implementation.
        """
        super().__init__()
        self._devices = (
            devices
            if devices is not None
            else ["/job:localhost/replica:0/task:0/device:GPU:0"]
        )
        self._cross_device_ops = cross_device_ops


class MultiWorkerMirroredStrategy(Strategy):
    """Synchronous training across multiple workers."""

    def __init__(
        self, cluster_resolver: Any = None, communication_options: Any = None
    ) -> None:
        """
        Initialize the object.

        Args:
            cluster_resolver (Any): The cluster resolver.
            communication_options (Any): Communication options.
        """
        super().__init__()
        self._cluster_resolver = cluster_resolver
        self._communication_options = communication_options


class OneDeviceStrategy(Strategy):
    """A distribution strategy for running on a single device."""

    def __init__(self, device: str) -> None:
        """
        Initialize the object.

        Args:
            device (str): The device string to use.
        """
        super().__init__()
        self._device = device


class TPUStrategy(Strategy):
    """Synchronous training on TPUs."""

    def __init__(self, tpu_cluster_resolver: Optional[Any] = None) -> None:
        """
        Initialize the object.

        Args:
            tpu_cluster_resolver (Optional[Any]): The TPU cluster resolver.
        """
        super().__init__()
        self._tpu_cluster_resolver = tpu_cluster_resolver


# Stubs from TODO_PLAN.md
from typing import Any


class CrossDeviceOps:
    """Stub for CrossDeviceOps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: CrossDeviceOps")


class DistributedDataset:
    """Stub for DistributedDataset."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: DistributedDataset")


class DistributedIterator:
    """Stub for DistributedIterator."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: DistributedIterator")


class DistributedValues:
    """Stub for DistributedValues."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: DistributedValues")


class HierarchicalCopyAllReduce:
    """Stub for HierarchicalCopyAllReduce."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: HierarchicalCopyAllReduce")


class InputContext:
    """Stub for InputContext."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: InputContext")


class InputOptions:
    """Stub for InputOptions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: InputOptions")


class InputReplicationMode:
    """Stub for InputReplicationMode."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: InputReplicationMode")


class NcclAllReduce:
    """Stub for NcclAllReduce."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: NcclAllReduce")


class ParameterServerStrategy:
    """Stub for ParameterServerStrategy."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: ParameterServerStrategy")


class ReductionToOneDevice:
    """Stub for ReductionToOneDevice."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: ReductionToOneDevice")


class ReplicaContext:
    """Stub for ReplicaContext."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: ReplicaContext")


class RunOptions:
    """Stub for RunOptions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: RunOptions")


class Server:
    """Stub for Server."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: Server")


class StrategyExtended:
    """Stub for StrategyExtended."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Not implemented: StrategyExtended")


class cluster_resolver:
    """Stub for cluster_resolver module."""

    class ClusterResolver:
        """Stub for ClusterResolver."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: ClusterResolver")

    class GCEClusterResolver:
        """Stub for GCEClusterResolver."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: GCEClusterResolver")

    class KubernetesClusterResolver:
        """Stub for KubernetesClusterResolver."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: KubernetesClusterResolver")

    class SimpleClusterResolver:
        """Stub for SimpleClusterResolver."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: SimpleClusterResolver")

    class SlurmClusterResolver:
        """Stub for SlurmClusterResolver."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: SlurmClusterResolver")

    class TFConfigClusterResolver:
        """Stub for TFConfigClusterResolver."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: TFConfigClusterResolver")

    class TPUClusterResolver:
        """Stub for TPUClusterResolver."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: TPUClusterResolver")

    class UnionResolver:
        """Stub for UnionResolver."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: UnionResolver")


class coordinator:
    """Stub for coordinator module."""

    class ClusterCoordinator:
        """Stub for ClusterCoordinator."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: ClusterCoordinator")

    class PerWorkerValue:
        """Stub for PerWorkerValue."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: PerWorkerValue")

    class RemoteValue:
        """Stub for RemoteValue."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: RemoteValue")

    @staticmethod
    def experimental_get_current_worker_index(*args: Any, **kwargs: Any) -> None:
        """Stub for experimental_get_current_worker_index."""
        raise NotImplementedError(
            "Not implemented: experimental_get_current_worker_index"
        )


class experimental:
    """Stub for experimental module."""

    class CentralStorageStrategy:
        """Stub for CentralStorageStrategy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: CentralStorageStrategy")

    class CollectiveCommunication:
        """Stub for CollectiveCommunication."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: CollectiveCommunication")

    class CollectiveHints:
        """Stub for CollectiveHints."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: CollectiveHints")

    class CommunicationImplementation:
        """Stub for CommunicationImplementation."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: CommunicationImplementation")

    class CommunicationOptions:
        """Stub for CommunicationOptions."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: CommunicationOptions")

    class MultiWorkerMirroredStrategy:
        """Stub for MultiWorkerMirroredStrategy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: MultiWorkerMirroredStrategy")

    class ParameterServerStrategy:
        """Stub for ParameterServerStrategy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: ParameterServerStrategy")

    class PreemptionCheckpointHandler:
        """Stub for PreemptionCheckpointHandler."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: PreemptionCheckpointHandler")

    class PreemptionWatcher:
        """Stub for PreemptionWatcher."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: PreemptionWatcher")

    class TPUStrategy:
        """Stub for TPUStrategy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: TPUStrategy")

    class TerminationConfig:
        """Stub for TerminationConfig."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: TerminationConfig")

    class ValueContext:
        """Stub for ValueContext."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("Not implemented: ValueContext")

    @staticmethod
    def coordinator(*args: Any, **kwargs: Any) -> None:
        """Stub for coordinator."""
        raise NotImplementedError("Not implemented: coordinator")

    @staticmethod
    def partitioners(*args: Any, **kwargs: Any) -> None:
        """Stub for partitioners."""
        raise NotImplementedError("Not implemented: partitioners")

    @staticmethod
    def rpc(*args: Any, **kwargs: Any) -> None:
        """Stub for rpc."""
        raise NotImplementedError("Not implemented: rpc")


def experimental_set_strategy(*args: Any, **kwargs: Any) -> None:
    """Stub for experimental_set_strategy."""
    raise NotImplementedError("Not implemented: experimental_set_strategy")


def get_replica_context(*args: Any, **kwargs: Any) -> None:
    """Stub for get_replica_context."""
    raise NotImplementedError("Not implemented: get_replica_context")


def get_strategy(*args: Any, **kwargs: Any) -> None:
    """Stub for get_strategy."""
    raise NotImplementedError("Not implemented: get_strategy")


def has_strategy(*args: Any, **kwargs: Any) -> None:
    """Stub for has_strategy."""
    raise NotImplementedError("Not implemented: has_strategy")


def in_cross_replica_context(*args: Any, **kwargs: Any) -> None:
    """Stub for in_cross_replica_context."""
    raise NotImplementedError("Not implemented: in_cross_replica_context")
