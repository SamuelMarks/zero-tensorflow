"""TensorFlow `distribute` strategy and cluster coordination module."""

from __future__ import annotations

import threading
from collections.abc import Callable as Callable
from collections.abc import Mapping, Sequence
from types import TracebackType

from zero_tensorflow._typing import CallableOperandProtocol, TensorOperand

from zero_keras.distribution import DataParallel as DataParallel
from zero_keras.distribution import ModelParallel as ModelParallel

__all__ = [
    "CrossDeviceOps",
    "DistributedDataset",
    "DistributedIterator",
    "DistributedValues",
    "HierarchicalCopyAllReduce",
    "InputContext",
    "InputOptions",
    "InputReplicationMode",
    "MirroredStrategy",
    "MultiWorkerMirroredStrategy",
    "NcclAllReduce",
    "OneDeviceStrategy",
    "ParameterServerStrategy",
    "ReduceOp",
    "ReductionToOneDevice",
    "ReplicaContext",
    "RunOptions",
    "Server",
    "Strategy",
    "StrategyExtended",
    "TPUStrategy",
    "cluster_resolver",
    "coordinator",
    "experimental",
    "experimental_set_strategy",
    "get_replica_context",
    "get_strategy",
    "has_strategy",
    "in_cross_replica_context",
]


class ReduceOp:
    """Enumeration of cross-replica reduction operations."""

    SUM: str = "SUM"
    MEAN: str = "MEAN"


class _StrategyLocalState(threading.local):
    """Thread-local container tracking the currently active `Strategy` and `ReplicaContext`."""

    def __init__(self) -> None:
        """Initialize the thread-local state with `current = None` and `replica_context = None`."""
        super().__init__()
        self.current: Strategy | None = None
        self.replica_context: ReplicaContext | None = None


_strategy_state = _StrategyLocalState()


class Strategy:
    """Base class for synchronous and asynchronous distribution strategies."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize the `Strategy` instance.

        Args:
            *args: Positional strategy arguments.
            **kwargs: Keyword strategy arguments.
        """
        self.args = args
        self.kwargs = kwargs

    def scope(self) -> _StrategyScope:
        """Return a context manager selecting this strategy as current.

        Returns:
            _StrategyScope: Context manager bound to `self`.
        """
        return _StrategyScope(self)

    def run(
        self,
        fn: CallableOperandProtocol,
        args: tuple[TensorOperand, ...] = (),
        kwargs: Mapping[str, TensorOperand] | None = None,
        options: RunOptions | TensorOperand = None,
    ) -> TensorOperand:
        """Run the computation defined by `fn` on each replica.

        Args:
            fn: The function to run on each replica.
            args: Optional positional tuple of arguments passed to `fn`.
            kwargs: Optional keyword dictionary of arguments passed to `fn`.
            options: Optional `RunOptions` controlling replica execution.

        Returns:
            TensorOperand: Result of executing `fn(*args, **kwargs)`.
        """
        del options
        resolved_kwargs: Mapping[str, TensorOperand] = kwargs if kwargs is not None else {}
        prev_replica_ctx = _strategy_state.replica_context
        _strategy_state.replica_context = ReplicaContext(self, replica_id_in_sync_group=0)
        try:
            return fn(*args, **resolved_kwargs)
        finally:
            _strategy_state.replica_context = prev_replica_ctx

    def reduce(
        self,
        reduce_op: str,
        value: TensorOperand,
        axis: int | Sequence[int] | None = None,
    ) -> TensorOperand:
        """Reduce `value` across replicas along `axis`.

        Args:
            reduce_op: Reduction operation identifier (`'SUM'` or `'MEAN'`).
            value: Per-replica value or tensor to reduce.
            axis: Optional dimension(s) along which to reduce.

        Returns:
            TensorOperand: Reduced tensor or value.
        """
        del axis
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) > 0
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        ):
            total: float = sum(float(v) for v in value if isinstance(v, (int, float)))
            if reduce_op.upper() == ReduceOp.MEAN:
                return total / len(value)
            return total
        return value


class _StrategyScope:
    """Context manager for activating and restoring a distribution `Strategy`."""

    def __init__(self, strategy: Strategy) -> None:
        """Initialize `_StrategyScope` for `strategy`.

        Args:
            strategy: The `Strategy` to activate on entry.
        """
        self._strategy = strategy
        self._previous_strategy: Strategy | None = None

    def __enter__(self) -> Strategy:
        """Enter the strategy scope and set `_strategy_state.current`.

        Returns:
            Strategy: Active strategy instance.
        """
        self._previous_strategy = _strategy_state.current
        _strategy_state.current = self._strategy
        return self._strategy

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the strategy scope and restore the previous strategy.

        Args:
            exc_type: Exception type if raised, else None.
            exc_val: Exception instance if raised, else None.
            exc_tb: Traceback if raised, else None.
        """
        del exc_type, exc_val, exc_tb
        _strategy_state.current = self._previous_strategy


class MirroredStrategy(Strategy):
    """Synchronous training across multiple replicas on one machine."""

    def __init__(
        self,
        devices: Sequence[str] | None = None,
        cross_device_ops: CrossDeviceOps | TensorOperand = None,
    ) -> None:
        """Initialize `MirroredStrategy`.

        Args:
            devices: Optional sequence of device strings to replicate across.
            cross_device_ops: Optional cross-device reduction implementation.
        """
        super().__init__()
        self._devices: Sequence[str] = (
            devices if devices is not None else ["/job:localhost/replica:0/task:0/device:GPU:0"]
        )
        self._cross_device_ops = cross_device_ops


class MultiWorkerMirroredStrategy(Strategy):
    """Synchronous training across multiple workers and multiple devices."""

    def __init__(
        self,
        cluster_resolver: TensorOperand = None,
        communication_options: TensorOperand = None,
    ) -> None:
        """Initialize `MultiWorkerMirroredStrategy`.

        Args:
            cluster_resolver: Optional cluster resolver instance.
            communication_options: Optional collective communication options.
        """
        super().__init__()
        self._cluster_resolver = cluster_resolver
        self._communication_options = communication_options


class OneDeviceStrategy(Strategy):
    """A distribution strategy for running on a single specified device."""

    def __init__(self, device: str | None = None) -> None:
        """Initialize `OneDeviceStrategy`.

        Args:
            device: Device string identifier (e.g., `'/gpu:0'`).
        """
        super().__init__()
        self._device = device


class TPUStrategy(Strategy):
    """Synchronous training on Tensor Processing Units (TPUs)."""

    def __init__(self, tpu_cluster_resolver: TensorOperand = None) -> None:
        """Initialize `TPUStrategy`.

        Args:
            tpu_cluster_resolver: Optional `TPUClusterResolver` instance.
        """
        super().__init__()
        self._tpu_cluster_resolver = tpu_cluster_resolver


class CrossDeviceOps:
    """Base class for cross-device reduction and broadcast algorithms."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `CrossDeviceOps`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class DistributedDataset:
    """Represents a dataset distributed across replicas."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `DistributedDataset`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class DistributedIterator:
    """An iterator over a `DistributedDataset`."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `DistributedIterator`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class DistributedValues:
    """Base class for values distributed across replicas."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `DistributedValues`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class HierarchicalCopyAllReduce:
    """Hierarchical copy all-reduce implementation of `CrossDeviceOps`."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `HierarchicalCopyAllReduce`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class InputContext:
    """Information regarding the compute replicas passed to an input-fn."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `InputContext`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class InputOptions:
    """Run options for `experimental_distribute_dataset`."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `InputOptions`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class InputReplicationMode:
    """Replication mode for input `Dataset`s or functions."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `InputReplicationMode`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class NcclAllReduce:
    """NCCL all-reduce implementation of `CrossDeviceOps`."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `NcclAllReduce`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class ParameterServerStrategy(Strategy):
    """Asynchronous multi-worker parameter server training strategy."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `ParameterServerStrategy`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        super().__init__(*args, **kwargs)


class ReductionToOneDevice:
    """CrossDeviceOps implementation copying values to a single device to reduce."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `ReductionToOneDevice`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class ReplicaContext:
    """Context class for execution within a replica block."""

    def __init__(self, *args: Strategy | TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `ReplicaContext`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class RunOptions:
    """Configuration options for `Strategy.run`."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `RunOptions`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class Server:
    """In-process TensorFlow distributed runtime server."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `Server`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class StrategyExtended:
    """Additional APIs for implementing custom distribution algorithms."""

    def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Initialize `StrategyExtended`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs


class cluster_resolver:
    """Cluster resolver implementations for distributed training environments."""

    class ClusterResolver:
        """Abstract base class for all `ClusterResolver` implementations."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `ClusterResolver`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class GCEClusterResolver:
        """ClusterResolver for Google Compute Engine (GCE)."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `GCEClusterResolver`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class KubernetesClusterResolver:
        """ClusterResolver for Kubernetes clusters."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `KubernetesClusterResolver`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class SimpleClusterResolver:
        """Simple implementation of `ClusterResolver` backed by fixed specs."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `SimpleClusterResolver`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class SlurmClusterResolver:
        """ClusterResolver for workloads launched under Slurm."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `SlurmClusterResolver`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class TFConfigClusterResolver:
        """ClusterResolver reading the `TF_CONFIG` environment variable."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `TFConfigClusterResolver`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class TPUClusterResolver:
        """ClusterResolver for Google Cloud TPUs."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `TPUClusterResolver`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class UnionResolver:
        """ClusterResolver combining multiple underlying `ClusterResolver`s."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `UnionResolver`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs


class coordinator:
    """Cluster coordinator namespace for asynchronous remote execution."""

    class ClusterCoordinator:
        """Coordinates remote function execution across a worker cluster."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `ClusterCoordinator`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class PerWorkerValue:
        """Container holding one value per worker in a cluster."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `PerWorkerValue`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class RemoteValue:
        """Handle to a value scheduled for execution on a remote worker."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `RemoteValue`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    @staticmethod
    def experimental_get_current_worker_index(
        *args: TensorOperand, **kwargs: TensorOperand
    ) -> None:
        """Return the index of the current worker when running scheduled closures.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return


class experimental:
    """Experimental distribution strategies and collective options."""

    class CentralStorageStrategy(Strategy):
        """Synchronous strategy storing variables on CPU or a single GPU."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `CentralStorageStrategy`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            super().__init__(*args, **kwargs)

    class CollectiveCommunication:
        """Communication implementation options for collective operations."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `CollectiveCommunication`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class CollectiveHints:
        """Hints for optimizing collective communication operations."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `CollectiveHints`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class CommunicationImplementation:
        """Cross-device communication backend enumeration."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `CommunicationImplementation`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class CommunicationOptions:
        """Options configuring collective communication behavior."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `CommunicationOptions`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class MultiWorkerMirroredStrategy(Strategy):
        """Experimental alias for `MultiWorkerMirroredStrategy`."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `MultiWorkerMirroredStrategy`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            super().__init__(*args, **kwargs)

    class ParameterServerStrategy(Strategy):
        """Experimental alias for `ParameterServerStrategy`."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `ParameterServerStrategy`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            super().__init__(*args, **kwargs)

    class PreemptionCheckpointHandler:
        """Preemption-aware checkpoint manager for multi-worker training."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `PreemptionCheckpointHandler`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class PreemptionWatcher:
        """Monitors cluster preemption signals."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `PreemptionWatcher`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class TPUStrategy(Strategy):
        """Experimental alias for `TPUStrategy`."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `TPUStrategy`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            super().__init__(*args, **kwargs)

    class TerminationConfig:
        """Configuration for graceful worker termination."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `TerminationConfig`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class ValueContext:
        """Context passed to `experimental_distribute_values_from_function`."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `ValueContext`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    @staticmethod
    def coordinator(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Experimental coordinator namespace stub.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def partitioners(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Experimental variable partitioners namespace stub.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def rpc(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Experimental RPC namespace stub.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return


_default_strategy = Strategy()


def experimental_set_strategy(
    strategy: Strategy | TensorOperand = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> None:
    """Override the default `tf.distribute.Strategy`.

    Args:
        strategy: `Strategy` instance or `None` to clear.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    """
    del args, kwargs
    if isinstance(strategy, Strategy) or strategy is None:
        _strategy_state.current = strategy


def get_replica_context(*args: TensorOperand, **kwargs: TensorOperand) -> ReplicaContext | None:
    """Return the current `ReplicaContext` or `None` if in cross-replica context.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        ReplicaContext | None: Active `ReplicaContext` during `Strategy.run`, else `None`.
    """
    del args, kwargs
    return _strategy_state.replica_context


def get_strategy(*args: TensorOperand, **kwargs: TensorOperand) -> Strategy:
    """Return the current default `tf.distribute.Strategy` object.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Strategy: Active `Strategy` if set, else the default `Strategy`.
    """
    del args, kwargs
    return _strategy_state.current if _strategy_state.current is not None else _default_strategy


def has_strategy(*args: TensorOperand, **kwargs: TensorOperand) -> bool:
    """Return whether there is a current non-default `tf.distribute.Strategy`.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        bool: `True` if inside a `Strategy.scope()` or set via `experimental_set_strategy`.
    """
    del args, kwargs
    return _strategy_state.current is not None


def in_cross_replica_context(*args: TensorOperand, **kwargs: TensorOperand) -> bool:
    """Return `True` if in a cross-replica context.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        bool: `True` when not inside a `Strategy.run` replica context.
    """
    del args, kwargs
    return _strategy_state.replica_context is None
