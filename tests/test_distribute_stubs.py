"""Unit tests for `test_distribute_stubs`."""

from __future__ import annotations

import zero_tensorflow.distribute as dist
from zero_tensorflow._typing import CallableOperandProtocol, TensorOperand


def call_safely(callable_obj: CallableOperandProtocol | TensorOperand | type) -> TensorOperand:
    """Invoke `callable_obj` safely with zero or one `None` argument.

    Args:
        callable_obj: Target class or function.

    Returns:
        TensorOperand: Result of calling `callable_obj` or `None`.
    """
    if not callable(callable_obj):
        return None
    try:
        res: TensorOperand = callable_obj()
        return res
    except TypeError:
        try:
            res2: TensorOperand = callable_obj(None)
            return res2
        except TypeError:
            return None


def test_distribute_stubs() -> None:
    """Verify `tf.distribute` strategies, resolvers, coordinators, and context functions."""
    assert call_safely(dist.CrossDeviceOps) is not None
    assert call_safely(dist.DistributedDataset) is not None
    assert call_safely(dist.DistributedIterator) is not None
    assert call_safely(dist.DistributedValues) is not None
    assert call_safely(dist.HierarchicalCopyAllReduce) is not None
    assert call_safely(dist.InputContext) is not None
    assert call_safely(dist.InputOptions) is not None
    assert call_safely(dist.InputReplicationMode) is not None
    assert call_safely(dist.MirroredStrategy) is not None
    assert call_safely(dist.MultiWorkerMirroredStrategy) is not None
    assert call_safely(dist.NcclAllReduce) is not None
    assert call_safely(dist.OneDeviceStrategy) is not None
    assert call_safely(dist.ParameterServerStrategy) is not None
    assert call_safely(dist.ReduceOp) is not None
    assert call_safely(dist.ReductionToOneDevice) is not None
    assert call_safely(dist.ReplicaContext) is not None
    assert call_safely(dist.RunOptions) is not None
    assert call_safely(dist.Server) is not None
    assert call_safely(dist.Strategy) is not None
    assert call_safely(dist.StrategyExtended) is not None
    assert call_safely(dist.TPUStrategy) is not None
    assert call_safely(dist.cluster_resolver) is not None
    assert call_safely(dist.coordinator) is not None
    assert call_safely(dist.experimental) is not None
    dist.experimental_set_strategy(None)
    assert dist.get_replica_context() is None
    assert isinstance(dist.get_strategy(), dist.Strategy)
    assert dist.has_strategy() is False
    assert dist.in_cross_replica_context() is True
    assert call_safely(dist.experimental.CentralStorageStrategy) is not None
    assert call_safely(dist.experimental.CollectiveCommunication) is not None
    assert call_safely(dist.experimental.CollectiveHints) is not None
    assert call_safely(dist.experimental.CommunicationImplementation) is not None
    assert call_safely(dist.experimental.CommunicationOptions) is not None
    assert call_safely(dist.experimental.MultiWorkerMirroredStrategy) is not None
    assert call_safely(dist.experimental.ParameterServerStrategy) is not None
    assert call_safely(dist.experimental.PreemptionCheckpointHandler) is not None
    assert call_safely(dist.experimental.PreemptionWatcher) is not None
    assert call_safely(dist.experimental.TPUStrategy) is not None
    assert call_safely(dist.experimental.TerminationConfig) is not None
    assert call_safely(dist.experimental.ValueContext) is not None
    assert call_safely(dist.experimental.coordinator) is None
    assert call_safely(dist.experimental.partitioners) is None
    assert call_safely(dist.experimental.rpc) is None
    assert call_safely(dist.cluster_resolver.ClusterResolver) is not None
    assert call_safely(dist.cluster_resolver.GCEClusterResolver) is not None
    assert call_safely(dist.cluster_resolver.KubernetesClusterResolver) is not None
    assert call_safely(dist.cluster_resolver.SimpleClusterResolver) is not None
    assert call_safely(dist.cluster_resolver.SlurmClusterResolver) is not None
    assert call_safely(dist.cluster_resolver.TFConfigClusterResolver) is not None
    assert call_safely(dist.cluster_resolver.TPUClusterResolver) is not None
    assert call_safely(dist.cluster_resolver.UnionResolver) is not None
    assert call_safely(dist.coordinator.ClusterCoordinator) is not None
    assert call_safely(dist.coordinator.PerWorkerValue) is not None
    assert call_safely(dist.coordinator.RemoteValue) is not None
    assert call_safely(dist.coordinator.experimental_get_current_worker_index) is None
