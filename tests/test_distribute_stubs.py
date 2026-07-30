import zero_tensorflow.distribute as dist


def call_safely(callable_obj):
    try:
        return callable_obj()
    except TypeError:
        try:
            return callable_obj(None)
        except TypeError:
            pass
    return None


def test_distribute_stubs():
    assert call_safely(dist.Callable) is None
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
    assert call_safely(dist.Sequence) is None
    assert call_safely(dist.Server) is not None
    assert call_safely(dist.Strategy) is not None
    assert call_safely(dist.StrategyExtended) is not None
    assert call_safely(dist.TPUStrategy) is not None
    assert call_safely(dist.cluster_resolver) is not None
    assert call_safely(dist.coordinator) is not None
    assert call_safely(dist.experimental) is not None
    assert call_safely(dist.experimental_set_strategy) is None
    assert call_safely(dist.get_replica_context) is None
    assert call_safely(dist.get_strategy) is None
    assert call_safely(dist.has_strategy) is None
    assert call_safely(dist.in_cross_replica_context) is None
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
