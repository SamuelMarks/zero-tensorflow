"""Behavioral unit tests for `zero_tensorflow.distribute` strategies, resolvers, and coordinators."""

from __future__ import annotations

import zero_tensorflow.distribute as dist
from zero_tensorflow._typing import TensorOperand


def _triple_fn(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
    """Multiply the first numeric positional argument by `3`.

    Args:
        *args: Positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        TensorOperand: Tripled value or `0`.
    """
    del kwargs
    if args and isinstance(args[0], (int, float)):
        return args[0] * 3
    return 0


def test_distribute_stubs() -> None:
    """Verify `tf.distribute` strategies, resolvers, coordinators, and context functions."""
    assert dist.ReduceOp.SUM == "SUM"
    assert dist.ReduceOp.MEAN == "MEAN"

    in_rep = dist.InputReplicationMode("PER_WORKER")
    assert in_rep.args == ("PER_WORKER",)
    reduce_op_inst = dist.ReduceOp()
    assert isinstance(reduce_op_inst, dist.ReduceOp)

    cross_ops = dist.CrossDeviceOps(num_packs=2)
    assert cross_ops.kwargs == {"num_packs": 2}
    hc_ops = dist.HierarchicalCopyAllReduce(num_packs=4)
    assert hc_ops.kwargs == {"num_packs": 4}
    nccl_ops = dist.NcclAllReduce(num_packs=1)
    assert nccl_ops.kwargs == {"num_packs": 1}
    r_one = dist.ReductionToOneDevice(reduce_to_device="/device:CPU:0")
    assert r_one.kwargs == {"reduce_to_device": "/device:CPU:0"}

    in_ctx = dist.InputContext(
        num_input_pipelines=4,
        input_pipeline_id=1,
        num_replicas_in_sync=8,
    )
    assert in_ctx.kwargs["num_input_pipelines"] == 4
    assert in_ctx.kwargs["input_pipeline_id"] == 1
    assert in_ctx.kwargs["num_replicas_in_sync"] == 8

    in_opts = dist.InputOptions(experimental_fetch_to_device=True)
    assert in_opts.kwargs == {"experimental_fetch_to_device": True}
    run_opts = dist.RunOptions(experimental_enable_dynamic_batch_size=True)
    assert run_opts.kwargs == {"experimental_enable_dynamic_batch_size": True}

    d_vals = dist.DistributedValues(10, 20)
    assert d_vals.args == (10, 20)
    d_ds = dist.DistributedDataset([1, 2, 3])
    assert d_ds.args == ([1, 2, 3],)
    d_it = dist.DistributedIterator([4, 5])
    assert d_it.args == ([4, 5],)

    mirrored = dist.MirroredStrategy(devices=["/cpu:0", "/cpu:1"])
    assert isinstance(mirrored, dist.Strategy)
    assert isinstance(dist.MultiWorkerMirroredStrategy(), dist.Strategy)
    assert isinstance(dist.OneDeviceStrategy("/cpu:0"), dist.Strategy)
    assert isinstance(dist.ParameterServerStrategy(), dist.ParameterServerStrategy)
    assert isinstance(dist.TPUStrategy(), dist.Strategy)
    assert isinstance(dist.StrategyExtended("ext"), dist.StrategyExtended)
    assert isinstance(dist.Server({"worker": ["localhost:2222"]}), dist.Server)

    with mirrored.scope():
        assert dist.has_strategy() is True
        assert dist.get_strategy() is mirrored
        ran_val = mirrored.run(_triple_fn, args=(7,))
        assert ran_val == 21
        assert mirrored.reduce(dist.ReduceOp.SUM, [10.0, 20.0]) == 30.0
        assert mirrored.reduce(dist.ReduceOp.MEAN, [10.0, 20.0]) == 15.0

    dist.experimental_set_strategy(None)
    assert dist.get_replica_context() is None
    assert isinstance(dist.get_strategy(), dist.Strategy)
    assert dist.has_strategy() is False
    assert dist.in_cross_replica_context() is True

    rep_ctx = dist.ReplicaContext(replica_id_in_sync_group=0)
    assert rep_ctx.kwargs == {"replica_id_in_sync_group": 0}

    assert isinstance(
        dist.experimental.CentralStorageStrategy(),
        dist.experimental.CentralStorageStrategy,
    )
    assert isinstance(
        dist.experimental.CollectiveCommunication("RING"),
        dist.experimental.CollectiveCommunication,
    )
    assert isinstance(
        dist.experimental.CommunicationImplementation("NCCL"),
        dist.experimental.CommunicationImplementation,
    )
    assert isinstance(
        dist.experimental.CollectiveHints(bytes_per_pack=1024),
        dist.experimental.CollectiveHints,
    )
    assert isinstance(
        dist.experimental.CommunicationOptions(implementation="NCCL"),
        dist.experimental.CommunicationOptions,
    )
    assert isinstance(
        dist.experimental.MultiWorkerMirroredStrategy(),
        dist.experimental.MultiWorkerMirroredStrategy,
    )
    assert isinstance(
        dist.experimental.ParameterServerStrategy(),
        dist.experimental.ParameterServerStrategy,
    )
    assert isinstance(
        dist.experimental.PreemptionCheckpointHandler(),
        dist.experimental.PreemptionCheckpointHandler,
    )
    assert isinstance(dist.experimental.PreemptionWatcher(), dist.experimental.PreemptionWatcher)
    assert isinstance(dist.experimental.TPUStrategy(), dist.experimental.TPUStrategy)
    assert isinstance(dist.experimental.TerminationConfig(), dist.experimental.TerminationConfig)
    val_ctx = dist.experimental.ValueContext(replica_id_in_sync_group=2, num_replicas_in_sync=4)
    assert val_ctx.kwargs["replica_id_in_sync_group"] == 2
    assert val_ctx.kwargs["num_replicas_in_sync"] == 4
    assert dist.experimental.coordinator() is None
    assert dist.experimental.partitioners() is None
    assert dist.experimental.rpc() is None

    for resolver_cls in (
        dist.cluster_resolver.ClusterResolver,
        dist.cluster_resolver.GCEClusterResolver,
        dist.cluster_resolver.KubernetesClusterResolver,
        dist.cluster_resolver.SimpleClusterResolver,
        dist.cluster_resolver.SlurmClusterResolver,
        dist.cluster_resolver.TFConfigClusterResolver,
        dist.cluster_resolver.TPUClusterResolver,
        dist.cluster_resolver.UnionResolver,
    ):
        inst = resolver_cls("worker")
        assert inst.args == ("worker",)

    assert isinstance(dist.cluster_resolver(), dist.cluster_resolver)
    assert isinstance(dist.coordinator(), dist.coordinator)
    assert isinstance(dist.experimental(), dist.experimental)

    coord = dist.coordinator.ClusterCoordinator("strategy")
    assert coord.args == ("strategy",)
    rem = dist.coordinator.RemoteValue(42)
    assert rem.args == (42,)
    pw = dist.coordinator.PerWorkerValue(42)
    assert pw.args == (42,)
    assert dist.coordinator.experimental_get_current_worker_index() is None
