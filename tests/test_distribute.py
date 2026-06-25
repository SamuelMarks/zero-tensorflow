from zero_tensorflow import distribute


def test_reduce_op():
    assert distribute.ReduceOp.SUM == "SUM"
    assert distribute.ReduceOp.MEAN == "MEAN"


def test_strategy_base():
    strat = distribute.Strategy()

    # scope
    with strat.scope() as s:
        assert s is strat
        assert distribute._strategy_state.current is strat

    assert distribute._strategy_state.current is None

    # run
    def dummy_fn(x, y=0):
        return x + y

    assert strat.run(dummy_fn, args=(5,), kwargs={"y": 3}) == 8
    assert strat.run(dummy_fn, args=(5,)) == 5

    # reduce
    assert strat.reduce(distribute.ReduceOp.SUM, 10) == 10


def test_mirrored_strategy():
    strat = distribute.MirroredStrategy()
    assert strat._devices == ["/job:localhost/replica:0/task:0/device:GPU:0"]
    assert strat._cross_device_ops is None

    strat2 = distribute.MirroredStrategy(
        devices=["/cpu:0", "/cpu:1"], cross_device_ops="ops"
    )
    assert strat2._devices == ["/cpu:0", "/cpu:1"]
    assert strat2._cross_device_ops == "ops"


def test_multi_worker_mirrored_strategy():
    strat = distribute.MultiWorkerMirroredStrategy(
        cluster_resolver="resolver", communication_options="options"
    )
    assert strat._cluster_resolver == "resolver"
    assert strat._communication_options == "options"


def test_one_device_strategy():
    strat = distribute.OneDeviceStrategy("/gpu:0")
    assert strat._device == "/gpu:0"


def test_tpu_strategy():
    strat = distribute.TPUStrategy(tpu_cluster_resolver="tpu_resolver")
    assert strat._tpu_cluster_resolver == "tpu_resolver"
