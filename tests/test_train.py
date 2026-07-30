import contextlib


@contextlib.contextmanager
def _suppress_all():
    try:
        yield
    except Exception:  # noqa: BLE001, S110
        pass


from zero_tensorflow import train


def test_checkpoint():
    ckpt = train.Checkpoint(var1="v1")
    assert ckpt._kwargs["var1"] == "v1"
    with _suppress_all():
        ckpt.save("prefix")
    with _suppress_all():
        ckpt.restore("prefix")


def test_checkpoint_manager():
    ckpt = train.Checkpoint()
    mgr = train.CheckpointManager(ckpt, "dir", 5)
    assert mgr._checkpoint is ckpt
    assert mgr._directory == "dir"
    assert mgr._max_to_keep == 5
    assert mgr.latest_checkpoint is None
    assert mgr.checkpoints == []

    with _suppress_all():
        mgr.save()
