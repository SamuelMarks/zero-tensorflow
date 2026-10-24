"""Unit tests for `test_train`."""

from __future__ import annotations

from zero_tensorflow import Variable, train


def test_checkpoint() -> None:
    """Verify `Checkpoint.save` and `Checkpoint.restore` round-trip variable state."""
    v = Variable(5.0)
    ckpt = train.Checkpoint(var1=v)
    assert ckpt._kwargs["var1"] is v
    prefix = ckpt.save("ckpt_dir/prefix")
    assert prefix == "ckpt_dir/prefix"
    v.assign(42.0)
    assert v.numpy() == 42.0
    status = ckpt.restore("ckpt_dir/prefix")
    assert status is not None
    assert status.assert_consumed() is status
    assert status.assert_existing_objects_matched() is status
    status.run_restore_ops()
    assert v.numpy() == 5.0


def test_checkpoint_manager() -> None:
    """Verify `CheckpointManager` tracks saved checkpoints and enforces `max_to_keep`."""
    v = Variable(1.0)
    ckpt = train.Checkpoint(v=v)
    mgr = train.CheckpointManager(ckpt, "dir", max_to_keep=2)
    assert mgr._checkpoint is ckpt
    assert mgr._directory == "dir"
    assert mgr._max_to_keep == 2
    assert mgr.latest_checkpoint is None
    assert mgr.checkpoints == []

    first = mgr.save()
    assert first == "dir/ckpt-1"
    assert mgr.latest_checkpoint == "dir/ckpt-1"
    assert mgr.checkpoints == ["dir/ckpt-1"]

    second = mgr.save(checkpoint_number=10)
    assert second == "dir/ckpt-10"
    assert mgr.latest_checkpoint == "dir/ckpt-10"
    assert mgr.checkpoints == ["dir/ckpt-1", "dir/ckpt-10"]
