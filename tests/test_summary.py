"""Unit tests for `zero_tensorflow.summary`."""

from __future__ import annotations

from pathlib import Path

import zero_tensorflow as tf
from zero_tensorflow import summary


class _FakeArrayFloat:
    """Pure-Python 1-D `ArrayLikeProtocol` test helper."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return 1-D shape.

        Returns:
            tuple[int, ...]: `(1,)`.
        """
        return (1,)

    @property
    def size(self) -> int:
        """Return element count.

        Returns:
            int: `1`.
        """
        return 1

    def tolist(self) -> list[float]:
        """Return Python list.

        Returns:
            list[float]: `[3.0]`.
        """
        return [3.0]

    def item(self) -> float:
        """Return scalar item.

        Returns:
            float: `3.0`.
        """
        return 3.0


def test_summary(tmp_path: Path) -> None:
    """Verify `tf.summary` writers, recording contexts, and summary emitters.

    Args:
        tmp_path: Temporary directory fixture.
    """
    summary.experimental.set_step(42)
    assert summary.experimental.get_step() == 42
    with summary.experimental.summary_scope("test") as (name, val):
        assert name == "summary_scope"
        assert val is None
    assert summary.experimental.write_raw_pb(b"data") is True
    assert summary.image("test_image", tf.Tensor(1.0)) is True

    logdir = str(tmp_path / "logs")
    writer = summary.create_file_writer(logdir, filename_suffix=".test")
    with writer.as_default(step=10):
        assert summary.scalar("loss", tf.Tensor(0.25)) is True
        assert summary.histogram("weights", [1.0, 2.0, 3.0]) is True
        assert summary.text("greeting", "hello") is True
        assert summary.audio("clip", [0.1, -0.1], sample_rate=16000) is True
        assert summary.write("custom_tag", tf.Variable(5.0)) is True
        summary.flush()

    assert len(writer.events) == 5
    writer.set_as_default(step=99)
    assert summary.scalar("acc", 0.95) is True
    summary.flush(writer)
    writer.close()
    assert summary.scalar("ignored_after_close", 1.0) is False
    summary._DEFAULT_WRITER = None

    noop = summary.create_noop_writer()
    noop.init()
    with noop.as_default():
        assert summary.scalar("noop_loss", 1.0) is False
        noop.flush()

    with writer.as_default(step=None):
        pass

    with summary.record_if(False):
        assert summary.should_record_summaries() is False
        assert summary.scalar("skipped", 1.0) is False

    with summary.record_if(lambda: True):
        assert summary.should_record_summaries() is True
        assert summary.scalar("recorded_fn", 1.0) is True

    assert summary._resolve_step(tf.Tensor(12)) == 12
    summary.experimental.set_step("not_a_step")
    assert summary._resolve_step(None) == 0
    assert summary._summarize_value(_FakeArrayFloat()) == "[3.0]"
    mem_writer = summary.SummaryWriter(logdir=None)
    mem_writer.init()
    mem_writer.flush()
    writer.set_as_default(step=None)
    summary._DEFAULT_WRITER = None
