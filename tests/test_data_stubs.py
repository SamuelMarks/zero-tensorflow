"""Unit tests for `test_data_stubs`."""

from __future__ import annotations

import gzip
import sqlite3
import zlib
from collections.abc import Iterator
from pathlib import Path

import zero_tensorflow as tf
import zero_tensorflow.data as tf_data

import pytest


def test_data_specs_and_iterators() -> None:
    """Verify `DatasetSpec`, `IteratorSpec`, `Iterator.get_next`, `from_tensors`, and `range`."""
    ds = tf_data.Dataset.range(1, 5, 2)
    assert list(ds) == [1, 3]
    assert list(tf_data.Dataset.range()) == []

    single_ds = tf_data.Dataset.from_tensors([10, 20])
    assert list(single_ds) == [[10, 20]]

    ds_spec = tf_data.DatasetSpec.from_value(ds)
    assert ds_spec.element_spec is None
    assert ds_spec.value_type is tf_data.Dataset
    assert ds_spec.is_compatible_with(ds) is True
    assert ds_spec.is_compatible_with(tf_data.DatasetSpec(element_spec="int")) is True
    assert (
        tf_data.DatasetSpec(element_spec="int").is_compatible_with(
            tf_data.DatasetSpec(element_spec="float")
        )
        is False
    )
    assert ds_spec.is_compatible_with("not_a_spec") is False

    it = tf_data.Iterator(ds)
    assert it.get_next() == 1
    it_spec = tf_data.IteratorSpec.from_value(it)
    assert it_spec.element_spec is None
    assert it_spec.value_type is tf_data.Iterator
    assert it_spec.is_compatible_with(it) is True
    assert it_spec.is_compatible_with(tf_data.IteratorSpec(element_spec="int")) is True
    assert (
        tf_data.IteratorSpec(element_spec="int").is_compatible_with(
            tf_data.IteratorSpec(element_spec="float")
        )
        is False
    )
    assert it_spec.is_compatible_with("not_a_spec") is False


def test_file_datasets_and_compression(tmp_path: Path) -> None:
    """Verify `TextLineDataset`, `FixedLengthRecordDataset`, `TFRecordDataset`, and `SqlDataset`.

    Args:
        tmp_path: Temporary directory fixture.
    """
    nl = chr(10)
    gz_txt = tmp_path / "lines.txt.gz"
    gz_txt.write_bytes(gzip.compress(f"line1{nl}line2{nl}".encode()))
    assert list(
        tf_data.TextLineDataset(
            tf_data.Dataset.from_tensor_slices([str(gz_txt)]), compression_type="GZIP"
        )
    ) == ["line1", "line2"]

    zlib_txt = tmp_path / "lines.txt.zlib"
    zlib_txt.write_bytes(zlib.compress(f"z1{nl}z2{nl}".encode()))
    assert list(tf_data.TextLineDataset(str(zlib_txt), compression_type="ZLIB")) == ["z1", "z2"]

    gz_bin = tmp_path / "rec.bin.gz"
    gz_bin.write_bytes(gzip.compress(b"AABBCC"))
    assert list(
        tf_data.FixedLengthRecordDataset(str(gz_bin), record_bytes=2, compression_type="GZIP")
    ) == [b"AA", b"BB", b"CC"]

    tfr_gz = tmp_path / "records.tfrecord.gz"
    exp_writer = tf_data.experimental.TFRecordWriter(str(tfr_gz), compression_type="GZIP")
    exp_writer.write(tf_data.Dataset.from_tensor_slices([b"r1", "r2"]))
    assert list(tf_data.TFRecordDataset(str(tfr_gz), compression_type="GZIP")) == [b"r1", b"r2"]

    tfr_zlib = tmp_path / "records.tfrecord.zlib"
    exp_writer_z = tf_data.experimental.TFRecordWriter(str(tfr_zlib), compression_type="ZLIB")
    exp_writer_z.write(tf_data.Dataset.from_tensor_slices([b"zr1"]))
    assert list(tf_data.TFRecordDataset(str(tfr_zlib), compression_type="ZLIB")) == [b"zr1"]

    tfr_raw = tmp_path / "records.tfrecord"
    tf_data.experimental.TFRecordWriter(str(tfr_raw)).write(
        tf_data.Dataset.from_tensor_slices([b"raw1"])
    )
    assert list(tf_data.TFRecordDataset(str(tfr_raw))) == [b"raw1"]
    tf_data.experimental.TFRecordWriter("").write(tf_data.Dataset.from_tensor_slices([b"noop"]))

    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO items VALUES (1, 'one'), (2, 'two')")
    conn.commit()
    conn.close()

    sql_ds = tf_data.experimental.SqlDataset(
        "sqlite", str(db_path), "SELECT id, name FROM items ORDER BY id", ("int32", "string")
    )
    assert list(sql_ds) == [(1, "one"), (2, "two")]


def test_experimental_classes_and_options(tmp_path: Path) -> None:
    """Verify `tf.data.experimental` classes, options, `Counter`, `CsvDataset`, and `Optional`.

    Args:
        tmp_path: Temporary directory fixture.
    """
    assert (
        tf_data.experimental.AutoShardPolicy(tf_data.experimental.AutoShardPolicy.FILE).value == 1
    )
    assert tf_data.experimental.AutotuneAlgorithm(2).value == 2
    auto_opt = tf_data.experimental.AutotuneOptions(enabled=False, cpu_budget=2, ram_budget=1024)
    assert auto_opt.enabled is False and auto_opt.cpu_budget == 2
    dist_opt = tf_data.experimental.DistributeOptions(auto_shard_policy=1, num_devices=4)
    assert dist_opt.auto_shard_policy == 1 and dist_opt.num_devices == 4
    ext_pol = tf_data.experimental.ExternalStatePolicy(
        tf_data.experimental.ExternalStatePolicy.FAIL
    )
    assert ext_pol.value == 2
    opt_options = tf_data.experimental.OptimizationOptions(filter_fusion=True)
    assert opt_options.filter_fusion is True

    counter_ds = tf_data.experimental.Counter(start=5, step=3)
    assert list(counter_ds.take(4)) == [5, 8, 11, 14]
    assert len(list(counter_ds)) == 1000

    rand_ds1 = tf_data.experimental.RandomDataset(seed=123).take(5)
    rand_ds2 = tf_data.experimental.RandomDataset(seed=123).take(5)
    assert list(rand_ds1) == list(rand_ds2)
    assert len(list(rand_ds1)) == 5
    assert len(list(tf_data.experimental.RandomDataset(seed=123))) == 1000

    nl = chr(10)
    csv_path = tmp_path / "table.csv"
    csv_path.write_text(f"col_a,col_b,col_c{nl}10,,30{nl}40,NA,60{nl}", encoding="utf-8")
    csv_ds = tf_data.experimental.CsvDataset(
        str(csv_path),
        record_defaults=["0", "default_b"],
        header=True,
        na_value="NA",
        select_cols=[0, 1],
    )
    assert list(csv_ds) == [["10", "default_b"], ["40", "default_b"]]

    csv_ds_all = tf_data.experimental.CsvDataset(
        str(csv_path),
        record_defaults=["0"],
        header=False,
    )
    assert len(list(csv_ds_all)) == 3
    assert list(tf_data.experimental.CsvDataset("nonexistent.csv", record_defaults=["d"])) == [
        ["d"]
    ]

    batched_csv = tf_data.experimental.make_csv_dataset(str(csv_path), batch_size=2)
    assert len(list(batched_csv)) == 2

    opt_present = tf_data.experimental.Optional.from_value(42)
    assert opt_present.has_value() is True
    assert opt_present.get_value() == 42

    opt_empty = tf_data.experimental.Optional.empty()
    assert opt_empty.has_value() is False
    with pytest.raises(ValueError, match="Optional has no value"):
        opt_empty.get_value()


def test_experimental_transformations(tmp_path: Path) -> None:
    """Verify `tf.data.experimental` dataset transformation functions.

    Args:
        tmp_path: Temporary directory fixture.
    """
    ds = tf_data.Dataset.from_tensor_slices([1, 2, 3, 4])
    assert list(ds.apply(tf_data.experimental.assert_cardinality(4))) == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="expected to contain 2"):
        list(ds.apply(tf_data.experimental.assert_cardinality(2)))

    assert list(
        ds.apply(tf_data.experimental.bucket_by_sequence_length(lambda x: 1, [2], [2, 2]))
    ) == [[1, 2], [3, 4]]
    assert list(ds.apply(tf_data.experimental.bucket_by_sequence_length())) == [
        [1],
        [2],
        [3],
        [4],
    ]

    d0 = tf_data.Dataset.from_tensor_slices(["a0", "a1"])
    d1 = tf_data.Dataset.from_tensor_slices(["b0", "b1"])
    choices = tf_data.Dataset.from_tensor_slices([0, 1, 0, 1, 0])
    assert list(tf_data.experimental.choose_from_datasets([d0, d1], choices)) == [
        "a0",
        "b0",
        "a1",
        "b1",
    ]
    assert list(tf_data.experimental.choose_from_datasets()) == []

    assert list(ds.apply(tf_data.experimental.copy_to_device("/cpu:0"))) == [1, 2, 3, 4]
    assert list(ds.apply(tf_data.experimental.dense_to_ragged_batch(2))) == [[1, 2], [3, 4]]
    assert list(ds.apply(tf_data.experimental.dense_to_sparse_batch(2, [4]))) == [[1, 2], [3, 4]]
    assert tf_data.experimental.enable_debug_mode() is True

    var = tf_data.experimental.to_variant(ds)
    assert list(tf_data.experimental.from_variant(var)) == [1, 2, 3, 4]
    assert list(tf_data.experimental.from_variant([9, 8])) == [9, 8]
    assert list(tf_data.experimental.to_variant([9, 8])) == [9, 8]

    it = tf_data.Iterator(tf_data.Dataset.from_tensor_slices([77]))
    opt1 = tf_data.experimental.get_next_as_optional(it)
    assert opt1.has_value() is True and opt1.get_value() == 77
    opt2 = tf_data.experimental.get_next_as_optional(it)
    assert opt2.has_value() is False

    opt_ds = tf_data.experimental.get_next_as_optional(tf_data.Dataset.from_tensor_slices([88]))
    assert opt_ds.has_value() is True and opt_ds.get_value() == 88
    opt_empty_ds = tf_data.experimental.get_next_as_optional(tf_data.Dataset.from_tensor_slices([]))
    assert opt_empty_ds.has_value() is False
    assert tf_data.experimental.get_next_as_optional(None).has_value() is False

    assert isinstance(tf_data.experimental.get_structure(ds), tf_data.DatasetSpec)
    assert isinstance(tf_data.experimental.get_structure(it), tf_data.IteratorSpec)
    assert isinstance(tf_data.experimental.get_structure(None), tf_data.DatasetSpec)

    reducer = tf_data.experimental.Reducer(
        init_func=lambda _: 0,
        reduce_func=lambda s, x: (
            (int(s) if isinstance(s, int) else 0) + (int(x) if isinstance(x, int) else 0)
        ),
        finalize_func=lambda s: (int(s) if isinstance(s, int) else 0) * 10,
    )
    grouped = ds.apply(
        tf_data.experimental.group_by_reducer(
            key_func=lambda x: (int(x) if isinstance(x, int) else 0) % 2,
            reducer=reducer,
        )
    )
    assert list(grouped) == [40, 60]
    assert list(ds.apply(tf_data.experimental.group_by_reducer())) == [1, 2, 3, 4]

    win_grouped = ds.apply(
        tf_data.experimental.group_by_window(
            key_func=lambda x: (int(x) if isinstance(x, int) else 0) % 2,
            reduce_func=lambda _, w: w,
            window_size=1,
        )
    )
    assert list(win_grouped) == [1, 2, 3, 4]
    win_grouped_rem = ds.apply(
        tf_data.experimental.group_by_window(
            key_func=lambda x: (int(x) if isinstance(x, int) else 0) % 2,
            reduce_func=lambda _, w: w,
            window_size=5,
        )
    )
    assert list(win_grouped_rem) == [1, 3, 2, 4]
    assert list(ds.apply(tf_data.experimental.group_by_window())) == [1, 2, 3, 4]

    def flaky_map(x: tf_data.DatasetElement) -> tf_data.DatasetElement:
        """Raise ValueError on 2, else return `x`.

        Args:
            x: Input dataset element.

        Returns:
            tf_data.DatasetElement: Unmodified `x`.

        Raises:
            ValueError: When `x == 2`.
        """
        if x == 2:
            raise ValueError("bad map item")
        return x

    assert list(
        tf_data.Dataset.from_tensor_slices([1, 2, 3])
        .map(flaky_map)
        .apply(tf_data.experimental.ignore_errors())
    ) == [1, 3]

    class FlakyDataset(tf_data.Dataset):
        """Dataset raising ValueError on its second element."""

        def _generator(self) -> Iterator[tf_data.DatasetElement]:
            """Yield 1, then raise ValueError."""
            yield 1
            raise ValueError("bad generator")

    assert list(FlakyDataset().apply(tf_data.experimental.ignore_errors())) == [1]

    vocab_ds = tf_data.Dataset.from_tensor_slices(["cat", "dog", "bird"])
    assert tf_data.experimental.index_table_from_dataset(vocab_ds, vocab_size=2) == {
        "cat": 0,
        "dog": 1,
    }
    kv_ds = tf_data.Dataset.from_tensor_slices([["k1", "v1"], ["k2", "v2"]])
    assert tf_data.experimental.table_from_dataset(kv_ds) == {"k1": "v1", "k2": "v2"}

    save_gz = str(tmp_path / "saved_ds.gz")
    assert tf_data.experimental.save(ds, save_gz, compression="GZIP") is True
    assert list(tf_data.experimental.load(save_gz, compression="GZIP")) == [1, 2, 3, 4]

    save_zlib = str(tmp_path / "saved_ds.zlib")
    assert tf_data.experimental.save(ds, save_zlib, compression="ZLIB") is True
    assert list(tf_data.experimental.load(save_zlib, compression="ZLIB")) == [1, 2, 3, 4]
    assert tf_data.experimental.save(None, "") is False
    assert list(tf_data.experimental.load("nonexistent_path")) == []

    tfr_path = str(tmp_path / "feats.tfrecord")
    with tf.io.TFRecordWriter(tfr_path) as w:
        w.write(b"f1")
        w.write(b"f2")
    assert list(tf_data.experimental.make_batched_features_dataset(tfr_path, batch_size=2)) == [
        [b"f1", b"f2"]
    ]

    assert tf_data.experimental.make_saveable_from_iterator(it) is it
    assert isinstance(tf_data.experimental.make_saveable_from_iterator(None), tf_data.Iterator)

    assert list(
        ds.apply(
            tf_data.experimental.map_and_batch(
                lambda x: (int(x) if isinstance(x, int) else 0) + 10, batch_size=2
            )
        )
    ) == [[11, 12], [13, 14]]
    assert list(ds.apply(tf_data.experimental.pad_to_cardinality(6))) == [1, 2, 3, 4, 0, 0]
    assert list(
        ds.apply(
            tf_data.experimental.parallel_interleave(
                lambda x: tf_data.Dataset.from_tensor_slices([x]), cycle_length=2
            )
        )
    ) == [1, 2, 3, 4]
    assert list(ds.apply(tf_data.experimental.parse_example_dataset())) == [1, 2, 3, 4]
    assert list(ds.apply(tf_data.experimental.prefetch_to_device("/cpu:0", buffer_size=2))) == [
        1,
        2,
        3,
        4,
    ]
    assert list(
        ds.apply(
            tf_data.experimental.rejection_resample(
                class_func=lambda x: (int(x) if isinstance(x, int) else 0) % 2
            )
        )
    ) == [[1, 1], [0, 2], [1, 3], [0, 4]]
    assert list(tf_data.experimental.sample_from_datasets([d0, d1])) == ["a0", "a1", "b0", "b1"]
    assert list(
        ds.apply(
            tf_data.experimental.scan(
                0,
                lambda st, x: (
                    (int(st) if isinstance(st, int) else 0) + (int(x) if isinstance(x, int) else 0),
                    (int(st) if isinstance(st, int) else 0) + (int(x) if isinstance(x, int) else 0),
                ),
            )
        )
    ) == [1, 3, 6, 10]
    assert tf_data.experimental.service(protocol="grpc") == {"protocol": "grpc"}
    assert len(list(ds.apply(tf_data.experimental.shuffle_and_repeat(buffer_size=2, count=2)))) == 8
    assert list(ds.apply(tf_data.experimental.snapshot())) == [1, 2, 3, 4]
