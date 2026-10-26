"""Unit tests for `test_coverage_batch_3`."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import zero_tensorflow as tf

import pytest


def test_data_stubs(tmp_path: Path) -> None:
    """Verify `tf.data` datasets, options, iterators, and experimental transformations.

    Args:
        tmp_path: Temporary directory path fixture.
    """
    assert tf.data.AUTOTUNE == -1
    assert tf.data.INFINITE_CARDINALITY == -1
    assert tf.data.UNKNOWN_CARDINALITY == -2
    assert tf.data.experimental.AUTOTUNE == -1
    assert tf.data.experimental.INFINITE_CARDINALITY == -1
    assert tf.data.experimental.UNKNOWN_CARDINALITY == -2
    assert tf.data.experimental.SHARD_HINT == -1

    np_it = tf.data.Dataset.from_tensor_slices([tf.Tensor(1.0), tf.Tensor(2.0)]).as_numpy_iterator()
    assert list(np_it) == [1.0, 2.0]

    assert tf.data.experimental.AutoShardPolicy() is not None
    assert tf.data.experimental.AutotuneAlgorithm() is not None
    assert tf.data.experimental.AutotuneOptions() is not None
    assert tf.data.experimental.Counter() is not None
    assert tf.data.experimental.CsvDataset() is not None
    assert tf.data.experimental.DatasetInitializer() is not None
    assert tf.data.experimental.DistributeOptions() is not None
    assert tf.data.experimental.ExternalStatePolicy() is not None
    assert tf.data.experimental.OptimizationOptions() is not None
    assert tf.data.experimental.Optional() is not None
    assert tf.data.experimental.RandomDataset() is not None
    assert tf.data.experimental.Reducer() is not None
    assert tf.data.experimental.SqlDataset() is not None
    assert tf.data.experimental.TFRecordWriter() is not None

    assert list(tf.data.FixedLengthRecordDataset("nonexistent", kwarg1="a")) == [b""]
    assert list(tf.data.TFRecordDataset("nonexistent", kwarg1="a")) == [b""]
    assert list(tf.data.TextLineDataset("nonexistent", kwarg1="a")) == [""]

    txt_file = tmp_path / "lines.txt"
    nl = chr(10)
    txt_file.write_text(f"alpha{nl}beta{nl}", encoding="utf-8")
    assert list(tf.data.TextLineDataset(str(txt_file))) == ["alpha", "beta"]

    bin_file = tmp_path / "records.bin"
    bin_file.write_bytes(b"HEADER11112222FOOTER")
    assert list(
        tf.data.FixedLengthRecordDataset(
            [str(bin_file)], record_bytes=4, header_bytes=6, footer_bytes=6
        )
    ) == [b"1111", b"2222"]

    tfr_file = tmp_path / "data.tfrecord"
    with tf.io.TFRecordWriter(str(tfr_file)) as writer:
        writer.write(b"record_payload")
    assert list(tf.data.TFRecordDataset(str(tfr_file))) == [b"record_payload"]

    opt3 = tf.data.Options(threading=None, autotune=True, extra_kwarg=42)
    assert opt3.options["extra_kwarg"] == 42
    th_opt = tf.data.ThreadingOptions(max_intra_op_parallelism=2, private_threadpool_size=4)
    assert th_opt.max_intra_op_parallelism == 2
    assert th_opt.private_threadpool_size == 4
    opt4 = tf.data.Options(threading=th_opt)
    assert opt4.threading is th_opt

    class EmptySubDataset(tf.data.Dataset):
        """Test subclass inheriting default `_generator`."""

    assert list(EmptySubDataset()) == []
    assert tf.data.experimental.ThreadingOptions() is not None

    ds_map_fallback_type = tf.data.Dataset.from_tensor_slices([1]).map(
        lambda x: x, num_parallel_calls="not an int"
    )
    assert list(ds_map_fallback_type) == [1]

    ds_map_fallback_val = tf.data.Dataset.from_tensor_slices([1]).map(
        lambda x: x, num_parallel_calls=0
    )
    assert list(ds_map_fallback_val) == [1]

    ds_map_none = tf.data.Dataset.from_tensor_slices([1]).map(lambda x: x)
    assert list(ds_map_none) == [1]

    ds_spec = tf.data.DatasetSpec(kwarg1=1)
    assert ds_spec._element_spec is None
    it_spec = tf.data.IteratorSpec(kwarg1=2)
    assert it_spec._element_spec is None
    it_base = tf.data.Iterator(tf.data.Dataset.from_tensor_slices([1]), kwarg1=3)
    assert iter(it_base) is it_base
    assert next(it_base) == 1
    it_arr = tf.data.ArrayIterator(tf.data.Dataset.from_tensor_slices([1]))
    assert next(it_arr) == 1

    ds_three = tf.data.Dataset.from_tensor_slices([10, 20, 30])
    assert list(ds_three.take(2)) == [10, 20]
    assert list(ds_three.take(-1)) == [10, 20, 30]
    assert list(ds_three.skip(1)) == [20, 30]
    assert list(ds_three.repeat(2)) == [10, 20, 30, 10, 20, 30]
    assert list(ds_three.enumerate(start=5)) == [[5, 10], [6, 20], [7, 30]]
    assert list(ds_three.flat_map(lambda x: tf.data.Dataset.from_tensor_slices([x, x]))) == [
        10,
        10,
        20,
        20,
        30,
        30,
    ]
    assert list(tf.data.Dataset.zip((ds_three, ds_three))) == [
        [10, 10],
        [20, 20],
        [30, 30],
    ]
    assert list(ds_three.concatenate(tf.data.Dataset.from_tensor_slices([40]))) == [
        10,
        20,
        30,
        40,
    ]
    assert ds_three.cardinality() == 3

    assert tf.data.experimental.cardinality(ds_three) == 3
    assert tf.data.experimental.cardinality(None) is None
    assert tf.data.experimental.at(ds_three, 1) == 20
    assert tf.data.experimental.at(None, 0) is None
    assert list(ds_three.apply(tf.data.experimental.enumerate_dataset(1))) == [
        [1, 10],
        [2, 20],
        [3, 30],
    ]
    ds_from_list = tf.data.experimental.from_list([7, 8])
    assert ds_from_list is not None and list(ds_from_list) == [7, 8]
    assert tf.data.experimental.from_list(None) is None
    assert tf.data.experimental.get_single_element(tf.data.Dataset.from_tensor_slices([99])) == 99
    assert tf.data.experimental.get_single_element(None) is None
    with pytest.raises(ValueError, match="exactly 1 element"):
        tf.data.experimental.get_single_element(ds_three)

    tw_fn = tf.data.experimental.take_while(lambda x: isinstance(x, int) and x < 25)
    assert tw_fn is not None
    assert list(ds_three.apply(tw_fn)) == [10, 20]
    assert tf.data.experimental.take_while(None) is None

    assert list(ds_three.batch(2).apply(tf.data.experimental.unbatch())) == [10, 20, 30]
    assert list(
        tf.data.Dataset.from_tensor_slices([1, 2, 1, 3, 2]).apply(tf.data.experimental.unique())
    ) == [1, 2, 3]

    ds2_para = tf.data.Dataset.from_tensor_slices([1, 2, 3]).map(
        lambda x: (int(x) if isinstance(x, int) else 0) * 2, num_parallel_calls=2
    )
    res2_para = list(ds2_para)
    assert 2 in res2_para and 4 in res2_para and 6 in res2_para

    class ExplodingDataset(tf.data.Dataset):
        """Test dataset raising `RuntimeError` after yielding one item."""

        def _generator(self) -> Iterator[tf.data.DatasetElement]:
            """Yield `1` then raise `RuntimeError`."""
            yield 1
            raise RuntimeError("boom")

    ds2_explode = ExplodingDataset().map(
        lambda x: (int(x) if isinstance(x, int) else 0) * 2, num_parallel_calls=2
    )
    with pytest.raises(RuntimeError):
        list(ds2_explode)

    ds_map_default = tf.data.Dataset.from_tensor_slices([1]).map(
        lambda x: x, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    assert list(ds_map_default) == [1]

    def bad_gen() -> Iterator[tf.data.DatasetElement]:
        """Yield `1` then raise `ValueError`."""
        yield 1
        raise ValueError("prefetch err")

    class BadPrefetchDS(tf.data.Dataset):
        """Test dataset wrapping `bad_gen`."""

        def _generator(self) -> Iterator[tf.data.DatasetElement]:
            """Return `bad_gen()`."""
            return bad_gen()

    ds_pref_err = BadPrefetchDS().prefetch(2)
    it_pref_err = iter(ds_pref_err)
    assert next(it_pref_err) == 1
    with pytest.raises(ValueError):
        next(it_pref_err)

    def bad_map(x: tf.data.DatasetElement) -> tf.data.DatasetElement:
        """Raise `ValueError` unconditionally.

        Args:
            x: Input dataset element.

        Raises:
            ValueError: Always raised.
        """
        del x
        raise ValueError("bad map")

    ds2_err = tf.data.Dataset.from_tensor_slices([1]).map(bad_map, num_parallel_calls=1)
    with pytest.raises(ValueError):
        list(ds2_err)

    ds_filter = tf.data.Dataset.from_tensor_slices([1, 2, 3]).filter(
        lambda x: isinstance(x, int) and x % 2 != 0
    )
    assert list(ds_filter) == [1, 3]

    with pytest.raises(ValueError):
        tf.data.Dataset.from_tensor_slices([1]).batch(0)

    assert list(tf.data.Dataset.from_tensor_slices([1, 2, 3]).batch(2, drop_remainder=True)) == [
        [1, 2]
    ]
    assert list(tf.data.Dataset.from_tensor_slices([1, 2, 3]).batch(2, drop_remainder=False)) == [
        [1, 2],
        [3],
    ]

    ds_shuf = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4]).shuffle(
        4, seed=42, reshuffle_each_iteration=False
    )
    assert list(ds_shuf) == list(ds_shuf)

    ds_shuf_each = tf.data.Dataset.from_tensor_slices([1, 2, 3]).shuffle(2, seed=42)
    assert len(list(ds_shuf_each)) == 3 and len(list(ds_shuf_each)) == 3

    ds_shuf_none = tf.data.Dataset.from_tensor_slices([1, 2, 3]).shuffle(
        2, seed=None, reshuffle_each_iteration=False
    )
    assert sorted(int(x) for x in ds_shuf_none if isinstance(x, int)) == [1, 2, 3]

    ds_shuf_0 = tf.data.Dataset.from_tensor_slices([1, 2, 3]).shuffle(0)
    assert list(ds_shuf_0) == [1, 2, 3]

    ds_pref = tf.data.Dataset.from_tensor_slices([1]).prefetch(buffer_size=-1)
    assert list(ds_pref) == [1]

    ds_cache = tf.data.Dataset.from_tensor_slices([1, 2]).cache()
    assert list(ds_cache) == [1, 2]
    assert list(ds_cache) == [1, 2]

    ds_cache_file = tf.data.Dataset.from_tensor_slices([1, 2]).cache("test_file")
    assert list(ds_cache_file) == [1, 2]

    ds_inter = tf.data.Dataset.from_tensor_slices([1]).interleave(
        lambda x: tf.data.Dataset.from_tensor_slices([]), cycle_length=None
    )
    assert list(ds_inter) == []

    ds_inter2 = tf.data.Dataset.from_tensor_slices([1, 2]).interleave(
        lambda x: tf.data.Dataset.from_tensor_slices(
            [
                x if not isinstance(x, tf.data.Dataset) else 0,
                x if not isinstance(x, tf.data.Dataset) else 0,
            ]
        ),
        cycle_length=2,
        block_length=2,
    )
    assert list(ds_inter2) == [1, 1, 2, 2]

    ds_win = tf.data.Dataset.from_tensor_slices([1, 2, 3]).window(2, drop_remainder=True, shift=1)
    res_win = [list(x) for x in ds_win if isinstance(x, tf.data.Dataset)]
    assert res_win == [[1, 2], [2, 3]]

    ds_win2 = tf.data.Dataset.from_tensor_slices([1, 2, 3]).window(2, drop_remainder=False)
    res_win2 = [list(x) for x in ds_win2 if isinstance(x, tf.data.Dataset)]
    assert res_win2 == [[1, 2], [3]]

    assert list(tf.data.Dataset.from_tensor_slices(None)) == []
    assert list(tf.data.Dataset.from_tensor_slices("ab")) == ["a", "b"]
