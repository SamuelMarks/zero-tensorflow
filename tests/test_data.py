"""Unit tests for `test_data`."""

from __future__ import annotations

from collections.abc import Iterator

from zero_tensorflow import data

import pytest


def test_threading_options() -> None:
    """Execute `test_threading_options`."""
    opt = data.ThreadingOptions(max_intra_op_parallelism=2, private_threadpool_size=4)
    assert opt.max_intra_op_parallelism == 2
    assert opt.private_threadpool_size == 4


def test_options() -> None:
    """Execute `test_options`."""
    opt = data.Options(autotune=True, dataset_name="name", threading=data.ThreadingOptions(2, 4))
    assert opt.autotune is True
    assert opt.options["dataset_name"] == "name"
    assert opt.threading.max_intra_op_parallelism == 2
    opt2 = data.Options()
    assert opt2.threading is not None


def test_dataset_spec() -> None:
    """Execute `test_dataset_spec`."""
    spec = data.DatasetSpec(element_spec="int", dataset_shape=(10,))
    assert spec._element_spec == "int"
    assert spec._dataset_shape == (10,)


def test_iterator_spec() -> None:
    """Execute `test_iterator_spec`."""
    spec = data.IteratorSpec(element_spec="int")
    assert spec._element_spec == "int"


def test_iterator() -> None:
    """Execute `test_iterator`."""
    ds = data.Dataset.from_tensor_slices([1, 2, 3])
    it = data.Iterator(ds)
    assert next(it) == 1
    assert next(iter(it)) == 2


def test_array_iterator() -> None:
    """Execute `test_array_iterator`."""
    ds = data.Dataset.from_tensor_slices([1, 2, 3])
    it = data.ArrayIterator(ds)
    assert next(it) == 1
    assert next(iter(it)) == 2


def test_dataset_base() -> None:
    """Execute `test_dataset_base`."""
    ds = data.Dataset()
    assert list(ds) == []


def test_dataset_from_tensor_slices() -> None:
    """Execute `test_dataset_from_tensor_slices`."""
    ds1 = data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    assert list(iter(ds1)) == [1, 2, 3, 4, 5]
    ds2 = data.Dataset.from_tensor_slices("hello")
    assert list(iter(ds2)) == ["h", "e", "l", "l", "o"]
    ds3 = data.Dataset.from_tensor_slices(None)
    assert list(iter(ds3)) == []


def test_dataset_batch() -> None:
    """Execute `test_dataset_batch`."""
    ds1 = data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    ds_batched = ds1.batch(2)
    assert list(ds_batched) == [[1, 2], [3, 4], [5]]
    ds_batched_drop = ds1.batch(2, drop_remainder=True)
    assert list(ds_batched_drop) == [[1, 2], [3, 4]]
    ds_pbatch = ds1.padded_batch(2, drop_remainder=False)
    assert list(ds_pbatch) == [[1, 2], [3, 4], [5]]
    assert ds_pbatch.element_spec is None
    assert data.experimental.Counter().cardinality() == data.INFINITE_CARDINALITY
    with pytest.raises(ValueError):
        ds1.batch(0)


def test_dataset_map() -> None:
    """Execute `test_dataset_map`."""
    ds1 = data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    ds_mapped = ds1.map(lambda x: (int(x) if isinstance(x, int) else 0) * 2)
    assert list(ds_mapped) == [2, 4, 6, 8, 10]
    ds_mapped_parallel = ds1.map(
        lambda x: (int(x) if isinstance(x, int) else 0) * 2, num_parallel_calls=2
    )
    assert list(ds_mapped_parallel) == [2, 4, 6, 8, 10]

    def error_map(x: data.DatasetElement) -> data.DatasetElement:
        """Execute `error_map`.

        Args:
            x: Parameter `x`.

        Returns:
            data.DatasetElement: Computed result.
        """
        if x == 3:
            raise ValueError("map error")
        return x

    ds_error = ds1.map(error_map, num_parallel_calls=2)
    it = iter(ds_error)
    assert next(it) == 1
    assert next(it) == 2
    with pytest.raises(ValueError, match="map error"):
        next(it)


def test_dataset_filter() -> None:
    """Execute `test_dataset_filter`."""
    ds1 = data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    ds_filtered = ds1.filter(lambda x: isinstance(x, int) and x % 2 == 0)
    assert list(ds_filtered) == [2, 4]


def test_dataset_shuffle() -> None:
    """Execute `test_dataset_shuffle`."""
    ds1 = data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    ds_shuffled = ds1.shuffle(10, seed=42)
    l1 = list(ds_shuffled)
    assert set(l1) == {1, 2, 3, 4, 5}
    l2 = list(ds_shuffled)
    assert l1 != l2
    ds_shuffled_no_reshuffle = ds1.shuffle(10, seed=42, reshuffle_each_iteration=False)
    assert list(ds_shuffled_no_reshuffle) == list(ds_shuffled_no_reshuffle)
    ds_shuffled_zero = ds1.shuffle(0)
    assert list(ds_shuffled_zero) == [1, 2, 3, 4, 5]


def test_dataset_prefetch() -> None:
    """Execute `test_dataset_prefetch`."""
    ds1 = data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    ds_prefetch = ds1.prefetch(2)
    assert list(ds_prefetch) == [1, 2, 3, 4, 5]
    ds_prefetch_auto = ds1.prefetch(-1)
    assert list(ds_prefetch_auto) == [1, 2, 3, 4, 5]

    def error_gen() -> Iterator[data.DatasetElement]:
        """Execute `error_gen`.

        Yields:
            Iterator[data.DatasetElement]: Yielded value.
        """
        yield 1
        raise ValueError("prefetch error")

    class ErrDataset(data.Dataset):
        """Test helper class `ErrDataset`."""

        def _generator(self) -> Iterator[data.DatasetElement]:
            """Execute `_generator`.

            Returns:
                Iterator[data.DatasetElement]: Computed result.
            """
            return error_gen()

    ds_err = ErrDataset().prefetch(2)
    it = iter(ds_err)
    assert next(it) == 1
    with pytest.raises(ValueError, match="prefetch error"):
        next(it)


def test_dataset_cache() -> None:
    """Execute `test_dataset_cache`."""
    ds1 = data.Dataset.from_tensor_slices([1, 2, 3])
    ds_cache = ds1.cache()
    assert list(ds_cache) == [1, 2, 3]
    assert list(ds_cache) == [1, 2, 3]
    ds_cache_file = ds1.cache("dummy_file")
    assert list(ds_cache_file) == [1, 2, 3]
    assert list(ds_cache_file) == [1, 2, 3]


def test_dataset_interleave() -> None:
    """Execute `test_dataset_interleave`."""
    ds1 = data.Dataset.from_tensor_slices([1, 2, 3])

    def map_func(x: data.DatasetElement) -> data.Dataset:
        """Execute `map_func`.

        Args:
            x: Parameter `x`.

        Returns:
            data.Dataset: Computed result.
        """
        v = int(x) if isinstance(x, int) else 0
        return data.Dataset.from_tensor_slices([v * 10, v * 10 + 1])

    ds_interleave = ds1.interleave(map_func, cycle_length=2, block_length=1)
    assert list(ds_interleave) == [10, 20, 11, 21, 30, 31]
    ds_interleave2 = ds1.interleave(map_func, cycle_length=1, block_length=2)
    assert list(ds_interleave2) == [10, 11, 20, 21, 30, 31]


def test_dataset_window() -> None:
    """Execute `test_dataset_window`."""
    ds1 = data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    ds_window = ds1.window(size=2, shift=2)
    windows = [list(w) for w in ds_window if isinstance(w, data.Dataset)]
    assert windows == [[1, 2], [3, 4], [5]]
    ds_window_drop = ds1.window(size=2, shift=2, drop_remainder=True)
    windows_drop = [list(w) for w in ds_window_drop if isinstance(w, data.Dataset)]
    assert windows_drop == [[1, 2], [3, 4]]


def test_fixed_length_record_dataset() -> None:
    """Execute `test_fixed_length_record_dataset`."""
    ds = data.FixedLengthRecordDataset(
        filenames="file.bin",
        record_bytes=10,
        header_bytes=1,
        footer_bytes=2,
        buffer_size=100,
        compression_type="GZIP",
        num_parallel_reads=4,
        name="my_dataset",
    )
    assert ds.filenames == "file.bin"
    assert ds.record_bytes == 10
    assert list(ds) == [b""]


def test_tf_record_dataset() -> None:
    """Execute `test_tf_record_dataset`."""
    ds = data.TFRecordDataset(
        filenames=["f1.tfrecord", "f2.tfrecord"],
        compression_type="ZLIB",
        buffer_size=200,
        num_parallel_reads=2,
        name="tfrec_ds",
    )
    assert ds.filenames == ["f1.tfrecord", "f2.tfrecord"]
    assert list(ds) == [b""]


def test_text_line_dataset() -> None:
    """Execute `test_text_line_dataset`."""
    ds = data.TextLineDataset(
        filenames="text.txt",
        compression_type=None,
        buffer_size=300,
        num_parallel_reads=1,
        name="text_ds",
    )
    assert ds.filenames == "text.txt"
    assert list(ds) == [""]


def test_dataset_map_input_exception() -> None:
    """Execute `test_dataset_map_input_exception`."""

    def error_gen() -> Iterator[data.DatasetElement]:
        """Execute `error_gen`.

        Yields:
            Iterator[data.DatasetElement]: Yielded value.
        """
        yield 1
        raise RuntimeError("input error")

    class ErrDataset(data.Dataset):
        """Test helper class `ErrDataset`."""

        def _generator(self) -> Iterator[data.DatasetElement]:
            """Execute `_generator`.

            Returns:
                Iterator[data.DatasetElement]: Computed result.
            """
            return error_gen()

    ds_err = ErrDataset().map(lambda x: x, num_parallel_calls=2)
    it = iter(ds_err)
    assert next(it) == 1
    with pytest.raises(RuntimeError, match="input error"):
        next(it)


def test_dataset_shuffle_buffer_full() -> None:
    """Execute `test_dataset_shuffle_buffer_full`."""
    ds1 = data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    ds_shuffled = ds1.shuffle(2, seed=42)
    assert len(list(ds_shuffled)) == 5


def test_dataset_interleave_pop() -> None:
    """Execute `test_dataset_interleave_pop`."""
    ds1 = data.Dataset.from_tensor_slices([1, 2, 3])

    def map_func(x: data.DatasetElement) -> data.Dataset:
        """Execute `map_func`.

        Args:
            x: Parameter `x`.

        Returns:
            data.Dataset: Computed result.
        """
        return data.Dataset.from_tensor_slices([x])

    ds_interleave = ds1.interleave(map_func, cycle_length=2, block_length=2)
    assert list(ds_interleave) == [1, 2, 3]
