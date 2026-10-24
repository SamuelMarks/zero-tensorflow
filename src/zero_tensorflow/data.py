"""TensorFlow `data` input pipeline and dataset transformations module."""

from __future__ import annotations

import os
import queue
import random
import threading
from collections.abc import Callable, Iterable, Sequence
from collections.abc import Iterator as PyIterator
from concurrent.futures import Future, ThreadPoolExecutor

from zero_tensorflow._typing import (
    DTypeLike,
    NumpyConvertibleProtocol,
    TensorOperand,
    as_operand,
)

from typing_extensions import TypeAlias

__all__ = [
    "AUTOTUNE",
    "ArrayIterator",
    "Dataset",
    "DatasetSpec",
    "FixedLengthRecordDataset",
    "INFINITE_CARDINALITY",
    "Iterator",
    "IteratorSpec",
    "NumpyIterator",
    "Options",
    "TFRecordDataset",
    "TextLineDataset",
    "ThreadingOptions",
    "UNKNOWN_CARDINALITY",
    "experimental",
]

DatasetElement: TypeAlias = "TensorOperand | Dataset"

AUTOTUNE: int = -1
"""Sentinel value instructing `tf.data` runtime to tune buffer sizes dynamically."""

INFINITE_CARDINALITY: int = -1
"""Sentinel value indicating a dataset has an infinite number of elements."""

UNKNOWN_CARDINALITY: int = -2
"""Sentinel value indicating a dataset's cardinality cannot be determined statically."""


class ThreadingOptions:
    """Threading options for dataset operations."""

    def __init__(
        self,
        max_intra_op_parallelism: int | None = None,
        private_threadpool_size: int | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `ThreadingOptions`.

        Args:
            max_intra_op_parallelism: Maximum intra-op parallelism threads.
            private_threadpool_size: Private threadpool size.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.max_intra_op_parallelism = max_intra_op_parallelism
        self.private_threadpool_size = private_threadpool_size
        self.args = args
        self.kwargs = kwargs


class Options:
    """Options for `tf.data.Dataset` execution and optimization."""

    def __init__(
        self,
        autotune: bool | None = None,
        threading: ThreadingOptions | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `Options`.

        Args:
            autotune: Whether to dynamically tune performance parameters.
            threading: Optional `ThreadingOptions` configuration.
            *args: Additional positional arguments.
            **kwargs: Additional keyword options stored in `self.options`.
        """
        self.autotune = autotune
        self.threading = threading if threading is not None else ThreadingOptions()
        self.args = args
        self.kwargs = kwargs
        self.options: dict[str, TensorOperand] = dict(kwargs)


class DatasetSpec:
    """Type specification for `tf.data.Dataset`."""

    __slots__ = ["_dataset_shape", "_element_spec"]

    def __init__(
        self,
        element_spec: TensorOperand = None,
        dataset_shape: TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `DatasetSpec`.

        Args:
            element_spec: Type specification for dataset elements.
            dataset_shape: Shape of the dataset.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self._element_spec = element_spec
        self._dataset_shape = dataset_shape


class IteratorSpec:
    """Type specification for `tf.data.Iterator`."""

    __slots__ = ["_element_spec"]

    def __init__(
        self,
        element_spec: TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `IteratorSpec`.

        Args:
            element_spec: Type specification for iterator elements.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self._element_spec = element_spec


class Iterator:
    """Represents the state of iterating through a `Dataset`."""

    def __init__(
        self,
        dataset: Dataset | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `Iterator`.

        Args:
            dataset: The `Dataset` to iterate over.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self._dataset = dataset
        self._iter: PyIterator[DatasetElement] = iter(dataset) if dataset is not None else iter(())

    def __next__(self) -> DatasetElement:
        """Return the next element from the iterator.

        Returns:
            DatasetElement: Next dataset element.
        """
        return next(self._iter)

    def __iter__(self) -> Iterator:
        """Return the iterator object itself.

        Returns:
            Iterator: Self reference.
        """
        return self


class ArrayIterator(Iterator):
    """An iterator that converts all dataset elements into NumPy-compatible arrays."""


class NumpyIterator(Iterator):
    """Iterator over a dataset yielding elements converted to NumPy arrays or Python values."""

    def __init__(
        self,
        dataset: Dataset | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `NumpyIterator`.

        Args:
            dataset: Source `Dataset` to iterate over.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(dataset, *args, **kwargs)
        self.args = args
        self.kwargs = kwargs

    def __next__(self) -> DatasetElement:
        """Return the next dataset element converted via `.numpy()` if supported.

        Returns:
            DatasetElement: Materialized NumPy/Python element.
        """
        elem = super().__next__()
        if isinstance(elem, NumpyConvertibleProtocol):
            return as_operand(elem.numpy())
        return elem


class Dataset:
    """Represents a potentially large set of elements and transformations."""

    _compiler_node: Iterable[DatasetElement] | None = None

    def __iter__(self) -> PyIterator[DatasetElement]:
        """Create an iterator for enumerating the elements of this dataset.

        Returns:
            PyIterator[DatasetElement]: Element generator iterator.
        """
        return self._generator()

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield elements from the dataset.

        Yields:
            DatasetElement: Dataset elements.
        """
        yield from ()

    @classmethod
    def from_tensor_slices(cls, tensors: TensorOperand) -> Dataset:
        """Create a `Dataset` whose elements are slices of the given tensors.

        Args:
            tensors: A dataset element, sequence, or string whose slices form the dataset.

        Returns:
            Dataset: Newly constructed `TensorSliceDataset`.
        """
        return TensorSliceDataset(tensors)

    @classmethod
    def zip(cls, datasets: Sequence[Dataset] | tuple[Dataset, ...]) -> Dataset:
        """Create a `Dataset` by zipping together the given datasets.

        Args:
            datasets: A sequence or tuple of `Dataset` instances to zip.

        Returns:
            Dataset: A new `ZipDataset`.
        """
        return ZipDataset(datasets)

    def map(
        self,
        map_func: Callable[[DatasetElement], DatasetElement],
        num_parallel_calls: int | str | None = None,
        deterministic: bool | None = None,
    ) -> Dataset:
        """Map `map_func` across the elements of this dataset.

        Args:
            map_func: Function mapping a dataset element to a transformed element.
            num_parallel_calls: Optional number of parallel worker threads.
            deterministic: Whether output order must be deterministic.

        Returns:
            Dataset: A new `MapDataset`.
        """
        return MapDataset(self, map_func, num_parallel_calls, deterministic)

    def filter(self, predicate: Callable[[DatasetElement], bool]) -> Dataset:
        """Filter this dataset according to `predicate`.

        Args:
            predicate: A function mapping a dataset element to a boolean.

        Returns:
            Dataset: A new `FilterDataset`.
        """
        return FilterDataset(self, predicate)

    def batch(self, batch_size: int, drop_remainder: bool = False) -> Dataset:
        """Combine consecutive elements of this dataset into batches.

        Args:
            batch_size: Number of consecutive elements to combine per batch.
            drop_remainder: Whether the last batch should be dropped if smaller than `batch_size`.

        Returns:
            Dataset: A new `BatchDataset`.
        """
        return BatchDataset(self, batch_size, drop_remainder)

    def unbatch(self) -> Dataset:
        """Split elements of a dataset into multiple elements along the batch dimension.

        Returns:
            Dataset: A new `UnbatchDataset`.
        """
        return UnbatchDataset(self)

    def shuffle(
        self,
        buffer_size: int,
        seed: int | None = None,
        reshuffle_each_iteration: bool | None = None,
    ) -> Dataset:
        """Randomly shuffle the elements of this dataset.

        Args:
            buffer_size: Number of elements from this dataset from which the new dataset will sample.
            seed: Optional random seed integer.
            reshuffle_each_iteration: Whether to reshuffle with a new permutation each epoch.

        Returns:
            Dataset: A new `ShuffleDataset`.
        """
        return ShuffleDataset(self, buffer_size, seed, reshuffle_each_iteration)

    def prefetch(self, buffer_size: int) -> Dataset:
        """Create a `Dataset` that prefetches elements from this dataset.

        Args:
            buffer_size: Maximum number of elements to buffer when prefetching.

        Returns:
            Dataset: A new `PrefetchDataset`.
        """
        return PrefetchDataset(self, buffer_size)

    def cache(self, filename: str = "") -> Dataset:
        """Cache the elements in this dataset in memory or on local storage.

        Args:
            filename: Optional path to a file on the filesystem to use for caching.

        Returns:
            Dataset: A new `CacheDataset`.
        """
        return CacheDataset(self, filename)

    def interleave(
        self,
        map_func: Callable[[DatasetElement], Dataset],
        cycle_length: int | None = None,
        block_length: int = 1,
        num_parallel_calls: int | None = None,
        deterministic: bool | None = None,
    ) -> Dataset:
        """Map `map_func` across this dataset and interleave the results.

        Args:
            map_func: Function mapping a dataset element to a `Dataset`.
            cycle_length: Number of input elements processed concurrently.
            block_length: Number of consecutive elements produced from each input element.
            num_parallel_calls: Optional number of parallel worker calls.
            deterministic: Whether output ordering must be deterministic.

        Returns:
            Dataset: A new `InterleaveDataset`.
        """
        return InterleaveDataset(
            self,
            map_func,
            cycle_length,
            block_length,
            num_parallel_calls,
            deterministic,
        )

    def window(
        self,
        size: int,
        shift: int | None = None,
        stride: int = 1,
        drop_remainder: bool = False,
    ) -> Dataset:
        """Combine input elements into a dataset of windows (nested datasets).

        Args:
            size: Number of elements per window.
            shift: Number of input elements by which the window moves on each iteration.
            stride: Stride of the input elements in the sliding window.
            drop_remainder: Whether windows with fewer than `size` elements should be dropped.

        Returns:
            Dataset: A new `WindowDataset`.
        """
        return WindowDataset(self, size, shift, stride, drop_remainder)

    def take(self, count: int) -> Dataset:
        """Create a `Dataset` with at most `count` elements from this dataset.

        Args:
            count: Number of elements to take (`-1` or negative takes all).

        Returns:
            Dataset: A new `TakeDataset`.
        """
        return TakeDataset(self, count)

    def skip(self, count: int) -> Dataset:
        """Create a `Dataset` that skips `count` elements from this dataset.

        Args:
            count: Number of elements to skip.

        Returns:
            Dataset: A new `SkipDataset`.
        """
        return SkipDataset(self, count)

    def repeat(self, count: int | None = None) -> Dataset:
        """Repeat this dataset `count` times.

        Args:
            count: Number of times to repeat (`None` or `-1` repeats indefinitely, capped at 0 if `<= 0`).

        Returns:
            Dataset: A new `RepeatDataset`.
        """
        return RepeatDataset(self, count)

    def enumerate(self, start: int = 0) -> Dataset:
        """Enumerate the elements of this dataset.

        Args:
            start: Starting index for enumeration.

        Returns:
            Dataset: A new `EnumerateDataset`.
        """
        return EnumerateDataset(self, start)

    def flat_map(self, map_func: Callable[[DatasetElement], Dataset]) -> Dataset:
        """Map `map_func` across this dataset and flatten the resulting datasets.

        Args:
            map_func: Function mapping an element to a `Dataset`.

        Returns:
            Dataset: A new `FlatMapDataset`.
        """
        return FlatMapDataset(self, map_func)

    def concatenate(self, dataset: Dataset) -> Dataset:
        """Create a `Dataset` by concatenating the given dataset with this dataset.

        Args:
            dataset: `Dataset` to concatenate after `self`.

        Returns:
            Dataset: A new `ConcatenateDataset`.
        """
        return ConcatenateDataset(self, dataset)

    def cardinality(self) -> int:
        """Return the cardinality (number of elements) of this dataset.

        Returns:
            int: Element count.
        """
        return sum(1 for _ in self)

    def as_numpy_iterator(self) -> NumpyIterator:
        """Return an iterator converting all elements of the dataset to NumPy/Python values.

        Returns:
            NumpyIterator: Iterator over materialized dataset elements.
        """
        return NumpyIterator(self)

    def apply(self, transformation_func: Callable[[Dataset], Dataset]) -> Dataset:
        """Apply a transformation function to this dataset.

        Args:
            transformation_func: Callable accepting a `Dataset` and returning a `Dataset`.

        Returns:
            Dataset: Transformed `Dataset`.
        """
        return transformation_func(self)


class TensorSliceDataset(Dataset):
    """A `Dataset` of slices from a tensor or sequence."""

    def __init__(self, elements: TensorOperand) -> None:
        """Initialize `TensorSliceDataset`.

        Args:
            elements: Iterable, string, or sequence of elements to slice.
        """
        if isinstance(elements, str):
            self.elements: list[DatasetElement] = list(elements)
        elif isinstance(elements, Iterable):
            self.elements = list(elements)
        else:
            self.elements = []

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield individual sliced elements.

        Yields:
            DatasetElement: Element from `self.elements`.
        """
        yield from self.elements


class MapDataset(Dataset):
    """A `Dataset` that maps a function over elements in its input."""

    def __init__(
        self,
        input_dataset: Dataset,
        map_func: Callable[[DatasetElement], DatasetElement],
        num_parallel_calls: int | str | None = None,
        deterministic: bool | None = None,
    ) -> None:
        """Initialize `MapDataset`.

        Args:
            input_dataset: Source `Dataset`.
            map_func: Transformation callable.
            num_parallel_calls: Optional number of parallel worker threads.
            deterministic: Optional determinism flag.
        """
        self._input_dataset = input_dataset
        self._map_func = map_func
        self._num_parallel_calls = num_parallel_calls
        self._deterministic = deterministic

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield mapped elements sequentially or via a thread pool.

        Yields:
            DatasetElement: Mapped dataset element.
        """
        if self._num_parallel_calls is None:
            for elem in self._input_dataset:
                yield self._map_func(elem)
        else:
            workers = (
                self._num_parallel_calls
                if isinstance(self._num_parallel_calls, int) and self._num_parallel_calls > 0
                else 4
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                q: queue.Queue[tuple[bool | None, Future[DatasetElement] | BaseException]] = (
                    queue.Queue(maxsize=workers * 2)
                )

                def submitter() -> None:
                    """Submit map tasks from the input dataset to the thread pool."""
                    try:
                        for elem in self._input_dataset:
                            fut = executor.submit(self._map_func, elem)
                            q.put((True, fut))
                    except BaseException as err:
                        q.put((False, err))
                    finally:
                        q.put((None, RuntimeError("done")))

                t = threading.Thread(target=submitter, daemon=True)
                t.start()
                while True:
                    ok, item = q.get()
                    if ok is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item.result()


class FilterDataset(Dataset):
    """A `Dataset` that filters its input according to a predicate function."""

    def __init__(
        self,
        input_dataset: Dataset,
        predicate: Callable[[DatasetElement], bool],
    ) -> None:
        """Initialize `FilterDataset`.

        Args:
            input_dataset: Source `Dataset`.
            predicate: Boolean filter predicate callable.
        """
        self._input_dataset = input_dataset
        self._predicate = predicate

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield elements for which `predicate(elem)` evaluates to True.

        Yields:
            DatasetElement: Filtered dataset element.
        """
        for elem in self._input_dataset:
            if self._predicate(elem):
                yield elem


class BatchDataset(Dataset):
    """A `Dataset` that batches contiguous elements from its input."""

    def __init__(
        self,
        input_dataset: Dataset,
        batch_size: int,
        drop_remainder: bool = False,
    ) -> None:
        """Initialize `BatchDataset`.

        Args:
            input_dataset: Source `Dataset`.
            batch_size: Number of elements per batch (`> 0`).
            drop_remainder: Whether to drop the final incomplete batch.

        Raises:
            ValueError: If `batch_size <= 0`.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self._input_dataset = input_dataset
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield lists of `batch_size` elements.

        Yields:
            DatasetElement: Batched list of elements.
        """
        batch: list[TensorOperand] = []
        for elem in self._input_dataset:
            if not isinstance(elem, Dataset):
                batch.append(elem)
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        if batch and not self._drop_remainder:
            yield batch


class UnbatchDataset(Dataset):
    """A `Dataset` that splits batched sequence elements into individual elements."""

    def __init__(self, input_dataset: Dataset) -> None:
        """Initialize `UnbatchDataset`.

        Args:
            input_dataset: Source batched `Dataset`.
        """
        self._input_dataset = input_dataset

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield unbatched individual elements.

        Yields:
            DatasetElement: Unbatched element.
        """
        for elem in self._input_dataset:
            if isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
                yield from elem
            else:
                yield elem


class ShuffleDataset(Dataset):
    """A `Dataset` that randomly shuffles the elements of its input."""

    def __init__(
        self,
        input_dataset: Dataset,
        buffer_size: int,
        seed: int | None = None,
        reshuffle_each_iteration: bool | None = None,
    ) -> None:
        """Initialize `ShuffleDataset`.

        Args:
            input_dataset: Source `Dataset`.
            buffer_size: Shuffle buffer size.
            seed: Optional random seed integer.
            reshuffle_each_iteration: Whether to advance the seed on each iteration.
        """
        self._input_dataset = input_dataset
        self._buffer_size = buffer_size
        self._seed = seed
        self._iteration_count: int = 0
        self._reshuffle_each_iteration = (
            reshuffle_each_iteration if reshuffle_each_iteration is not None else True
        )

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield shuffled elements sampled from the internal buffer.

        Yields:
            DatasetElement: Randomly sampled element.
        """
        effective_seed: int | None = (
            (self._seed + self._iteration_count if self._reshuffle_each_iteration else self._seed)
            if self._seed is not None
            else None
        )
        self._iteration_count += 1
        rng = random.Random(effective_seed)

        if self._buffer_size <= 0:
            yield from self._input_dataset
            return

        buffer: list[DatasetElement] = []
        for elem in self._input_dataset:
            buffer.append(elem)
            if len(buffer) >= self._buffer_size:
                idx = rng.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)

        while buffer:
            idx = rng.randint(0, len(buffer) - 1)
            yield buffer.pop(idx)


class PrefetchDataset(Dataset):
    """A `Dataset` that asynchronously prefetches elements from its input."""

    def __init__(self, input_dataset: Dataset, buffer_size: int | str) -> None:
        """Initialize `PrefetchDataset`.

        Args:
            input_dataset: Source `Dataset`.
            buffer_size: Maximum prefetch queue size.
        """
        self._input_dataset = input_dataset
        self._buffer_size = buffer_size

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield prefetched elements from a background producer thread.

        Yields:
            DatasetElement: Prefetched dataset element.
        """
        bs = (
            self._buffer_size
            if isinstance(self._buffer_size, int) and self._buffer_size > 0
            else 10
        )
        q: queue.Queue[tuple[bool | None, DatasetElement | BaseException | None]] = queue.Queue(
            maxsize=bs
        )

        def worker() -> None:
            """Produce elements from `_input_dataset` into the prefetch queue."""
            try:
                for elem in self._input_dataset:
                    q.put((True, elem))
            except BaseException as err:
                q.put((False, err))
            finally:
                q.put((None, None))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            ok, elem = q.get()
            if ok is None:
                break
            if isinstance(elem, BaseException):
                raise elem
            yield elem


class CacheDataset(Dataset):
    """A `Dataset` that caches elements of its input."""

    def __init__(self, input_dataset: Dataset, filename: str = "") -> None:
        """Initialize `CacheDataset`.

        Args:
            input_dataset: Source `Dataset`.
            filename: Optional cache file path (empty string caches in memory).
        """
        self._input_dataset = input_dataset
        self._filename = filename
        self._cache: list[DatasetElement] | None = None

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield elements from memory cache or underlying dataset.

        Yields:
            DatasetElement: Cached dataset element.
        """
        if self._filename == "":
            if self._cache is None:
                self._cache = []
                for elem in self._input_dataset:
                    self._cache.append(elem)
                    yield elem
            else:
                yield from self._cache
        else:
            yield from self._input_dataset


class InterleaveDataset(Dataset):
    """A `Dataset` that interleaves elements from mapped sub-datasets."""

    def __init__(
        self,
        input_dataset: Dataset,
        map_func: Callable[[DatasetElement], Dataset],
        cycle_length: int | None = None,
        block_length: int = 1,
        num_parallel_calls: int | None = None,
        deterministic: bool | None = None,
    ) -> None:
        """Initialize `InterleaveDataset`.

        Args:
            input_dataset: Source `Dataset`.
            map_func: Function mapping an input element to a `Dataset`.
            cycle_length: Number of datasets interleaved concurrently.
            block_length: Number of consecutive elements yielded per dataset in cycle.
            num_parallel_calls: Optional number of parallel worker calls.
            deterministic: Optional determinism flag.
        """
        self._input_dataset = input_dataset
        self._map_func = map_func
        self._cycle_length = cycle_length if cycle_length is not None else 1
        self._block_length = block_length
        self._num_parallel_calls = num_parallel_calls
        self._deterministic = deterministic

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield interleaved elements across active sub-dataset iterators.

        Yields:
            DatasetElement: Interleaved element.
        """
        iterators: list[PyIterator[DatasetElement]] = []
        input_iter = iter(self._input_dataset)

        def fill_iterators() -> None:
            """Replenish active sub-dataset iterators up to `_cycle_length`."""
            while len(iterators) < self._cycle_length:
                try:
                    elem = next(input_iter)
                    iterators.append(iter(self._map_func(elem)))
                except StopIteration:
                    break

        fill_iterators()
        while iterators:
            for i in range(len(iterators)):
                try:
                    for _ in range(self._block_length):
                        yield next(iterators[i])
                except StopIteration:
                    iterators.pop(i)
                    fill_iterators()
                    break


class WindowDataset(Dataset):
    """A `Dataset` that produces sliding window sub-datasets from its input."""

    def __init__(
        self,
        input_dataset: Dataset,
        size: int,
        shift: int | None = None,
        stride: int = 1,
        drop_remainder: bool = False,
    ) -> None:
        """Initialize `WindowDataset`.

        Args:
            input_dataset: Source `Dataset`.
            size: Window size.
            shift: Window shift step between iterations.
            stride: Element stride within each window.
            drop_remainder: Whether to drop trailing partial windows.
        """
        self._input_dataset = input_dataset
        self._size = size
        self._shift = shift if shift is not None else size
        self._stride = stride
        self._drop_remainder = drop_remainder

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield `TensorSliceDataset` windows over the input dataset.

        Yields:
            DatasetElement: Window sub-dataset.
        """
        elements: list[TensorOperand] = [
            e for e in self._input_dataset if not isinstance(e, Dataset)
        ]
        i = 0
        while i < len(elements):
            w = elements[i : i + self._size * self._stride : self._stride]
            if len(w) == self._size or (not self._drop_remainder and len(w) > 0):
                yield TensorSliceDataset(w)
            i += self._shift


class TakeDataset(Dataset):
    """A `Dataset` that takes at most `count` elements from its input."""

    def __init__(self, input_dataset: Dataset, count: int) -> None:
        """Initialize `TakeDataset`.

        Args:
            input_dataset: Source `Dataset`.
            count: Maximum number of elements (`-1` takes all).
        """
        self._input_dataset = input_dataset
        self._count = count

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield up to `_count` elements.

        Yields:
            DatasetElement: Taken dataset element.
        """
        if self._count < 0:
            yield from self._input_dataset
            return
        for taken, elem in enumerate(self._input_dataset):
            if taken >= self._count:
                break
            yield elem


class SkipDataset(Dataset):
    """A `Dataset` that skips the first `count` elements from its input."""

    def __init__(self, input_dataset: Dataset, count: int) -> None:
        """Initialize `SkipDataset`.

        Args:
            input_dataset: Source `Dataset`.
            count: Number of leading elements to skip.
        """
        self._input_dataset = input_dataset
        self._count = count

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield elements after skipping `_count` items.

        Yields:
            DatasetElement: Remaining dataset element.
        """
        skipped = 0
        for elem in self._input_dataset:
            if skipped < self._count:
                skipped += 1
                continue
            yield elem


class RepeatDataset(Dataset):
    """A `Dataset` that repeats its input a specified number of times."""

    def __init__(self, input_dataset: Dataset, count: int | None = None) -> None:
        """Initialize `RepeatDataset`.

        Args:
            input_dataset: Source `Dataset`.
            count: Number of repetitions (`None` defaults to 1 safe epoch if finite, `<= 0` yields none).
        """
        self._input_dataset = input_dataset
        self._count = count

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield repeated elements from `_input_dataset`.

        Yields:
            DatasetElement: Repeated dataset element.
        """
        reps = 1 if self._count is None else self._count
        for _ in range(max(0, reps)):
            yield from self._input_dataset


class EnumerateDataset(Dataset):
    """A `Dataset` that pairs each element with its sequential index."""

    def __init__(self, input_dataset: Dataset, start: int = 0) -> None:
        """Initialize `EnumerateDataset`.

        Args:
            input_dataset: Source `Dataset`.
            start: Starting index integer.
        """
        self._input_dataset = input_dataset
        self._start = start

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield `[index, element]` pairs.

        Yields:
            DatasetElement: Two-element `[index, element]` list.
        """
        idx = self._start
        for elem in self._input_dataset:
            if not isinstance(elem, Dataset):
                yield [idx, elem]
            idx += 1


class FlatMapDataset(Dataset):
    """A `Dataset` that maps a function to sub-datasets and flattens them."""

    def __init__(
        self,
        input_dataset: Dataset,
        map_func: Callable[[DatasetElement], Dataset],
    ) -> None:
        """Initialize `FlatMapDataset`.

        Args:
            input_dataset: Source `Dataset`.
            map_func: Function mapping an element to a `Dataset`.
        """
        self._input_dataset = input_dataset
        self._map_func = map_func

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield flattened elements from each mapped `Dataset`.

        Yields:
            DatasetElement: Flattened element.
        """
        for elem in self._input_dataset:
            sub_ds = self._map_func(elem)
            yield from sub_ds


class ZipDataset(Dataset):
    """A `Dataset` that zips multiple datasets together."""

    def __init__(self, datasets: Sequence[Dataset] | tuple[Dataset, ...]) -> None:
        """Initialize `ZipDataset`.

        Args:
            datasets: Sequence of `Dataset` instances to zip.
        """
        self._datasets = list(datasets)

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield lists of corresponding elements across `_datasets`.

        Yields:
            DatasetElement: Zipped element list.
        """
        for items in zip(*self._datasets):
            yield [i for i in items if not isinstance(i, Dataset)]


class ConcatenateDataset(Dataset):
    """A `Dataset` that concatenates two datasets sequentially."""

    def __init__(self, first: Dataset, second: Dataset) -> None:
        """Initialize `ConcatenateDataset`.

        Args:
            first: First `Dataset`.
            second: Second `Dataset`.
        """
        self._first = first
        self._second = second

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield all elements of `_first` followed by all elements of `_second`.

        Yields:
            DatasetElement: Concatenated element.
        """
        yield from self._first
        yield from self._second


class TakeWhileDataset(Dataset):
    """A `Dataset` that stops iteration when `predicate` evaluates to `False`."""

    def __init__(
        self,
        input_dataset: Dataset,
        predicate: Callable[[DatasetElement], bool],
    ) -> None:
        """Initialize `TakeWhileDataset`.

        Args:
            input_dataset: Source `Dataset`.
            predicate: Stopping condition predicate.
        """
        self._input_dataset = input_dataset
        self._predicate = predicate

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield elements while `_predicate(elem)` is `True`.

        Yields:
            DatasetElement: Element satisfying `_predicate`.
        """
        for elem in self._input_dataset:
            if not self._predicate(elem):
                break
            yield elem


class UniqueDataset(Dataset):
    """A `Dataset` that filters out duplicate elements from its input."""

    def __init__(self, input_dataset: Dataset) -> None:
        """Initialize `UniqueDataset`.

        Args:
            input_dataset: Source `Dataset`.
        """
        self._input_dataset = input_dataset

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield unique elements in order of first appearance.

        Yields:
            DatasetElement: Distinct element.
        """
        seen: list[DatasetElement] = []
        for elem in self._input_dataset:
            if elem not in seen:
                seen.append(elem)
                yield elem


def _iter_file_paths(filenames: TensorOperand) -> list[str]:
    """Extract existing file paths from a `filenames` operand.

    Args:
        filenames: Single path string or sequence of path strings.

    Returns:
        list[str]: Existing file paths.
    """
    if isinstance(filenames, str):
        return [filenames] if os.path.isfile(filenames) else []
    if isinstance(filenames, Sequence) and not isinstance(filenames, bytes):
        return [p for p in filenames if isinstance(p, str) and os.path.isfile(p)]
    return []


class FixedLengthRecordDataset(Dataset):
    """A `Dataset` of fixed-length records from one or more binary files."""

    def __init__(
        self,
        filenames: TensorOperand = None,
        record_bytes: int | None = None,
        header_bytes: int = 0,
        footer_bytes: int = 0,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `FixedLengthRecordDataset`.

        Args:
            filenames: File path(s) to read from.
            record_bytes: Number of bytes per record.
            header_bytes: Number of leading header bytes to skip.
            footer_bytes: Number of trailing footer bytes to omit.
            *args: Additional positional arguments.
            **kwargs: Additional keyword record options stored in `self.options`.
        """
        self.filenames = filenames
        self.record_bytes = record_bytes
        self.header_bytes = header_bytes
        self.footer_bytes = footer_bytes
        self.args = args
        self.kwargs = kwargs
        self.options: dict[str, TensorOperand] = dict(kwargs)

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield fixed-length binary records from files if present, else `b""`.

        Yields:
            DatasetElement: Byte string record.
        """
        paths = _iter_file_paths(self.filenames)
        rec_size = (
            self.record_bytes if isinstance(self.record_bytes, int) and self.record_bytes > 0 else 0
        )
        if paths and rec_size > 0:
            for p in paths:
                with open(p, "rb") as f:
                    data = f.read()
                end = len(data) - self.footer_bytes if self.footer_bytes > 0 else len(data)
                payload = data[self.header_bytes : end]
                for offset in range(0, len(payload), rec_size):
                    chunk = payload[offset : offset + rec_size]
                    if len(chunk) == rec_size:
                        yield chunk
            return
        yield b""


class TFRecordDataset(Dataset):
    """A `Dataset` comprising records from one or more TFRecord files."""

    def __init__(
        self,
        filenames: TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `TFRecordDataset`.

        Args:
            filenames: TFRecord file path(s) to read from.
            *args: Additional positional arguments.
            **kwargs: Additional keyword options stored in `self.options`.
        """
        self.filenames = filenames
        self.args = args
        self.kwargs = kwargs
        self.options: dict[str, TensorOperand] = dict(kwargs)

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield TFRecord byte strings.

        Yields:
            DatasetElement: Byte string record.
        """
        paths = _iter_file_paths(self.filenames)
        if paths:
            for p in paths:
                with open(p, "rb") as f:
                    yield f.read()
            return
        yield b""


class TextLineDataset(Dataset):
    """A `Dataset` comprising lines from one or more text files."""

    def __init__(
        self,
        filenames: TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `TextLineDataset`.

        Args:
            filenames: Text file path(s) to read from.
            *args: Additional positional arguments.
            **kwargs: Additional keyword options stored in `self.options`.
        """
        self.filenames = filenames
        self.args = args
        self.kwargs = kwargs
        self.options: dict[str, TensorOperand] = dict(kwargs)

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield text lines from `self.filenames` if files exist, else `""`.

        Yields:
            DatasetElement: Text line string without trailing newline.
        """
        paths = _iter_file_paths(self.filenames)
        if paths:
            for p in paths:
                with open(p, encoding="utf-8") as f:
                    yield from f.read().splitlines()
            return
        yield ""


class _SqlDatasetImpl(Dataset):
    """A `Dataset` consisting of the results from a SQL query."""

    def __init__(
        self,
        driver_name: str = "",
        data_source_name: str = "",
        query: str = "",
        output_types: DTypeLike | tuple[DTypeLike, ...] = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `_SqlDatasetImpl`.

        Args:
            driver_name: Database type (e.g., `'sqlite'`).
            data_source_name: Database connection string.
            query: SQL query string to execute.
            output_types: Tuple or single `tf.DType` specifying column types.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments stored in `self.options`.
        """
        self.driver_name = driver_name
        self.data_source_name = data_source_name
        self.query = query
        self.output_types = output_types
        self.args = args
        self.kwargs = kwargs
        self.options: dict[str, TensorOperand] = dict(kwargs)

    def _generator(self) -> PyIterator[tuple[TensorOperand, ...]]:
        """Yield empty record tuples for the stub SQL dataset.

        Yields:
            tuple[TensorOperand, ...]: Empty record tuple.
        """
        yield ()


class experimental:
    """Experimental `tf.data` APIs and options."""

    AUTOTUNE: int = -1
    """Experimental alias for `tf.data.AUTOTUNE`."""

    class AutoShardPolicy:
        """Policy for automatic dataset sharding in distributed training."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `AutoShardPolicy`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class AutotuneAlgorithm:
        """Algorithm controlling `tf.data` autotuning."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `AutotuneAlgorithm`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class AutotuneOptions:
        """Options controlling `tf.data` autotuning behavior."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `AutotuneOptions`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class Counter(Dataset):
        """Dataset factory producing a sequence of integer counts."""

        def __init__(
            self,
            start: int = 0,
            step: int = 1,
            dtype: DTypeLike = None,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `Counter`.

            Args:
                start: Starting count value.
                step: Step increment.
                dtype: Optional output data type.
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.start = start
            self.step = step
            self.dtype = dtype
            self.args = args
            self.kwargs = kwargs

    class CsvDataset(Dataset):
        """Dataset comprising parsed records from one or more CSV files."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `CsvDataset`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class DatasetInitializer:
        """Table initializer backed by a `tf.data.Dataset`."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `DatasetInitializer`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class DistributeOptions:
        """Options controlling dataset distribution across devices/workers."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `DistributeOptions`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class ExternalStatePolicy:
        """Policy for handling external state during dataset serialization."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `ExternalStatePolicy`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    INFINITE_CARDINALITY: int = -1
    """Experimental alias for `tf.data.INFINITE_CARDINALITY`."""

    class OptimizationOptions:
        """Options for `tf.data` graph optimizations."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `OptimizationOptions`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class Optional:
        """Represents a value that may or may not be present."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `Optional`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class RandomDataset(Dataset):
        """Dataset producing a stream of pseudorandom integers."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `RandomDataset`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class Reducer:
        """Reducer specification for `group_by_reducer`."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `Reducer`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    SHARD_HINT: int = -1
    """Sentinel constant indicating shard count should be supplied by `AutoShardPolicy`."""

    SqlDataset = _SqlDatasetImpl

    class TFRecordWriter:
        """Experimental writer for saving a dataset to a TFRecord file."""

        def __init__(self, *args: TensorOperand, **kwargs: TensorOperand) -> None:
            """Initialize `TFRecordWriter`.

            Args:
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.args = args
            self.kwargs = kwargs

    class ThreadingOptions(ThreadingOptions):
        """Experimental alias for `tf.data.ThreadingOptions`."""

    UNKNOWN_CARDINALITY: int = -2
    """Experimental alias for `tf.data.UNKNOWN_CARDINALITY`."""

    @staticmethod
    def assert_cardinality(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Assert the cardinality of the input dataset.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def at(
        dataset: Dataset | TensorOperand = None,
        index: int = 0,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> DatasetElement:
        """Return the element at a specific index in a dataset.

        Args:
            dataset: Input `Dataset`.
            index: Element index.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            DatasetElement: Element at `index` if `dataset` is a `Dataset`, else `None`.
        """
        del args, kwargs
        if isinstance(dataset, Dataset):
            items = list(dataset)
            if 0 <= index < len(items):
                return items[index]
        return None

    @staticmethod
    def bucket_by_sequence_length(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Bucket elements in a dataset by length and pad them.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def cardinality(
        dataset: Dataset | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> int | None:
        """Return the cardinality of `dataset`, if known.

        Args:
            dataset: Input `Dataset`.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            int | None: Dataset cardinality if `dataset` is a `Dataset`, else `None`.
        """
        del args, kwargs
        if isinstance(dataset, Dataset):
            return dataset.cardinality()
        return None

    @staticmethod
    def choose_from_datasets(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Create a dataset that deterministically chooses elements from `datasets`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def copy_to_device(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Create a transformation that copies dataset elements to a target device.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def dense_to_ragged_batch(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Batch elements of unequal length into `tf.RaggedTensor`s.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def dense_to_sparse_batch(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Batch elements of unequal shape into `tf.sparse.SparseTensor`s.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def enable_debug_mode(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Enable eager execution of `tf.data` transformations for debugging.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def enumerate_dataset(
        start: int | TensorOperand = 0,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Return a transformation function that enumerates the elements of a dataset.

        Args:
            start: Starting index for enumeration.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Transformation callable for `Dataset.apply`.
        """
        del args, kwargs
        idx_start = start if isinstance(start, int) and not isinstance(start, bool) else 0
        return lambda ds: ds.enumerate(idx_start)

    @staticmethod
    def from_list(
        elements: TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Dataset | None:
        """Create a `Dataset` comprising the given list of elements.

        Args:
            elements: Sequence of dataset elements.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Dataset | None: Constructed `TensorSliceDataset` if `elements` is provided, else `None`.
        """
        del args, kwargs
        if elements is None:
            return None
        return TensorSliceDataset(elements)

    @staticmethod
    def from_variant(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Construct a dataset from the given variant tensor and structure.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def get_next_as_optional(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Return an `Optional` wrapping the next element of the iterator.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def get_single_element(
        dataset: Dataset | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> DatasetElement:
        """Return the single element of `dataset`.

        Args:
            dataset: Input `Dataset` containing exactly one element.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            DatasetElement: The single element of `dataset`, or `None` if not a `Dataset`.

        Raises:
            ValueError: If `dataset` is a `Dataset` that does not have exactly 1 element.
        """
        del args, kwargs
        if isinstance(dataset, Dataset):
            items = list(dataset)
            if len(items) != 1:
                raise ValueError(f"Dataset must have exactly 1 element, got {len(items)}.")
            return items[0]
        return None

    @staticmethod
    def get_structure(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Return the type specification of the elements of a dataset or iterator.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def group_by_reducer(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Group elements by key and reduce them using a `Reducer`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def group_by_window(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Group elements by key and reduce windows of matching elements.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def ignore_errors(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Create a dataset that silently drops elements that cause errors.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def index_table_from_dataset(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Create an index lookup table from a dataset.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def load(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Load a previously saved dataset from disk.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def make_batched_features_dataset(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Return a dataset of batched feature dicts from `Example` protos.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def make_csv_dataset(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Read CSV files into a dataset of feature dictionaries.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def make_saveable_from_iterator(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Return a SaveableObject for saving/restoring iterator state.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def map_and_batch(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Fused implementation of `map` and `batch`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def pad_to_cardinality(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Pad a dataset with dummy elements up to a target cardinality.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def parallel_interleave(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Map `map_func` across input and interleave results in parallel.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def parse_example_dataset(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Transform a dataset of serialized `Example`s into parsed tensors.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def prefetch_to_device(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Transform a dataset by prefetching elements to a target device.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def rejection_resample(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Resample a dataset to achieve a target class distribution.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def sample_from_datasets(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Sample elements at random from the datasets in `datasets`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def save(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Save the content of the given dataset to disk.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def scan(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Transform a dataset by scanning a stateful function across its elements.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def service(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Experimental `tf.data.experimental.service` namespace stub.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def shuffle_and_repeat(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Shuffle and repeat a dataset.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def snapshot(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Persist the output of a dataset pipeline to disk on the fly.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def table_from_dataset(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Create a lookup table from a dataset of `(key, value)` pairs.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def take_while(
        predicate: Callable[[DatasetElement], bool] | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset] | None:
        """Stop dataset iteration when `predicate` evaluates to `False`.

        Args:
            predicate: Stopping condition predicate callable.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Callable[[Dataset], Dataset] | None: Transformation callable if `predicate` is callable.
        """
        del args, kwargs
        if callable(predicate):
            return lambda ds: TakeWhileDataset(ds, predicate)
        return None

    @staticmethod
    def to_variant(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Return a scalar variant tensor representing the given dataset.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return

    @staticmethod
    def unbatch(*args: TensorOperand, **kwargs: TensorOperand) -> Callable[[Dataset], Dataset]:
        """Split elements of a dataset into multiple elements along the batch dimension.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Unbatch transformation callable.
        """
        del args, kwargs
        return lambda ds: ds.unbatch()

    @staticmethod
    def unique(*args: TensorOperand, **kwargs: TensorOperand) -> Callable[[Dataset], Dataset]:
        """Filter duplicate elements from a dataset.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Unique-element transformation callable.
        """
        del args, kwargs
        return lambda ds: UniqueDataset(ds)
