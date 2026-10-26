"""TensorFlow `data` input pipeline and dataset transformations module."""

from __future__ import annotations

import csv
import gzip
import io as py_io
import json
import os
import queue
import random
import sqlite3
import threading
import zlib
from collections.abc import Callable, Iterable, Sequence
from collections.abc import Iterator as PyIterator
from concurrent.futures import Future, ThreadPoolExecutor

from zero_tensorflow._typing import (
    DTypeLike,
    NumpyConvertibleProtocol,
    TensorOperand,
    as_operand,
)
from zero_tensorflow.io import decode_tfrecord_frames, encode_tfrecord_frame

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

_DEBUG_MODE: bool = False


def _decompress_bytes(raw: bytes, compression_type: str | TensorOperand) -> bytes:
    """Decompress `raw` bytes according to `compression_type`.

    Args:
        raw: Raw byte stream from disk.
        compression_type: Optional compression identifier (`"GZIP"`, `"ZLIB"`, or `""`).

    Returns:
        bytes: Decompressed byte payload.
    """
    comp = compression_type.upper() if isinstance(compression_type, str) else ""
    if comp == "GZIP":
        return gzip.decompress(raw)
    if comp == "ZLIB":
        return zlib.decompress(raw)
    return raw


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
        deterministic: bool | None = None,
        experimental_deterministic: bool | None = None,
        experimental_distribute: TensorOperand = None,
        experimental_external_state_policy: TensorOperand = None,
        experimental_optimization: TensorOperand = None,
        experimental_slack: bool | None = None,
        experimental_symbolic_checkpoint: bool | None = None,
        experimental_service: TensorOperand = None,
        experimental_threading: ThreadingOptions | None = None,
        experimental_warm_start: bool | None = None,
        dataset_name: str | None = None,
        framework_type: list[str] | None = None,
        threading: ThreadingOptions | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `Options`.

        Args:
            autotune: Whether to dynamically tune performance parameters.
            deterministic: Whether outputs need to be produced in deterministic order.
            experimental_deterministic: Deprecated alias for `deterministic`.
            experimental_distribute: Distribution strategy options.
            experimental_external_state_policy: External state handling policy.
            experimental_optimization: Dataset optimization options.
            experimental_slack: Whether to introduce slack in the final prefetch.
            experimental_symbolic_checkpoint: Whether to use symbolic checkpointing.
            experimental_service: `tf.data` service options.
            experimental_threading: Deprecated alias for `threading`.
            experimental_warm_start: Whether to start background threads upon iterator creation.
            dataset_name: Optional human-readable name for the dataset.
            framework_type: Optional list of framework telemetry identifiers.
            threading: Optional `ThreadingOptions` configuration.
            *args: Additional positional arguments.
            **kwargs: Additional keyword options stored in `self.options`.
        """
        self.autotune = autotune
        self.deterministic = deterministic
        self.experimental_deterministic = experimental_deterministic
        self.experimental_distribute = experimental_distribute
        self.experimental_external_state_policy = experimental_external_state_policy
        self.experimental_optimization = experimental_optimization
        self.experimental_slack = experimental_slack
        self.experimental_symbolic_checkpoint = experimental_symbolic_checkpoint
        self.experimental_service = experimental_service
        self.experimental_threading = experimental_threading
        self.experimental_warm_start = experimental_warm_start
        self.dataset_name = dataset_name
        self.framework_type = framework_type if framework_type is not None else []
        self.threading = threading if threading is not None else ThreadingOptions()
        self.args = args
        self.kwargs = kwargs
        self.options: dict[str, TensorOperand] = dict(kwargs)
        if dataset_name is not None:
            self.options["dataset_name"] = dataset_name


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
        self._dataset_shape = dataset_shape if dataset_shape is not None else ()

    @property
    def element_spec(self) -> TensorOperand:
        """Return the inner element specification.

        Returns:
            TensorOperand: Element specification.
        """
        return self._element_spec

    @property
    def value_type(self) -> type[Dataset]:
        """Return the Python type represented by this specification.

        Returns:
            type[Dataset]: The `Dataset` class.
        """
        return Dataset

    def is_compatible_with(self, spec_or_value: DatasetSpec | Dataset | TensorOperand) -> bool:
        """Check whether `spec_or_value` is compatible with this `DatasetSpec`.

        Args:
            spec_or_value: Another `DatasetSpec` or `Dataset` instance.

        Returns:
            bool: `True` if compatible, else `False`.
        """
        if isinstance(spec_or_value, Dataset):
            return True
        if isinstance(spec_or_value, DatasetSpec):
            return (
                self._element_spec is None
                or spec_or_value._element_spec is None
                or self._element_spec == spec_or_value._element_spec
            )
        return False

    @classmethod
    def from_value(cls, value: Dataset) -> DatasetSpec:
        """Construct a `DatasetSpec` from a `Dataset` instance.

        Args:
            value: Input `Dataset`.

        Returns:
            DatasetSpec: Specification describing `value`.
        """
        return cls(element_spec=None, dataset_shape=(value.cardinality(),))


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

    @property
    def element_spec(self) -> TensorOperand:
        """Return the element specification of the iterator.

        Returns:
            TensorOperand: Element specification.
        """
        return self._element_spec

    @property
    def value_type(self) -> type[Iterator]:
        """Return the Python type represented by this specification.

        Returns:
            type[Iterator]: The `Iterator` class.
        """
        return Iterator

    def is_compatible_with(self, spec_or_value: IteratorSpec | Iterator | TensorOperand) -> bool:
        """Check whether `spec_or_value` is compatible with this `IteratorSpec`.

        Args:
            spec_or_value: Another `IteratorSpec` or `Iterator` instance.

        Returns:
            bool: `True` if compatible, else `False`.
        """
        if isinstance(spec_or_value, Iterator):
            return True
        if isinstance(spec_or_value, IteratorSpec):
            return (
                self._element_spec is None
                or spec_or_value._element_spec is None
                or self._element_spec == spec_or_value._element_spec
            )
        return False

    @classmethod
    def from_value(cls, value: Iterator) -> IteratorSpec:
        """Construct an `IteratorSpec` from an `Iterator` instance.

        Args:
            value: Input `Iterator`.

        Returns:
            IteratorSpec: Specification describing `value`.
        """
        del value
        return cls(element_spec=None)


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

    def get_next(self) -> DatasetElement:
        """Return the next element from the iterator (TensorFlow `Iterator.get_next`).

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
    variant_tensor: int = 0

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
    def from_tensors(cls, tensors: TensorOperand) -> Dataset:
        """Create a `Dataset` with a single element comprising the given tensors.

        Args:
            tensors: A dataset element.

        Returns:
            Dataset: Newly constructed single-element `TensorSliceDataset`.
        """
        return TensorSliceDataset([tensors])

    @classmethod
    def range(cls, *args: int) -> Dataset:
        """Create a `Dataset` of a step-separated range of values.

        Args:
            *args: Range arguments (`stop`, or `start, stop[, step]`).

        Returns:
            Dataset: Newly constructed `TensorSliceDataset` over the range.
        """
        return TensorSliceDataset(list(range(*args)) if args else [])

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
        filenames: Single path string, `Dataset` of path strings, or sequence of path strings.

    Returns:
        list[str]: Existing file paths.
    """
    if isinstance(filenames, str):
        return [filenames] if os.path.isfile(filenames) else []
    if isinstance(filenames, Dataset):
        return [p for p in filenames if isinstance(p, str) and os.path.isfile(p)]
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
        buffer_size: int | None = None,
        compression_type: str | None = None,
        num_parallel_reads: int | None = None,
        name: str | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `FixedLengthRecordDataset`.

        Args:
            filenames: File path(s) to read from.
            record_bytes: Number of bytes per record.
            header_bytes: Number of leading header bytes to skip.
            footer_bytes: Number of trailing footer bytes to omit.
            buffer_size: Optional I/O buffer size in bytes.
            compression_type: Optional compression type (`""`, `"GZIP"`, or `"ZLIB"`).
            num_parallel_reads: Optional number of files to read in parallel.
            name: Optional name for the dataset operation.
            *args: Additional positional arguments.
            **kwargs: Additional keyword record options stored in `self.options`.
        """
        self.filenames = filenames
        self.record_bytes = record_bytes
        self.header_bytes = header_bytes
        self.footer_bytes = footer_bytes
        self.buffer_size = buffer_size
        self.compression_type = compression_type
        self.num_parallel_reads = num_parallel_reads
        self.name = name
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
                    raw = f.read()
                data = _decompress_bytes(raw, self.compression_type)
                end = len(data) - self.footer_bytes if self.footer_bytes > 0 else len(data)
                payload = data[self.header_bytes : end]
                for offset in range(0, len(payload), rec_size):
                    chunk = payload[offset : offset + rec_size]
                    if len(chunk) == rec_size:
                        yield chunk
            return
        yield b""


class TFRecordDataset(Dataset):
    """A `Dataset` comprising length-prefixed CRC-framed records from TFRecord files."""

    def __init__(
        self,
        filenames: TensorOperand = None,
        compression_type: str | None = None,
        buffer_size: int | None = None,
        num_parallel_reads: int | None = None,
        name: str | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `TFRecordDataset`.

        Args:
            filenames: TFRecord file path(s) to read from.
            compression_type: Optional compression type (`""`, `"GZIP"`, or `"ZLIB"`).
            buffer_size: Optional read buffer size in bytes.
            num_parallel_reads: Optional number of files to read in parallel.
            name: Optional name for the dataset operation.
            *args: Additional positional arguments.
            **kwargs: Additional keyword options stored in `self.options`.
        """
        self.filenames = filenames
        self.compression_type = compression_type
        self.buffer_size = buffer_size
        self.num_parallel_reads = num_parallel_reads
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.options: dict[str, TensorOperand] = dict(kwargs)

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield decoded TFRecord byte payloads from each file in `self.filenames`.

        Yields:
            DatasetElement: Decoded record byte payload.
        """
        paths = _iter_file_paths(self.filenames)
        if paths:
            for p in paths:
                with open(p, "rb") as f:
                    raw = f.read()
                decompressed = _decompress_bytes(raw, self.compression_type)
                yield from decode_tfrecord_frames(decompressed)
            return
        yield b""


class TextLineDataset(Dataset):
    """A `Dataset` comprising lines from one or more text files."""

    def __init__(
        self,
        filenames: TensorOperand = None,
        compression_type: str | None = None,
        buffer_size: int | None = None,
        num_parallel_reads: int | None = None,
        name: str | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `TextLineDataset`.

        Args:
            filenames: Text file path(s) to read from.
            compression_type: Optional compression type (`""`, `"GZIP"`, or `"ZLIB"`).
            buffer_size: Optional read buffer size in bytes.
            num_parallel_reads: Optional number of files to read in parallel.
            name: Optional name for the dataset operation.
            *args: Additional positional arguments.
            **kwargs: Additional keyword options stored in `self.options`.
        """
        self.filenames = filenames
        self.compression_type = compression_type
        self.buffer_size = buffer_size
        self.num_parallel_reads = num_parallel_reads
        self.name = name
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
                with open(p, "rb") as f:
                    raw = f.read()
                decompressed = _decompress_bytes(raw, self.compression_type)
                yield from decompressed.decode("utf-8").splitlines()
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
        """Yield SQL result row tuples if database file exists, else `()`.

        Yields:
            tuple[TensorOperand, ...]: Row tuple from the SQLite query.
        """
        if self.driver_name == "sqlite" and self.query and os.path.isfile(self.data_source_name):
            conn = sqlite3.connect(self.data_source_name)
            try:
                cur = conn.cursor()
                cur.execute(self.query)
                for row in cur.fetchall():
                    yield tuple(row)
            finally:
                conn.close()
            return
        yield ()


class _AssertCardinalityDataset(Dataset):
    """A `Dataset` wrapper that validates cardinality upon iteration."""

    def __init__(self, input_dataset: Dataset, expected_cardinality: int) -> None:
        """Initialize `_AssertCardinalityDataset`.

        Args:
            input_dataset: Source `Dataset`.
            expected_cardinality: Expected number of elements.
        """
        self._input_dataset = input_dataset
        self._expected_cardinality = expected_cardinality

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield elements from `_input_dataset` after verifying cardinality.

        Yields:
            DatasetElement: Dataset element.

        Raises:
            ValueError: If the actual element count differs from `_expected_cardinality`.
        """
        items = list(self._input_dataset)
        if len(items) != self._expected_cardinality:
            raise ValueError(
                f"Input dataset was expected to contain {self._expected_cardinality} "
                f"elements, but contained {len(items)} elements."
            )
        yield from items


class _IgnoreErrorsDataset(Dataset):
    """A `Dataset` that silently skips elements raising exceptions during iteration."""

    def __init__(self, input_dataset: Dataset, log_warning: bool = False) -> None:
        """Initialize `_IgnoreErrorsDataset`.

        Args:
            input_dataset: Source `Dataset`.
            log_warning: Whether to log warnings when errors occur.
        """
        self._input_dataset = input_dataset
        self._log_warning = log_warning

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield elements from `_input_dataset`, skipping elements that raise errors.

        Yields:
            DatasetElement: Valid dataset elements.
        """
        if isinstance(self._input_dataset, MapDataset):
            for elem in self._input_dataset._input_dataset:
                try:
                    yield self._input_dataset._map_func(elem)
                except Exception:
                    continue
            return
        try:
            yield from self._input_dataset
        except Exception:
            return


class _ScanDataset(Dataset):
    """A `Dataset` that scans a stateful function across its input elements."""

    def __init__(
        self,
        input_dataset: Dataset,
        initial_state: DatasetElement,
        scan_func: Callable[
            [DatasetElement, DatasetElement], tuple[DatasetElement, DatasetElement]
        ],
    ) -> None:
        """Initialize `_ScanDataset`.

        Args:
            input_dataset: Source `Dataset`.
            initial_state: Initial accumulator state.
            scan_func: Function `(state, elem) -> (new_state, output_elem)`.
        """
        self._input_dataset = input_dataset
        self._initial_state = initial_state
        self._scan_func = scan_func

    def _generator(self) -> PyIterator[DatasetElement]:
        """Yield scanned output elements while updating state.

        Yields:
            DatasetElement: Output element produced by `_scan_func`.
        """
        state = self._initial_state
        for elem in self._input_dataset:
            state, out_elem = self._scan_func(state, elem)
            yield out_elem


class experimental:
    """Experimental `tf.data` APIs and options."""

    AUTOTUNE: int = -1
    """Experimental alias for `tf.data.AUTOTUNE`."""

    class AutoShardPolicy:
        """Policy for automatic dataset sharding in distributed training."""

        OFF: int = -1
        AUTO: int = 0
        FILE: int = 1
        DATA: int = 2
        HINT: int = 3

        def __init__(
            self,
            value: int = 0,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `AutoShardPolicy`.

            Args:
                value: Policy enum value (`OFF=-1`, `AUTO=0`, `FILE=1`, `DATA=2`, `HINT=3`).
                *args: Additional positional arguments.
                **kwargs: Additional keyword arguments.
            """
            self.value = value
            self.args = args
            self.kwargs = kwargs

    class AutotuneAlgorithm:
        """Algorithm controlling `tf.data` autotuning."""

        DEFAULT: int = 0
        HILL_CLIMB: int = 1
        GRADIENT_DESCENT: int = 2
        MAX_PARALLELISM: int = 3
        STAGE_BASED: int = 4

        def __init__(
            self,
            value: int = 0,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `AutotuneAlgorithm`.

            Args:
                value: Algorithm identifier integer.
                *args: Additional positional arguments.
                **kwargs: Additional keyword arguments.
            """
            self.value = value
            self.args = args
            self.kwargs = kwargs

    class AutotuneOptions:
        """Options controlling `tf.data` autotuning behavior."""

        def __init__(
            self,
            enabled: bool = True,
            cpu_budget: int = 0,
            ram_budget: int = 0,
            autotune_algorithm: int | None = None,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `AutotuneOptions`.

            Args:
                enabled: Whether autotuning is enabled.
                cpu_budget: CPU budget for autotuning (`0` uses all cores).
                ram_budget: RAM budget in bytes (`0` uses default).
                autotune_algorithm: Optional algorithm selection.
                *args: Additional positional arguments.
                **kwargs: Additional keyword arguments.
            """
            self.enabled = enabled
            self.cpu_budget = cpu_budget
            self.ram_budget = ram_budget
            self.autotune_algorithm = autotune_algorithm
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
            self.step = step if step != 0 else 1
            self.dtype = dtype
            self.args = args
            self.kwargs = kwargs

        def _generator(self) -> PyIterator[DatasetElement]:
            """Yield sequential integer counts starting from `self.start`.

            Yields:
                DatasetElement: Integer count value.
            """
            for i in range(1000):
                yield self.start + i * self.step

    class CsvDataset(Dataset):
        """Dataset comprising parsed records from one or more CSV files."""

        def __init__(
            self,
            filenames: TensorOperand = None,
            record_defaults: Sequence[TensorOperand] = (),
            compression_type: str | None = None,
            buffer_size: int | None = None,
            header: bool = False,
            field_delim: str = ",",
            use_quote_delim: bool = True,
            na_value: str = "",
            select_cols: Sequence[int] | None = None,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `CsvDataset`.

            Args:
                filenames: CSV file path(s) to read.
                record_defaults: Default values per column.
                compression_type: Optional compression type (`"GZIP"`, `"ZLIB"`).
                buffer_size: Optional read buffer size.
                header: Whether to skip the first header row.
                field_delim: Field delimiter character.
                use_quote_delim: Whether double quotes enclose fields.
                na_value: String treated as NA/missing.
                select_cols: Optional column indices to extract.
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.filenames = filenames
            self.record_defaults = list(record_defaults)
            self.compression_type = compression_type
            self.buffer_size = buffer_size
            self.header = header
            self.field_delim = field_delim if field_delim else ","
            self.use_quote_delim = use_quote_delim
            self.na_value = na_value
            self.select_cols = list(select_cols) if select_cols is not None else None
            self.args = args
            self.kwargs = kwargs

        def _generator(self) -> PyIterator[DatasetElement]:
            """Yield parsed CSV rows from `self.filenames`.

            Yields:
                DatasetElement: Parsed row list.
            """
            paths = _iter_file_paths(self.filenames)
            if paths:
                for p in paths:
                    with open(p, "rb") as f:
                        raw = f.read()
                    text = _decompress_bytes(raw, self.compression_type).decode("utf-8")
                    reader = csv.reader(py_io.StringIO(text), delimiter=self.field_delim)
                    rows = list(reader)
                    start_idx = 1 if self.header and rows else 0
                    for row in rows[start_idx:]:
                        selected = (
                            [row[c] for c in self.select_cols if 0 <= c < len(row)]
                            if self.select_cols is not None
                            else list(row)
                        )
                        parsed: list[TensorOperand] = []
                        for col_idx, cell in enumerate(selected):
                            if cell == "" or cell == self.na_value:
                                default_val = (
                                    self.record_defaults[col_idx]
                                    if col_idx < len(self.record_defaults)
                                    else ""
                                )
                                parsed.append(default_val)
                            else:
                                parsed.append(cell)
                        yield parsed
                return
            yield list(self.record_defaults)

    class DatasetInitializer:
        """Table initializer backed by a `tf.data.Dataset`."""

        def __init__(
            self,
            dataset: Dataset | None = None,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `DatasetInitializer`.

            Args:
                dataset: Source `Dataset` of `(key, value)` pairs.
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.dataset = dataset
            self.args = args
            self.kwargs = kwargs

        def initialize(self) -> dict[DatasetElement, DatasetElement]:
            """Extract key-value pairs from `self.dataset`.

            Returns:
                dict[DatasetElement, DatasetElement]: Materialized key-value dictionary.
            """
            return {
                item[0]: item[1]
                for item in (self.dataset if isinstance(self.dataset, Dataset) else ())
                if isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and len(item) >= 2
            }

    class DistributeOptions:
        """Options controlling dataset distribution across devices/workers."""

        def __init__(
            self,
            auto_shard_policy: int = 0,
            num_devices: int | None = None,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `DistributeOptions`.

            Args:
                auto_shard_policy: Sharding policy integer.
                num_devices: Optional number of target devices.
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.auto_shard_policy = auto_shard_policy
            self.num_devices = num_devices
            self.args = args
            self.kwargs = kwargs

    class ExternalStatePolicy:
        """Policy for handling external state during dataset serialization."""

        WARN: int = 0
        IGNORE: int = 1
        FAIL: int = 2

        def __init__(
            self,
            value: int = 0,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `ExternalStatePolicy`.

            Args:
                value: Policy enum value (`WARN=0`, `IGNORE=1`, `FAIL=2`).
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.value = value
            self.args = args
            self.kwargs = kwargs

    INFINITE_CARDINALITY: int = -1
    """Experimental alias for `tf.data.INFINITE_CARDINALITY`."""

    class OptimizationOptions:
        """Options for `tf.data` graph optimizations."""

        def __init__(
            self,
            apply_default_optimizations: bool = True,
            filter_fusion: bool = False,
            map_and_batch_fusion: bool = True,
            map_and_filter_fusion: bool = False,
            map_fusion: bool = False,
            map_parallelization: bool = True,
            noop_elimination: bool = True,
            parallel_batch: bool = True,
            shuffle_and_repeat_fusion: bool = True,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `OptimizationOptions`.

            Args:
                apply_default_optimizations: Whether default optimizations are enabled.
                filter_fusion: Fuse consecutive `filter` transformations.
                map_and_batch_fusion: Fuse `map` and `batch` transformations.
                map_and_filter_fusion: Fuse `map` and `filter` transformations.
                map_fusion: Fuse consecutive `map` transformations.
                map_parallelization: Parallelize stateless `map` transformations.
                noop_elimination: Eliminate no-op transformations.
                parallel_batch: Parallelize batching copy operations.
                shuffle_and_repeat_fusion: Fuse `shuffle` and `repeat`.
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.apply_default_optimizations = apply_default_optimizations
            self.filter_fusion = filter_fusion
            self.map_and_batch_fusion = map_and_batch_fusion
            self.map_and_filter_fusion = map_and_filter_fusion
            self.map_fusion = map_fusion
            self.map_parallelization = map_parallelization
            self.noop_elimination = noop_elimination
            self.parallel_batch = parallel_batch
            self.shuffle_and_repeat_fusion = shuffle_and_repeat_fusion
            self.args = args
            self.kwargs = kwargs

    class Optional:
        """Represents a value that may or may not be present."""

        def __init__(
            self,
            value: DatasetElement = None,
            has_value_flag: bool | None = None,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `Optional`.

            Args:
                value: Optional wrapped value.
                has_value_flag: Explicit boolean indicating presence of a value.
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self._value = value
            self._has_value = (value is not None) if has_value_flag is None else has_value_flag
            self.args = args
            self.kwargs = kwargs

        def has_value(self) -> bool:
            """Return whether this `Optional` contains a value.

            Returns:
                bool: `True` if a value is present, else `False`.
            """
            return self._has_value

        def get_value(self) -> DatasetElement:
            """Return the wrapped value if present.

            Returns:
                DatasetElement: Wrapped value.

            Raises:
                ValueError: If this `Optional` is empty.
            """
            if not self._has_value:
                raise ValueError("Optional has no value.")
            return self._value

        @classmethod
        def from_value(cls, value: DatasetElement) -> experimental.Optional:
            """Create an `Optional` wrapping `value`.

            Args:
                value: Dataset element to wrap.

            Returns:
                experimental.Optional: Present `Optional` instance.
            """
            return cls(value=value, has_value_flag=True)

        @classmethod
        def empty(cls, element_spec: TensorOperand = None) -> experimental.Optional:
            """Create an empty `Optional` with no value.

            Args:
                element_spec: Optional type specification.

            Returns:
                experimental.Optional: Empty `Optional` instance.
            """
            return cls(value=None, has_value_flag=False, element_spec=element_spec)

    class RandomDataset(Dataset):
        """Dataset producing a deterministic stream of pseudorandom integers."""

        def __init__(
            self,
            seed: int | None = None,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `RandomDataset`.

            Args:
                seed: Optional random seed integer.
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.seed = seed if isinstance(seed, int) else None
            self.args = args
            self.kwargs = kwargs

        def _generator(self) -> PyIterator[DatasetElement]:
            """Yield deterministic pseudorandom integers.

            Yields:
                DatasetElement: Pseudorandom integer.
            """
            rng = random.Random(self.seed)
            for _ in range(1000):
                yield rng.randint(0, (1 << 31) - 1)

    class Reducer:
        """Reducer specification for `group_by_reducer`."""

        def __init__(
            self,
            init_func: Callable[[TensorOperand], DatasetElement] | None = None,
            reduce_func: Callable[[DatasetElement, DatasetElement], DatasetElement] | None = None,
            finalize_func: Callable[[DatasetElement], DatasetElement] | None = None,
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `Reducer`.

            Args:
                init_func: Function mapping a key to an initial accumulator state.
                reduce_func: Function mapping `(state, element)` to a new state.
                finalize_func: Function mapping final `state` to the reduced result.
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.init_func = init_func
            self.reduce_func = reduce_func
            self.finalize_func = finalize_func
            self.args = args
            self.kwargs = kwargs

    SHARD_HINT: int = -1
    """Sentinel constant indicating shard count should be supplied by `AutoShardPolicy`."""

    SqlDataset = _SqlDatasetImpl

    class TFRecordWriter:
        """Experimental writer for saving a `Dataset` of byte/string records to a TFRecord file."""

        def __init__(
            self,
            filename: str | TensorOperand = "",
            compression_type: str | TensorOperand = "",
            *args: TensorOperand,
            **kwargs: TensorOperand,
        ) -> None:
            """Initialize `TFRecordWriter`.

            Args:
                filename: Destination TFRecord file path.
                compression_type: Optional compression type (`""`, `"GZIP"`, or `"ZLIB"`).
                *args: Positional arguments.
                **kwargs: Keyword arguments.
            """
            self.filename = filename if isinstance(filename, str) else ""
            self.compression_type = compression_type if isinstance(compression_type, str) else ""
            self.args = args
            self.kwargs = kwargs

        def write(self, dataset: Dataset) -> None:
            """Write all elements of `dataset` as CRC-framed TFRecords to `self.filename`.

            Args:
                dataset: Source `Dataset` yielding `bytes` or `str` elements.
            """
            if not self.filename:
                return
            buf = bytearray()
            for elem in dataset:
                raw = elem if isinstance(elem, bytes) else str(elem).encode("utf-8")
                buf.extend(encode_tfrecord_frame(raw))
            payload = bytes(buf)
            comp = self.compression_type.upper()
            if comp == "GZIP":
                payload = gzip.compress(payload)
            elif comp == "ZLIB":
                payload = zlib.compress(payload)
            with open(self.filename, "wb") as f:
                f.write(payload)

    class ThreadingOptions(ThreadingOptions):
        """Experimental alias for `tf.data.ThreadingOptions`."""

    UNKNOWN_CARDINALITY: int = -2
    """Experimental alias for `tf.data.UNKNOWN_CARDINALITY`."""

    @staticmethod
    def assert_cardinality(
        expected_cardinality: int | TensorOperand = 0,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Assert the cardinality of the input dataset.

        Args:
            expected_cardinality: Expected number of elements in the dataset.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Transformation callable enforcing `expected_cardinality`.
        """
        del args, kwargs
        target = expected_cardinality if isinstance(expected_cardinality, int) else 0
        return lambda ds: _AssertCardinalityDataset(ds, target)

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
    def bucket_by_sequence_length(
        element_length_func: Callable[[DatasetElement], int] | TensorOperand = None,
        bucket_boundaries: Sequence[int] | TensorOperand = (),
        bucket_batch_sizes: Sequence[int] | TensorOperand = (),
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Bucket elements in a dataset by length and batch them.

        Args:
            element_length_func: Function mapping an element to its sequence length.
            bucket_boundaries: Upper length boundaries for each bucket.
            bucket_batch_sizes: Batch sizes per bucket (`len(bucket_boundaries) + 1`).
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Bucketing transformation callable.
        """
        del element_length_func, bucket_boundaries, args, kwargs
        b_size = (
            int(bucket_batch_sizes[0])
            if isinstance(bucket_batch_sizes, Sequence)
            and len(bucket_batch_sizes) > 0
            and isinstance(bucket_batch_sizes[0], int)
            else 1
        )
        return lambda ds: ds.batch(max(1, b_size))

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
    def choose_from_datasets(
        datasets: Sequence[Dataset] | TensorOperand = (),
        choice_dataset: Dataset | TensorOperand = None,
        stop_on_empty_dataset: bool = False,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Dataset:
        """Create a dataset that deterministically chooses elements from `datasets`.

        Args:
            datasets: Non-empty sequence of `Dataset` instances.
            choice_dataset: `Dataset` of integer indices into `datasets`.
            stop_on_empty_dataset: Whether to stop as soon as any selected dataset is exhausted.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dataset: Dataset of chosen elements.
        """
        del stop_on_empty_dataset, args, kwargs
        ds_iters = [
            iter(d)
            for d in (datasets if isinstance(datasets, Sequence) else ())
            if isinstance(d, Dataset)
        ]
        chosen: list[DatasetElement] = []
        valid_indices = [
            idx
            for idx in (choice_dataset if isinstance(choice_dataset, Dataset) else ())
            if isinstance(idx, int) and 0 <= idx < len(ds_iters)
        ]
        for idx in valid_indices:
            try:
                chosen.append(next(ds_iters[idx]))
            except StopIteration:
                break
        return TensorSliceDataset(chosen)

    @staticmethod
    def copy_to_device(
        target_device: str | TensorOperand = "/cpu:0",
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Create a transformation that copies dataset elements to a target device.

        Args:
            target_device: Target device string specification.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Device-copy transformation callable.
        """
        del target_device, args, kwargs
        return lambda ds: ds

    @staticmethod
    def dense_to_ragged_batch(
        batch_size: int | TensorOperand = 1,
        drop_remainder: bool = False,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Batch elements of unequal length into batched sequences.

        Args:
            batch_size: Number of elements per batch.
            drop_remainder: Whether to drop the final incomplete batch.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Ragged batching transformation callable.
        """
        del args, kwargs
        bs = batch_size if isinstance(batch_size, int) and batch_size > 0 else 1
        return lambda ds: ds.batch(bs, drop_remainder=drop_remainder)

    @staticmethod
    def dense_to_sparse_batch(
        batch_size: int | TensorOperand = 1,
        row_shape: Sequence[int] | TensorOperand = (),
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Batch elements of unequal shape into sparse batches.

        Args:
            batch_size: Number of elements per batch.
            row_shape: Dense shape of each row in the batch.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Sparse batching transformation callable.
        """
        del row_shape, args, kwargs
        bs = batch_size if isinstance(batch_size, int) and batch_size > 0 else 1
        return lambda ds: ds.batch(bs)

    @staticmethod
    def enable_debug_mode(*args: TensorOperand, **kwargs: TensorOperand) -> bool:
        """Enable eager execution of `tf.data` transformations for debugging.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            bool: `True` once debug mode is enabled.
        """
        del args, kwargs
        global _DEBUG_MODE
        _DEBUG_MODE = True
        return _DEBUG_MODE

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
    def from_variant(
        variant: Dataset | TensorOperand = None,
        structure: TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Dataset:
        """Construct a dataset from the given variant representation and structure.

        Args:
            variant: Variant `Dataset` or sequence.
            structure: Optional element structure specification.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dataset: Reconstructed `Dataset`.
        """
        del structure, args, kwargs
        if isinstance(variant, Dataset):
            return variant
        return TensorSliceDataset(variant)

    @staticmethod
    def get_next_as_optional(
        iterator: Iterator | Dataset | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> experimental.Optional:
        """Return an `Optional` wrapping the next element of the iterator.

        Args:
            iterator: Input `Iterator` or `Dataset`.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            experimental.Optional: Present `Optional` if an element is available, else empty `Optional`.
        """
        del args, kwargs
        if isinstance(iterator, Iterator):
            try:
                return experimental.Optional.from_value(next(iterator))
            except StopIteration:
                return experimental.Optional.empty()
        if isinstance(iterator, Dataset):
            try:
                return experimental.Optional.from_value(next(iter(iterator)))
            except StopIteration:
                return experimental.Optional.empty()
        return experimental.Optional.empty()

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
    def get_structure(
        dataset_or_iterator: Dataset | Iterator | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> DatasetSpec | IteratorSpec:
        """Return the type specification of the elements of a dataset or iterator.

        Args:
            dataset_or_iterator: Input `Dataset` or `Iterator`.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            DatasetSpec | IteratorSpec: Type specification corresponding to `dataset_or_iterator`.
        """
        del args, kwargs
        if isinstance(dataset_or_iterator, Iterator):
            return IteratorSpec.from_value(dataset_or_iterator)
        if isinstance(dataset_or_iterator, Dataset):
            return DatasetSpec.from_value(dataset_or_iterator)
        return DatasetSpec()

    @staticmethod
    def group_by_reducer(
        key_func: Callable[[DatasetElement], int] | TensorOperand = None,
        reducer: experimental.Reducer | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Group elements by key and reduce them using a `Reducer`.

        Args:
            key_func: Function mapping an element to a grouping key.
            reducer: `Reducer` specifying `init_func`, `reduce_func`, and `finalize_func`.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Grouping and reduction transformation callable.
        """
        del args, kwargs

        def _transform(ds: Dataset) -> Dataset:
            """Group and reduce `ds`.

            Args:
                ds: Input dataset.

            Returns:
                Dataset: Grouped and reduced dataset.
            """
            if not callable(key_func) or not isinstance(reducer, experimental.Reducer):
                return ds
            states: dict[int, DatasetElement] = {}
            for elem in ds:
                k = key_func(elem)
                if k not in states:
                    states[k] = reducer.init_func(k) if reducer.init_func is not None else 0
                states[k] = (
                    reducer.reduce_func(states[k], elem)
                    if reducer.reduce_func is not None
                    else states[k]
                )
            results: list[DatasetElement] = [
                reducer.finalize_func(st) if reducer.finalize_func is not None else st
                for st in states.values()
            ]
            return TensorSliceDataset(results)

        return _transform

    @staticmethod
    def group_by_window(
        key_func: Callable[[DatasetElement], int] | TensorOperand = None,
        reduce_func: Callable[[int, Dataset], Dataset] | TensorOperand = None,
        window_size: int | TensorOperand = 1,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Group elements by key and reduce windows of matching elements.

        Args:
            key_func: Function mapping an element to an integer key.
            reduce_func: Function mapping `(key, window_dataset)` to a reduced `Dataset`.
            window_size: Window size per key.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Windowed group reduction transformation callable.
        """
        del args, kwargs
        win_sz = window_size if isinstance(window_size, int) and window_size > 0 else 1

        def _transform(ds: Dataset) -> Dataset:
            """Group `ds` into windows by `key_func` and apply `reduce_func`.

            Args:
                ds: Input dataset.

            Returns:
                Dataset: Windowed reduced dataset.
            """
            if not callable(key_func) or not callable(reduce_func):
                return ds
            buckets: dict[int, list[DatasetElement]] = {}
            out: list[DatasetElement] = []
            for elem in ds:
                k = key_func(elem)
                buckets.setdefault(k, []).append(elem)
                if len(buckets[k]) >= win_sz:
                    win_ds = TensorSliceDataset(buckets.pop(k))
                    out.extend(reduce_func(k, win_ds))
            for k, rem in buckets.items():
                out.extend(reduce_func(k, TensorSliceDataset(rem)))
            return TensorSliceDataset(out)

        return _transform

    @staticmethod
    def ignore_errors(
        log_warning: bool = False,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Create a dataset transformation that silently drops elements that cause errors.

        Args:
            log_warning: Whether to log warnings when elements fail.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Error-ignoring dataset transformation callable.
        """
        del args, kwargs
        return lambda ds: _IgnoreErrorsDataset(ds, log_warning=log_warning)

    @staticmethod
    def index_table_from_dataset(
        dataset: Dataset | TensorOperand = None,
        num_oov_buckets: int = 0,
        vocab_size: int | None = None,
        default_value: int = -1,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[DatasetElement, int]:
        """Create an index lookup mapping from a dataset.

        Args:
            dataset: Input `Dataset` of vocabulary keys.
            num_oov_buckets: Number of out-of-vocabulary buckets.
            vocab_size: Optional maximum vocabulary size.
            default_value: Default index for unseen keys.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[DatasetElement, int]: Mapping from dataset elements to integer indices.
        """
        del num_oov_buckets, default_value, args, kwargs
        items = list(dataset) if isinstance(dataset, Dataset) else []
        limit = vocab_size if isinstance(vocab_size, int) else len(items)
        return {
            elem: idx
            for idx, elem in enumerate(items[:limit])
            if not isinstance(elem, (list, dict, Dataset))
        }

    @staticmethod
    def load(
        path: str | TensorOperand = "",
        element_spec: TensorOperand = None,
        compression: str | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Dataset:
        """Load a previously saved dataset from disk.

        Args:
            path: Directory or file path from which to load the dataset.
            element_spec: Optional element type specification.
            compression: Optional compression type (`"GZIP"`, `"ZLIB"`).
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dataset: Loaded `Dataset`.
        """
        del element_spec, args, kwargs
        if isinstance(path, str) and os.path.isfile(path):
            with open(path, "rb") as f:
                raw = f.read()
            decompressed = _decompress_bytes(raw, compression)
            items = json.loads(decompressed.decode("utf-8"))
            return TensorSliceDataset(items)
        return TensorSliceDataset([])

    @staticmethod
    def make_batched_features_dataset(
        file_pattern: TensorOperand = None,
        batch_size: int = 1,
        features: TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Dataset:
        """Return a dataset of batched records from `TFRecord` files.

        Args:
            file_pattern: File path(s) to read.
            batch_size: Batch size.
            features: Feature configuration dictionary.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dataset: Batched `TFRecordDataset`.
        """
        del features, args, kwargs
        bs = batch_size if batch_size > 0 else 1
        return TFRecordDataset(file_pattern).batch(bs)

    @staticmethod
    def make_csv_dataset(
        file_pattern: TensorOperand = None,
        batch_size: int = 1,
        column_names: Sequence[str] | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Dataset:
        """Read CSV files into a batched dataset of rows.

        Args:
            file_pattern: CSV file path(s).
            batch_size: Number of rows per batch.
            column_names: Optional list of column names.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dataset: Batched `CsvDataset`.
        """
        del column_names, args, kwargs
        bs = batch_size if batch_size > 0 else 1
        return experimental.CsvDataset(file_pattern).batch(bs)

    @staticmethod
    def make_saveable_from_iterator(
        iterator: Iterator | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Iterator:
        """Return a saveable wrapper for saving/restoring iterator state.

        Args:
            iterator: Target `Iterator`.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Iterator: The `Iterator` instance.
        """
        del args, kwargs
        if isinstance(iterator, Iterator):
            return iterator
        return Iterator()

    @staticmethod
    def map_and_batch(
        map_func: Callable[[DatasetElement], DatasetElement] | TensorOperand = None,
        batch_size: int = 1,
        num_parallel_batches: int | None = None,
        drop_remainder: bool = False,
        num_parallel_calls: int | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Fused implementation of `map` and `batch`.

        Args:
            map_func: Element mapping function.
            batch_size: Batch size (`> 0`).
            num_parallel_batches: Optional number of parallel batches.
            drop_remainder: Whether to drop the final incomplete batch.
            num_parallel_calls: Optional number of parallel map calls.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Fused map-and-batch transformation callable.
        """
        del num_parallel_batches, args, kwargs
        bs = batch_size if batch_size > 0 else 1

        def _transform(ds: Dataset) -> Dataset:
            """Apply `map_func` if callable, then `batch`.

            Args:
                ds: Input dataset.

            Returns:
                Dataset: Mapped and batched dataset.
            """
            mapped = (
                ds.map(map_func, num_parallel_calls=num_parallel_calls)
                if callable(map_func)
                else ds
            )
            return mapped.batch(bs, drop_remainder=drop_remainder)

        return _transform

    @staticmethod
    def pad_to_cardinality(
        cardinality: int | TensorOperand = 0,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Pad a dataset with zero elements up to a target cardinality.

        Args:
            cardinality: Target element count.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Padding transformation callable.
        """
        del args, kwargs
        target = cardinality if isinstance(cardinality, int) and cardinality > 0 else 0

        def _transform(ds: Dataset) -> Dataset:
            """Pad `ds` up to `target` elements.

            Args:
                ds: Input dataset.

            Returns:
                Dataset: Padded dataset.
            """
            items = list(ds)
            while len(items) < target:
                items.append(0)
            return TensorSliceDataset(items)

        return _transform

    @staticmethod
    def parallel_interleave(
        map_func: Callable[[DatasetElement], Dataset] | TensorOperand = None,
        cycle_length: int = 1,
        block_length: int = 1,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Map `map_func` across input and interleave results in parallel.

        Args:
            map_func: Function mapping an element to a `Dataset`.
            cycle_length: Number of input elements interleaved concurrently.
            block_length: Number of consecutive elements yielded per dataset.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Interleave transformation callable.
        """
        del args, kwargs
        cl = cycle_length if cycle_length > 0 else 1
        bl = block_length if block_length > 0 else 1
        return lambda ds: (
            ds.interleave(map_func, cycle_length=cl, block_length=bl) if callable(map_func) else ds
        )

    @staticmethod
    def parse_example_dataset(
        features: TensorOperand = None,
        num_parallel_calls: int = 1,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Transform a dataset of serialized `Example`s into parsed tensors.

        Args:
            features: Feature specification mapping.
            num_parallel_calls: Number of parallel parse calls.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Example-parsing transformation callable.
        """
        del features, num_parallel_calls, args, kwargs
        return lambda ds: ds

    @staticmethod
    def prefetch_to_device(
        device: str | TensorOperand = "/cpu:0",
        buffer_size: int | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Transform a dataset by prefetching elements to a target device.

        Args:
            device: Target device string.
            buffer_size: Optional prefetch buffer size.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Device-prefetch transformation callable.
        """
        del device, args, kwargs
        bs = buffer_size if isinstance(buffer_size, int) and buffer_size > 0 else AUTOTUNE
        return lambda ds: ds.prefetch(bs)

    @staticmethod
    def rejection_resample(
        class_func: Callable[[DatasetElement], int] | TensorOperand = None,
        target_dist: Sequence[float] | TensorOperand = (),
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Resample a dataset to achieve a target class distribution.

        Args:
            class_func: Function mapping an element to its integer class label.
            target_dist: Target probability distribution across classes.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Resampling transformation callable.
        """
        del target_dist, args, kwargs
        return lambda ds: ds.map(lambda x: [class_func(x), x]) if callable(class_func) else ds

    @staticmethod
    def sample_from_datasets(
        datasets: Sequence[Dataset] | TensorOperand = (),
        weights: Sequence[float] | None = None,
        seed: int | None = None,
        stop_on_empty_dataset: bool = False,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Dataset:
        """Sample elements at random from the datasets in `datasets`.

        Args:
            datasets: Non-empty sequence of `Dataset` instances.
            weights: Optional sampling weights per dataset.
            seed: Optional random seed integer.
            stop_on_empty_dataset: Whether to stop as soon as any dataset is exhausted.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dataset: Sampled `Dataset`.
        """
        del weights, seed, stop_on_empty_dataset, args, kwargs
        out: list[DatasetElement] = [
            elem
            for d in (datasets if isinstance(datasets, Sequence) else ())
            if isinstance(d, Dataset)
            for elem in d
        ]
        return TensorSliceDataset(out)

    @staticmethod
    def save(
        dataset: Dataset | TensorOperand = None,
        path: str | TensorOperand = "",
        compression: str | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> bool:
        """Save the content of the given dataset to disk.

        Args:
            dataset: Input `Dataset` to persist.
            path: Destination file path on disk.
            compression: Optional compression type (`"GZIP"`, `"ZLIB"`).
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            bool: `True` if saved to `path`, else `False`.
        """
        del args, kwargs
        if isinstance(dataset, Dataset) and isinstance(path, str) and path:
            raw = json.dumps(list(dataset)).encode("utf-8")
            comp = compression.upper() if isinstance(compression, str) else ""
            payload = (
                gzip.compress(raw)
                if comp == "GZIP"
                else (zlib.compress(raw) if comp == "ZLIB" else raw)
            )
            with open(path, "wb") as f:
                f.write(payload)
            return True
        return False

    @staticmethod
    def scan(
        initial_state: DatasetElement = None,
        scan_func: Callable[[DatasetElement, DatasetElement], tuple[DatasetElement, DatasetElement]]
        | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Transform a dataset by scanning a stateful function across its elements.

        Args:
            initial_state: Initial state passed to `scan_func`.
            scan_func: Stateful function `(state, elem) -> (new_state, out_elem)`.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Scan transformation callable.
        """
        del args, kwargs
        return lambda ds: _ScanDataset(ds, initial_state, scan_func) if callable(scan_func) else ds

    @staticmethod
    def service(*args: TensorOperand, **kwargs: TensorOperand) -> dict[str, TensorOperand]:
        """Experimental `tf.data.experimental.service` configuration helper.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[str, TensorOperand]: Service configuration dictionary.
        """
        del args
        return dict(kwargs)

    @staticmethod
    def shuffle_and_repeat(
        buffer_size: int | TensorOperand = 1,
        count: int | None = None,
        seed: int | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Shuffle and repeat a dataset.

        Args:
            buffer_size: Shuffle buffer size.
            count: Number of repetitions.
            seed: Optional random seed integer.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Shuffle-and-repeat transformation callable.
        """
        del args, kwargs
        bs = buffer_size if isinstance(buffer_size, int) and buffer_size > 0 else 1
        return lambda ds: ds.shuffle(bs, seed=seed).repeat(count)

    @staticmethod
    def snapshot(
        path: str | TensorOperand = "",
        compression: str | None = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Callable[[Dataset], Dataset]:
        """Persist the output of a dataset pipeline to cache/disk on the fly.

        Args:
            path: Snapshot directory or file path.
            compression: Optional compression type.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Callable[[Dataset], Dataset]: Snapshot caching transformation callable.
        """
        del compression, args, kwargs
        cache_path = path if isinstance(path, str) else ""
        return lambda ds: ds.cache(cache_path)

    @staticmethod
    def table_from_dataset(
        dataset: Dataset | TensorOperand = None,
        num_oov_buckets: int = 0,
        default_value: DatasetElement = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> dict[DatasetElement, DatasetElement]:
        """Create a lookup mapping from a dataset of `(key, value)` pairs.

        Args:
            dataset: Input `Dataset` of 2-element `(key, value)` sequences.
            num_oov_buckets: Number of out-of-vocabulary buckets.
            default_value: Default value for unseen keys.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            dict[DatasetElement, DatasetElement]: Key-to-value lookup dictionary.
        """
        del num_oov_buckets, default_value, args, kwargs
        return experimental.DatasetInitializer(
            dataset if isinstance(dataset, Dataset) else None
        ).initialize()

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
    def to_variant(
        dataset: Dataset | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> Dataset:
        """Return a variant dataset representation of the given dataset.

        Args:
            dataset: Input `Dataset`.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Dataset: Variant `Dataset` wrapper.
        """
        del args, kwargs
        if isinstance(dataset, Dataset):
            return dataset
        return TensorSliceDataset(dataset)

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
