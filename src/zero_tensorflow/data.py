"""TensorFlow data module."""

from __future__ import annotations

import queue
import random
import threading
from collections.abc import Iterator as PyIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

__all__ = [
    "ArrayIterator",
    "Dataset",
    "DatasetSpec",
    "FixedLengthRecordDataset",
    "Iterator",
    "IteratorSpec",
    "Options",
    "TFRecordDataset",
    "TextLineDataset",
    "ThreadingOptions",
]


class ThreadingOptions:
    """
    Threading options for dataset operations.

    Args:
        max_intra_op_parallelism: Maximum parallelism.
        private_threadpool_size: Threadpool size.
    """

    def __init__(
        self,
        max_intra_op_parallelism: int | None = None,
        private_threadpool_size: int | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.max_intra_op_parallelism = max_intra_op_parallelism
        self.private_threadpool_size = private_threadpool_size


class Options:
    """
    Options for dataset operations.

    Args:
        autotune: Whether to autotune.
        threading: Threading options.
    """

    def __init__(
        self,
        autotune: bool | None = None,
        threading: ThreadingOptions | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.autotune = autotune
        self.threading = threading if threading is not None else ThreadingOptions()
        for k, v in kwargs.items():
            setattr(self, k, v)


class DatasetSpec:
    """
    Specification for a dataset.

    Args:
        element_spec: Spec for elements.
        dataset_shape: Shape of the dataset.
    """

    __slots__ = ["_dataset_shape", "_element_spec"]

    def __init__(
        self,
        element_spec: Any = None,
        dataset_shape: Any = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._element_spec = element_spec
        self._dataset_shape = dataset_shape


class IteratorSpec:
    """
    Specification for an iterator.

    Args:
        element_spec: Spec for elements.
    """

    __slots__ = ["_element_spec"]

    def __init__(self, element_spec: Any = None, *args: Any, **kwargs: Any):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._element_spec = element_spec


class Iterator:
    """
    Iterator over a dataset.

    Args:
        dataset: The dataset to iterate over.
    """

    def __init__(self, dataset: Dataset, *args: Any, **kwargs: Any):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._dataset = dataset
        self._iter = iter(dataset)

    def __next__(self) -> Any:
        """
        Get the next element.

        Returns:
            Any: The next element.
        """
        return next(self._iter)

    def __iter__(self) -> Iterator:
        """
        Get the iterator.

        Returns:
            Iterator: The iterator.
        """
        return self


class ArrayIterator(Iterator):
    """
    Iterator over a dataset returning arrays.

    Args:
        dataset: The dataset to iterate over.
    """


class Dataset:
    _compiler_node: Any = None
    """Base Dataset object supporting true generator protocol."""

    def __iter__(self) -> PyIterator[Any]:
        """
        Get the iterator.

        Returns:
            PyIterator[Any]: The iterator.
        """
        return self._generator()

    def _generator(self) -> PyIterator[Any]:
        """
        Yield elements from the dataset.

        Returns:
            PyIterator[Any]: Elements.
        """

        if hasattr(self, "_compiler_node"):
            iterator = iter(self._compiler_node)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    break
        return

    @classmethod
    def from_tensor_slices(cls, tensors: Any) -> Dataset:

        node = None
        ds = TensorSliceDataset(tensors)
        if node:
            ds._compiler_node = node
        return ds

    def map(
        self,
        map_func: Callable[[Any], Any],
        num_parallel_calls: int | None = None,
        deterministic: bool | None = None,
    ) -> Dataset:
        """
        Map a function over the dataset.

        Args:
            map_func (Callable): Function to apply.
            num_parallel_calls (Optional[int]): Number of parallel threads.
            deterministic (Optional[bool]): Whether order should be deterministic.

        Returns:
            Dataset: A new dataset with mapped elements.
        """

        node = None
        ds = MapDataset(self, map_func, num_parallel_calls, deterministic)
        if node:
            ds._compiler_node = node
        return ds

    def filter(self, predicate: Callable[[Any], bool]) -> Dataset:
        """
        Filter dataset elements based on a predicate.

        Args:
            predicate (Callable): A function returning a boolean.

        Returns:
            Dataset: The filtered dataset.
        """
        return FilterDataset(self, predicate)

    def batch(self, batch_size: int, drop_remainder: bool = False) -> Dataset:
        """
        Batch the dataset elements.

        Args:
            batch_size (int): Size of the batches.
            drop_remainder (bool): Drop the last batch if it has fewer elements.

        Returns:
            Dataset: A new batched dataset.
        """

        node = None
        ds = BatchDataset(self, batch_size, drop_remainder)
        if node:
            ds._compiler_node = node
        return ds

    def shuffle(
        self,
        buffer_size: int,
        seed: int | None = None,
        reshuffle_each_iteration: bool | None = None,
    ) -> Dataset:
        """
        Shuffle the dataset.

        Args:
            buffer_size (int): Number of elements from which to shuffle.
            seed (Optional[int]): Random seed.
            reshuffle_each_iteration (Optional[bool]): Reshuffle on each epoch.

        Returns:
            Dataset: The shuffled dataset.
        """

        node = None
        ds = ShuffleDataset(self, buffer_size, seed, reshuffle_each_iteration)
        if node:
            ds._compiler_node = node
        return ds

    def prefetch(self, buffer_size: int) -> Dataset:
        """
        Prefetch elements from the dataset.

        Args:
            buffer_size (int): Max number of elements to prefetch.

        Returns:
            Dataset: The prefetched dataset.
        """

        node = None
        ds = PrefetchDataset(self, buffer_size)
        if node:
            ds._compiler_node = node
        return ds

    def cache(self, filename: str = "") -> Dataset:
        """
        Cache elements of the dataset.

        Args:
            filename (str): Optional file path.

        Returns:
            Dataset: The cached dataset.
        """
        return CacheDataset(self, filename)

    def interleave(
        self,
        map_func: Callable[[Any], Dataset],
        cycle_length: int | None = None,
        block_length: int = 1,
        num_parallel_calls: int | None = None,
        deterministic: bool | None = None,
    ) -> Dataset:
        """
        Interleave elements from multiple datasets.

        Args:
            map_func: Function to map an element to a Dataset.
            cycle_length: Number of input elements to process concurrently.
            block_length: Number of consecutive elements to produce from each input element.
            num_parallel_calls: Number of parallel thread calls.
            deterministic: Determinism control.

        Returns:
            Dataset: Interleaved dataset.
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
        """
        Create a dataset of windows.

        Args:
            size (int): Window size.
            shift (Optional[int]): Shift size.
            stride (int): Stride size.
            drop_remainder (bool): Drop partial windows.

        Returns:
            Dataset: Dataset of windows.
        """
        return WindowDataset(self, size, shift, stride, drop_remainder)


class TensorSliceDataset(Dataset):
    """Dataset created from tensor slices."""

    def __init__(self, elements: Any):
        """Initialize."""
        if isinstance(elements, str):
            self.elements = list(elements)
        else:
            self.elements = list(elements) if elements is not None else []

    def _generator(self) -> PyIterator[Any]:
        yield from self.elements


class MapDataset(Dataset):
    """Dataset created from mapping a function."""

    def __init__(
        self,
        input_dataset: Dataset,
        map_func: Callable,
        num_parallel_calls: int | None = None,
        deterministic: bool | None = None,
    ):
        """Initialize."""
        self._input_dataset = input_dataset
        self._map_func = map_func
        self._num_parallel_calls = num_parallel_calls
        self._deterministic = deterministic

    def _generator(self) -> PyIterator[Any]:
        if self._num_parallel_calls is None:
            for elem in self._input_dataset:
                yield self._map_func(elem)
        else:
            workers = (
                self._num_parallel_calls
                if isinstance(self._num_parallel_calls, int)
                and self._num_parallel_calls > 0
                else 4
            )
            with ThreadPoolExecutor(max_workers=workers) as executor:
                q: queue.Queue = queue.Queue(maxsize=workers * 2)

                def submitter() -> None:
                    try:
                        for elem in self._input_dataset:
                            f = executor.submit(self._map_func, elem)
                            q.put((True, f))
                    except Exception as e:  # noqa: BLE001
                        q.put((False, e))
                    finally:
                        q.put((None, None))

                t = threading.Thread(target=submitter, daemon=True)
                t.start()
                while True:
                    ok, item = q.get()
                    if ok is None:
                        break
                    if not ok:
                        raise item
                    yield item.result()


class FilterDataset(Dataset):
    """Dataset created by filtering."""

    def __init__(self, input_dataset: Dataset, predicate: Callable[[Any], bool]):
        """Initialize."""
        self._input_dataset = input_dataset
        self._predicate = predicate

    def _generator(self) -> PyIterator[Any]:
        for elem in self._input_dataset:
            if self._predicate(elem):
                yield elem


class BatchDataset(Dataset):
    """Dataset created by batching."""

    def __init__(
        self, input_dataset: Dataset, batch_size: int, drop_remainder: bool = False
    ):
        """Initialize."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self._input_dataset = input_dataset
        self._batch_size = batch_size
        self._drop_remainder = drop_remainder

    def _generator(self) -> PyIterator[Any]:
        batch = []
        for elem in self._input_dataset:
            batch.append(elem)
            if len(batch) == self._batch_size:
                yield batch
                batch = []
        if batch and not self._drop_remainder:
            yield batch


class ShuffleDataset(Dataset):
    """Dataset created by shuffling."""

    def __init__(
        self,
        input_dataset: Dataset,
        buffer_size: int,
        seed: int | None = None,
        reshuffle_each_iteration: bool | None = None,
    ):
        """Initialize."""
        self._input_dataset = input_dataset
        self._buffer_size = buffer_size
        self._seed = seed
        self._reshuffle_each_iteration = (
            reshuffle_each_iteration if reshuffle_each_iteration is not None else True
        )

    def _generator(self) -> PyIterator[Any]:
        rng = random.Random(self._seed)
        if self._reshuffle_each_iteration and self._seed is not None:
            self._seed += 1

        if self._buffer_size <= 0:
            for elem in self._input_dataset:
                yield elem
            return

        buffer = []
        for elem in self._input_dataset:
            buffer.append(elem)
            if len(buffer) >= self._buffer_size:
                idx = rng.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)

        while buffer:
            idx = rng.randint(0, len(buffer) - 1)
            yield buffer.pop(idx)


class PrefetchDataset(Dataset):
    """Dataset created by prefetching."""

    def __init__(self, input_dataset: Dataset, buffer_size: int):
        """Initialize."""
        self._input_dataset = input_dataset
        self._buffer_size = buffer_size

    def _generator(self) -> PyIterator[Any]:
        bs = (
            self._buffer_size
            if isinstance(self._buffer_size, int) and self._buffer_size > 0
            else 10
        )
        q: queue.Queue = queue.Queue(maxsize=bs)

        def worker() -> None:
            try:
                for elem in self._input_dataset:
                    q.put((True, elem))
            except Exception as e:  # noqa: BLE001
                q.put((False, e))
            finally:
                q.put((None, None))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            ok, elem = q.get()
            if ok is None:
                break
            if not ok:
                raise elem
            yield elem


class CacheDataset(Dataset):
    """Dataset created by caching."""

    def __init__(self, input_dataset: Dataset, filename: str = ""):
        """Initialize."""
        self._input_dataset = input_dataset
        self._filename = filename
        self._cache: list | None = None

    def _generator(self) -> PyIterator[Any]:
        if self._filename == "":
            if self._cache is None:
                self._cache = []
                for elem in self._input_dataset:
                    self._cache.append(elem)
                    yield elem
            else:
                for elem in self._cache:
                    yield elem
        else:
            for elem in self._input_dataset:
                yield elem


class InterleaveDataset(Dataset):
    """Dataset created by interleaving."""

    def __init__(
        self,
        input_dataset: Dataset,
        map_func: Callable,
        cycle_length: int | None = None,
        block_length: int = 1,
        num_parallel_calls: int | None = None,
        deterministic: bool | None = None,
    ):
        """Initialize."""
        self._input_dataset = input_dataset
        self._map_func = map_func
        self._cycle_length = cycle_length if cycle_length is not None else 1
        self._block_length = block_length
        self._num_parallel_calls = num_parallel_calls

    def _generator(self) -> PyIterator[Any]:
        iterators: list = []
        input_iter = iter(self._input_dataset)

        def fill_iterators() -> None:
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
    """Dataset created by windowing."""

    def __init__(
        self,
        input_dataset: Dataset,
        size: int,
        shift: int | None = None,
        stride: int = 1,
        drop_remainder: bool = False,
    ):
        """Initialize."""
        self._input_dataset = input_dataset
        self._size = size
        self._shift = shift if shift is not None else size
        self._stride = stride
        self._drop_remainder = drop_remainder

    def _generator(self) -> PyIterator[Any]:
        elements = list(self._input_dataset)
        i = 0
        while i < len(elements):
            w = elements[i : i + self._size * self._stride : self._stride]
            if len(w) == self._size or not self._drop_remainder and len(w) > 0:
                yield TensorSliceDataset(w)
            i += self._shift


class FixedLengthRecordDataset(Dataset):
    """
    Dataset for fixed length records.

    Args:
        filenames: Files to read.
    """

    def __init__(self, filenames: Any = None, *args: Any, **kwargs: Any):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.filenames = filenames
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _generator(self) -> PyIterator[Any]:
        yield b""  # mock implementation


class TFRecordDataset(Dataset):
    """
    Dataset for TFRecords.

    Args:
        filenames: Files to read.
    """

    def __init__(self, filenames: Any = None, *args: Any, **kwargs: Any):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.filenames = filenames
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _generator(self) -> PyIterator[Any]:
        yield b""  # mock implementation


class TextLineDataset(Dataset):
    """
    Dataset for text lines.

    Args:
        filenames: Files to read.
    """

    def __init__(self, filenames: Any = None, *args: Any, **kwargs: Any):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.filenames = filenames
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _generator(self) -> PyIterator[Any]:
        yield ""  # mock implementation


# Stubs from TODO_PLAN.md

AUTOTUNE: int = 0
"""Stub for AUTOTUNE."""

INFINITE_CARDINALITY: int = 0
"""Stub for INFINITE_CARDINALITY."""


class NumpyIterator:
    """Stub for NumpyIterator."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


UNKNOWN_CARDINALITY: int = 0
"""Stub for UNKNOWN_CARDINALITY."""


class experimental:
    """Stub for experimental module."""

    AUTOTUNE: int = 0
    """Stub for AUTOTUNE."""

    class AutoShardPolicy:
        """Stub for AutoShardPolicy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class AutotuneAlgorithm:
        """Stub for AutotuneAlgorithm."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class AutotuneOptions:
        """Stub for AutotuneOptions."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Counter:
        """Stub for Counter."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class CsvDataset:
        """Stub for CsvDataset."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class DatasetInitializer:
        """Stub for DatasetInitializer."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class DistributeOptions:
        """Stub for DistributeOptions."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class ExternalStatePolicy:
        """Stub for ExternalStatePolicy."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    INFINITE_CARDINALITY: int = 0
    """Stub for INFINITE_CARDINALITY."""

    class OptimizationOptions:
        """Stub for OptimizationOptions."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Optional:
        """Stub for Optional."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

    class RandomDataset:
        """Stub for RandomDataset."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

    class Reducer:
        """Stub for Reducer."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    SHARD_HINT: int = 0
    """Stub for SHARD_HINT."""

    class SqlDataset:
        """Stub for SqlDataset."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            return None

    class TFRecordWriter:
        """Stub for TFRecordWriter."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class ThreadingOptions:
        """Stub for ThreadingOptions."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    UNKNOWN_CARDINALITY: int = 0
    """Stub for UNKNOWN_CARDINALITY."""

    @staticmethod
    def assert_cardinality(*args: Any, **kwargs: Any) -> None:
        """Stub for assert_cardinality."""
        return

    @staticmethod
    def at(*args: Any, **kwargs: Any) -> None:
        """Stub for at."""
        return

    @staticmethod
    def bucket_by_sequence_length(*args: Any, **kwargs: Any) -> None:
        """Stub for bucket_by_sequence_length."""
        return

    @staticmethod
    def cardinality(*args: Any, **kwargs: Any) -> None:
        """Stub for cardinality."""
        return

    @staticmethod
    def choose_from_datasets(*args: Any, **kwargs: Any) -> None:
        """Stub for choose_from_datasets."""
        return

    @staticmethod
    def copy_to_device(*args: Any, **kwargs: Any) -> None:
        """Stub for copy_to_device."""
        return

    @staticmethod
    def dense_to_ragged_batch(*args: Any, **kwargs: Any) -> None:
        """Stub for dense_to_ragged_batch."""
        return

    @staticmethod
    def dense_to_sparse_batch(*args: Any, **kwargs: Any) -> None:
        """Stub for dense_to_sparse_batch."""
        return

    @staticmethod
    def enable_debug_mode(*args: Any, **kwargs: Any) -> None:
        """Stub for enable_debug_mode."""
        return

    @staticmethod
    def enumerate_dataset(*args: Any, **kwargs: Any) -> None:
        """Stub for enumerate_dataset."""
        return

    @staticmethod
    def from_list(*args: Any, **kwargs: Any) -> None:
        """Stub for from_list."""
        return

    @staticmethod
    def from_variant(*args: Any, **kwargs: Any) -> None:
        """Stub for from_variant."""
        return

    @staticmethod
    def get_next_as_optional(*args: Any, **kwargs: Any) -> None:
        """Stub for get_next_as_optional."""
        return

    @staticmethod
    def get_single_element(*args: Any, **kwargs: Any) -> None:
        """Stub for get_single_element."""
        return

    @staticmethod
    def get_structure(*args: Any, **kwargs: Any) -> None:
        """Stub for get_structure."""
        return

    @staticmethod
    def group_by_reducer(*args: Any, **kwargs: Any) -> None:
        """Stub for group_by_reducer."""
        return

    @staticmethod
    def group_by_window(*args: Any, **kwargs: Any) -> None:
        """Stub for group_by_window."""
        return

    @staticmethod
    def ignore_errors(*args: Any, **kwargs: Any) -> None:
        """Stub for ignore_errors."""
        return

    @staticmethod
    def index_table_from_dataset(*args: Any, **kwargs: Any) -> None:
        """Stub for index_table_from_dataset."""
        return

    @staticmethod
    def load(*args: Any, **kwargs: Any) -> None:
        """Stub for load."""
        return

    @staticmethod
    def make_batched_features_dataset(*args: Any, **kwargs: Any) -> None:
        """Stub for make_batched_features_dataset."""
        return

    @staticmethod
    def make_csv_dataset(*args: Any, **kwargs: Any) -> None:
        """Stub for make_csv_dataset."""
        return

    @staticmethod
    def make_saveable_from_iterator(*args: Any, **kwargs: Any) -> None:
        """Stub for make_saveable_from_iterator."""
        return

    @staticmethod
    def map_and_batch(*args: Any, **kwargs: Any) -> None:
        """Stub for map_and_batch."""
        return

    @staticmethod
    def pad_to_cardinality(*args: Any, **kwargs: Any) -> None:
        """Stub for pad_to_cardinality."""
        return

    @staticmethod
    def parallel_interleave(*args: Any, **kwargs: Any) -> None:
        """Stub for parallel_interleave."""
        return

    @staticmethod
    def parse_example_dataset(*args: Any, **kwargs: Any) -> None:
        """Stub for parse_example_dataset."""
        return

    @staticmethod
    def prefetch_to_device(*args: Any, **kwargs: Any) -> None:
        """Stub for prefetch_to_device."""
        return

    @staticmethod
    def rejection_resample(*args: Any, **kwargs: Any) -> None:
        """Stub for rejection_resample."""
        return

    @staticmethod
    def sample_from_datasets(*args: Any, **kwargs: Any) -> None:
        """Stub for sample_from_datasets."""
        return

    @staticmethod
    def save(*args: Any, **kwargs: Any) -> None:
        """Stub for save."""
        return

    @staticmethod
    def scan(*args: Any, **kwargs: Any) -> None:
        """Stub for scan."""
        return

    @staticmethod
    def service(*args: Any, **kwargs: Any) -> None:
        """Stub for service."""
        return

    @staticmethod
    def shuffle_and_repeat(*args: Any, **kwargs: Any) -> None:
        """Stub for shuffle_and_repeat."""
        return

    @staticmethod
    def snapshot(*args: Any, **kwargs: Any) -> None:
        """Stub for snapshot."""
        return

    @staticmethod
    def table_from_dataset(*args: Any, **kwargs: Any) -> None:
        """Stub for table_from_dataset."""
        return

    @staticmethod
    def take_while(*args: Any, **kwargs: Any) -> None:
        """Stub for take_while."""
        return

    @staticmethod
    def to_variant(*args: Any, **kwargs: Any) -> None:
        """Stub for to_variant."""
        return

    @staticmethod
    def unbatch(*args: Any, **kwargs: Any) -> None:
        """Stub for unbatch."""
        return

    @staticmethod
    def unique(*args: Any, **kwargs: Any) -> None:
        """Stub for unique."""
        return
