"""TensorFlow data module."""

from typing import (
    Any,
    Callable,
    Iterator as PyIterator,
)
import numpy as np

__all__ = [
    "Dataset",
    "DatasetSpec",
    "FixedLengthRecordDataset",
    "Iterator",
    "IteratorSpec",
    "NumpyIterator",
    "Options",
    "TFRecordDataset",
    "TextLineDataset",
    "ThreadingOptions",
]


class ThreadingOptions:
    """Represents options for dataset threading."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self.max_intra_op_parallelism = kwargs.get(
            "max_intra_op_parallelism", args[0] if len(args) > 0 else None
        )
        self.private_threadpool_size = kwargs.get(
            "private_threadpool_size", args[1] if len(args) > 1 else None
        )


class Options:
    """Represents options for `tf.data.Dataset`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self.autotune = kwargs.get("autotune")
        self.deterministic = kwargs.get("deterministic")
        self.experimental_deterministic = kwargs.get("experimental_deterministic")
        self.experimental_distribute = kwargs.get("experimental_distribute")
        self.experimental_external_state_policy = kwargs.get(
            "experimental_external_state_policy"
        )
        self.experimental_optimization = kwargs.get("experimental_optimization")
        self.experimental_slack = kwargs.get("experimental_slack")
        self.experimental_symbolic_checkpoint = kwargs.get(
            "experimental_symbolic_checkpoint"
        )
        self.experimental_service = kwargs.get("experimental_service")
        self.experimental_threading = kwargs.get("experimental_threading")
        self.experimental_warm_start = kwargs.get("experimental_warm_start")
        self.dataset_name = kwargs.get("dataset_name")
        self.framework_type = kwargs.get("framework_type")
        self.threading = (
            kwargs.get("threading")
            if kwargs.get("threading") is not None
            else ThreadingOptions()
        )


class DatasetSpec:
    """Type specification for `tf.data.Dataset`."""

    __slots__ = ["_element_spec", "_dataset_shape"]

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self._element_spec = kwargs.get("element_spec") or (args[0] if args else None)
        self._dataset_shape = kwargs.get(
            "dataset_shape", args[1] if len(args) > 1 else ()
        )


class IteratorSpec:
    """Type specification for `tf.data.Iterator`."""

    __slots__ = ["_element_spec"]

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self._element_spec = kwargs.get("element_spec") or (args[0] if args else None)


class Iterator:
    """Represents an iterator of a `tf.data.Dataset`."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        dataset = (
            kwargs.get("dataset") if "dataset" in kwargs else (args[0] if args else [])
        )
        self.dataset = dataset
        self._iter = iter(dataset)

    def __next__(self) -> Any:
        """__next__ docstring."""
        return next(self._iter)

    def __iter__(self) -> "Iterator":
        """__iter__ docstring."""
        return self


class NumpyIterator:
    """Iterator over a dataset with elements converted to numpy."""

    __slots__ = ["_iterator"]

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        dataset = (
            kwargs.get("dataset") if "dataset" in kwargs else (args[0] if args else [])
        )
        self._iterator = iter(dataset)

    def __next__(self) -> Any:
        """__next__ docstring."""
        return next(self._iterator)

    def __iter__(self) -> "NumpyIterator":
        """__iter__ docstring."""
        return self


class Dataset:
    """Represents a potentially large set of elements."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self._elements = (
            kwargs.get("elements")
            if "elements" in kwargs
            else (args[0] if args else [])
        )

    @classmethod
    def from_tensor_slices(cls, tensors: Any) -> "Dataset":
        """from_tensor_slices docstring."""
        if isinstance(tensors, list):
            return cls(elements=tensors)
        return cls(elements=list(tensors))

    def batch(self, batch_size: int) -> "Dataset":
        """batch docstring."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        batched = [
            self._elements[i : i + batch_size]
            for i in range(0, len(self._elements), batch_size)
        ]
        return Dataset(elements=batched)

    def map(self, map_func: Callable[[Any], Any]) -> "Dataset":
        """map docstring."""
        mapped = [map_func(e) for e in self._elements]
        return Dataset(elements=mapped)

    def shuffle(self, buffer_size: int) -> "Dataset":
        """shuffle docstring."""
        shuffled = list(self._elements)
        if buffer_size > 0:
            np.random.shuffle(shuffled)
        return Dataset(elements=shuffled)

    def __iter__(self) -> PyIterator[Any]:
        """__iter__ docstring."""
        return iter(self._elements)


class FixedLengthRecordDataset(Dataset):
    """A `Dataset` of fixed-length records from one or more binary files."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__()
        self.filenames = kwargs.get("filenames") or (args[0] if len(args) > 0 else None)
        self.record_bytes = kwargs.get("record_bytes") or (
            args[1] if len(args) > 1 else None
        )
        self.header_bytes = kwargs.get("header_bytes")
        self.footer_bytes = kwargs.get("footer_bytes")
        self.buffer_size = kwargs.get("buffer_size")
        self.compression_type = kwargs.get("compression_type")
        self.num_parallel_reads = kwargs.get("num_parallel_reads")
        self.name = kwargs.get("name")


class TFRecordDataset(Dataset):
    """A `Dataset` comprising records from one or more TFRecord files."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__()
        self.filenames = kwargs.get("filenames") or (args[0] if len(args) > 0 else None)
        self.compression_type = kwargs.get("compression_type")
        self.buffer_size = kwargs.get("buffer_size")
        self.num_parallel_reads = kwargs.get("num_parallel_reads")
        self.name = kwargs.get("name")


class TextLineDataset(Dataset):
    """Creates a `Dataset` comprising lines from one or more text files."""

    def __init__(self, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__()
        self.filenames = kwargs.get("filenames") or (args[0] if len(args) > 0 else None)
        self.compression_type = kwargs.get("compression_type")
        self.buffer_size = kwargs.get("buffer_size")
        self.num_parallel_reads = kwargs.get("num_parallel_reads")
        self.name = kwargs.get("name")
