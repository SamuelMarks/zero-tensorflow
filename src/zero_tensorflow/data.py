"TensorFlow data module."

from typing import Any, Callable, Iterator as PyIterator

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
    """ThreadingOptions docstring."""

    def __init__(
        self,
        max_intra_op_parallelism=None,
        private_threadpool_size=None,
        *args: Any,
        **kwargs: Any,
    ):
        """__init__ docstring."""
        self.max_intra_op_parallelism = max_intra_op_parallelism
        self.private_threadpool_size = private_threadpool_size


class Options:
    """Options docstring."""

    def __init__(self, autotune=None, threading=None, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self.autotune = autotune
        self.threading = threading if threading is not None else ThreadingOptions()
        for k, v in kwargs.items():
            setattr(self, k, v)


class DatasetSpec:
    """DatasetSpec docstring."""

    __slots__ = ["_element_spec", "_dataset_shape"]

    def __init__(
        self, element_spec=None, dataset_shape=None, *args: Any, **kwargs: Any
    ):
        """__init__ docstring."""
        self._element_spec = element_spec
        self._dataset_shape = dataset_shape


class IteratorSpec:
    """IteratorSpec docstring."""

    __slots__ = ["_element_spec"]

    def __init__(self, element_spec=None, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self._element_spec = element_spec


class Iterator:
    """Iterator docstring."""

    def __init__(self, dataset, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self._dataset = dataset
        self._iter = iter(dataset)

    def __next__(self) -> Any:
        """__next__ docstring."""
        return next(self._iter)

    def __iter__(self) -> "Iterator":
        """__iter__ docstring."""
        return self


class NumpyIterator:
    """NumpyIterator docstring."""

    __slots__ = ["_iterator"]

    def __init__(self, dataset, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self._iterator = iter(dataset)

    def __next__(self) -> Any:
        """__next__ docstring."""
        return next(self._iterator)

    def __iter__(self) -> "NumpyIterator":
        """__iter__ docstring."""
        return self


class Dataset:
    """Dataset docstring."""

    def __init__(self, elements, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        self.elements = list(elements) if elements is not None else []

    @classmethod
    def from_tensor_slices(cls, tensors: Any) -> "Dataset":
        """from_tensor_slices docstring."""
        return cls(tensors)

    def batch(self, batch_size: int) -> "Dataset":
        """batch docstring."""
        batched = [
            self.elements[i : i + batch_size]
            for i in range(0, len(self.elements), batch_size)
        ]
        return Dataset(batched)

    def map(self, map_func: Callable[[Any], Any]) -> "Dataset":
        """map docstring."""
        return Dataset([map_func(e) for e in self.elements])

    def shuffle(self, buffer_size: int) -> "Dataset":
        """shuffle docstring."""
        return self

    def __iter__(self) -> PyIterator[Any]:
        """__iter__ docstring."""
        return iter(self.elements)


class FixedLengthRecordDataset(Dataset):
    """FixedLengthRecordDataset docstring."""

    def __init__(self, filenames=None, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__([])
        self.filenames = filenames
        for k, v in kwargs.items():
            setattr(self, k, v)


class TFRecordDataset(Dataset):
    """TFRecordDataset docstring."""

    def __init__(self, filenames=None, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__([])
        self.filenames = filenames
        for k, v in kwargs.items():
            setattr(self, k, v)


class TextLineDataset(Dataset):
    """TextLineDataset docstring."""

    def __init__(self, filenames=None, *args: Any, **kwargs: Any):
        """__init__ docstring."""
        super().__init__([])
        self.filenames = filenames
        for k, v in kwargs.items():
            setattr(self, k, v)
