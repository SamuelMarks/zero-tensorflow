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
    def __init__(
        self,
        max_intra_op_parallelism=None,
        private_threadpool_size=None,
        *args: Any,
        **kwargs: Any,
    ):
        self.max_intra_op_parallelism = max_intra_op_parallelism
        self.private_threadpool_size = private_threadpool_size


class Options:
    def __init__(self, autotune=None, threading=None, *args: Any, **kwargs: Any):
        self.autotune = autotune
        self.threading = threading if threading is not None else ThreadingOptions()
        for k, v in kwargs.items():
            setattr(self, k, v)


class DatasetSpec:
    __slots__ = ["_element_spec", "_dataset_shape"]

    def __init__(
        self, element_spec=None, dataset_shape=None, *args: Any, **kwargs: Any
    ):
        self._element_spec = element_spec
        self._dataset_shape = dataset_shape


class IteratorSpec:
    __slots__ = ["_element_spec"]

    def __init__(self, element_spec=None, *args: Any, **kwargs: Any):
        self._element_spec = element_spec


class Iterator:
    def __init__(self, dataset, *args: Any, **kwargs: Any):
        self._dataset = dataset
        self._iter = iter(dataset)

    def __next__(self) -> Any:
        return next(self._iter)

    def __iter__(self) -> "Iterator":
        return self


class NumpyIterator:
    __slots__ = ["_iterator"]

    def __init__(self, dataset, *args: Any, **kwargs: Any):
        self._iterator = iter(dataset)

    def __next__(self) -> Any:
        return next(self._iterator)

    def __iter__(self) -> "NumpyIterator":
        return self


class Dataset:
    def __init__(self, elements, *args: Any, **kwargs: Any):
        self.elements = list(elements) if elements is not None else []

    @classmethod
    def from_tensor_slices(cls, tensors: Any) -> "Dataset":
        return cls(tensors)

    def batch(self, batch_size: int) -> "Dataset":
        batched = [
            self.elements[i : i + batch_size]
            for i in range(0, len(self.elements), batch_size)
        ]
        return Dataset(batched)

    def map(self, map_func: Callable[[Any], Any]) -> "Dataset":
        return Dataset([map_func(e) for e in self.elements])

    def shuffle(self, buffer_size: int) -> "Dataset":
        return self

    def __iter__(self) -> PyIterator[Any]:
        return iter(self.elements)


class FixedLengthRecordDataset(Dataset):
    def __init__(self, filenames=None, *args: Any, **kwargs: Any):
        super().__init__([])
        self.filenames = filenames
        for k, v in kwargs.items():
            setattr(self, k, v)


class TFRecordDataset(Dataset):
    def __init__(self, filenames=None, *args: Any, **kwargs: Any):
        super().__init__([])
        self.filenames = filenames
        for k, v in kwargs.items():
            setattr(self, k, v)


class TextLineDataset(Dataset):
    def __init__(self, filenames=None, *args: Any, **kwargs: Any):
        super().__init__([])
        self.filenames = filenames
        for k, v in kwargs.items():
            setattr(self, k, v)
