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
    """
    Threading options for dataset operations.

    Args:
        max_intra_op_parallelism: Maximum parallelism.
        private_threadpool_size: Threadpool size.
    """

    def __init__(
        self,
        max_intra_op_parallelism=None,
        private_threadpool_size=None,
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

    def __init__(self, autotune=None, threading=None, *args: Any, **kwargs: Any):
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

    __slots__ = ["_element_spec", "_dataset_shape"]

    def __init__(
        self, element_spec=None, dataset_shape=None, *args: Any, **kwargs: Any
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

    def __init__(self, element_spec=None, *args: Any, **kwargs: Any):
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

    def __init__(self, dataset, *args: Any, **kwargs: Any):
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

        Args:
            None

        Returns:
            Any: The next element.
        """
        return next(self._iter)

    def __iter__(self) -> "Iterator":
        """
        Get the iterator.

        Args:
            None

        Returns:
            Iterator: The iterator.
        """
        return self


class NumpyIterator:
    """
    Iterator over a dataset returning numpy arrays.

    Args:
        dataset: The dataset to iterate over.
    """

    __slots__ = ["_iterator"]

    def __init__(self, dataset, *args: Any, **kwargs: Any):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._iterator = iter(dataset)

    def __next__(self) -> Any:
        """
        Get the next element.

        Args:
            None

        Returns:
            Any: The next element.
        """
        return next(self._iterator)

    def __iter__(self) -> "NumpyIterator":
        """
        Get the iterator.

        Args:
            None

        Returns:
            NumpyIterator: The iterator.
        """
        return self


class Dataset:
    """
    Dataset object.

    Args:
        elements: Elements of the dataset.
    """

    def __init__(self, elements, *args: Any, **kwargs: Any):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.elements = list(elements) if elements is not None else []

    @classmethod
    def from_tensor_slices(cls, tensors: Any) -> "Dataset":
        """
        Create a dataset from tensor slices.

        Args:
            tensors (Any): Tensors to slice.

        Returns:
            Dataset: The created dataset.
        """
        return cls(tensors)

    def batch(self, batch_size: int) -> "Dataset":
        """
        Batch the dataset elements.

        Args:
            batch_size (int): Size of the batches.

        Returns:
            Dataset: A new batched dataset.
        """
        batched = [
            self.elements[i : i + batch_size]
            for i in range(0, len(self.elements), batch_size)
        ]
        return Dataset(batched)

    def map(self, map_func: Callable[[Any], Any]) -> "Dataset":
        """
        Map a function over the dataset.

        Args:
            map_func (Callable): Function to apply.

        Returns:
            Dataset: A new dataset with mapped elements.
        """
        return Dataset([map_func(e) for e in self.elements])

    def shuffle(self, buffer_size: int) -> "Dataset":
        """
        Shuffle the dataset.

        Args:
            buffer_size (int): Number of elements from which to shuffle.

        Returns:
            Dataset: The shuffled dataset.
        """
        return self

    def __iter__(self) -> "PyIterator[Any]":
        """
        Get the iterator.

        Args:
            None

        Returns:
            PyIterator[Any]: The iterator.
        """
        return iter(self.elements)


class FixedLengthRecordDataset(Dataset):
    """
    Dataset for fixed length records.

    Args:
        filenames: Files to read.
    """

    def __init__(self, filenames=None, *args: Any, **kwargs: Any):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__([])
        self.filenames = filenames
        for k, v in kwargs.items():
            setattr(self, k, v)


class TFRecordDataset(Dataset):
    """
    Dataset for TFRecords.

    Args:
        filenames: Files to read.
    """

    def __init__(self, filenames=None, *args: Any, **kwargs: Any):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__([])
        self.filenames = filenames
        for k, v in kwargs.items():
            setattr(self, k, v)


class TextLineDataset(Dataset):
    """
    Dataset for text lines.

    Args:
        filenames: Files to read.
    """

    def __init__(self, filenames=None, *args: Any, **kwargs: Any):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__([])
        self.filenames = filenames
        for k, v in kwargs.items():
            setattr(self, k, v)
