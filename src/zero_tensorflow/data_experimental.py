"""TensorFlow data.experimental module."""

from typing import Any, Iterator as PyIterator
from zero_tensorflow.data import Dataset

__all__ = ["SqlDataset"]


class SqlDataset(Dataset):
    """
    Dataset from SQL query.

    Args:
        driver_name: Database driver.
        data_source_name: Connection string.
        query: SQL query.
        output_types: Types of columns.
    """

    def __init__(
        self,
        driver_name: str,
        data_source_name: str,
        query: str,
        output_types: Any,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.driver_name = driver_name
        self.data_source_name = data_source_name
        self.query = query
        self.output_types = output_types
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _generator(self) -> PyIterator[Any]:
        yield ()
