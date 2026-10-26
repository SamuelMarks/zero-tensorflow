"""TensorFlow lookup table module."""

from __future__ import annotations

from collections.abc import Sequence

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    NumpyConvertibleProtocol,
    ScalarValue,
    TensorOperand,
    as_operand,
)
from zero_tensorflow.core_tensor import Tensor

__all__ = [
    "KeyValueTensorInitializer",
    "StaticHashTable",
    "StaticVocabularyTable",
    "TextFileIndex",
    "TextFileInitializer",
    "experimental",
]


def _to_flat_keys(x: TensorOperand | _BaseHashTable) -> list[ScalarValue]:
    """Convert a tensor, array, or sequence of keys/values into a flat list of scalars.

    Args:
        x: Input keys or values operand.

    Returns:
        list[ScalarValue]: Flattened scalar list.
    """
    if isinstance(x, NumpyConvertibleProtocol):
        x = as_operand(x.numpy())
    if isinstance(x, ArrayLikeProtocol):
        x = x.tolist()
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return [
            item if isinstance(item, (int, float, bool, str, bytes)) else str(item) for item in x
        ]
    if isinstance(x, (int, float, bool, str, bytes)) or x is None:
        return [x]
    return [str(x)]


class TextFileIndex:
    """Constants specifying which column/element of a text file represents keys or values."""

    WHOLE_LINE: int = -2
    LINE_NUMBER: int = -1


class KeyValueTensorInitializer:
    """Table initializer given `keys` and `values` tensors."""

    def __init__(
        self,
        keys: TensorOperand,
        values: TensorOperand,
        key_dtype: TensorOperand = None,
        value_dtype: TensorOperand = None,
        name: str | None = None,
    ) -> None:
        """Initialize `KeyValueTensorInitializer`.

        Args:
            keys: Keys tensor or sequence.
            values: Values tensor or sequence.
            key_dtype: Optional key dtype.
            value_dtype: Optional value dtype.
            name: Optional operation name.
        """
        self.keys = _to_flat_keys(keys)
        self.values = _to_flat_keys(values)
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.name = name


class TextFileInitializer(KeyValueTensorInitializer):
    """Table initializer from a text file."""

    def __init__(
        self,
        filename: str = "",
        key_dtype: TensorOperand = None,
        key_index: int = 0,
        value_dtype: TensorOperand = None,
        value_index: int = 1,
        vocab_size: int | None = None,
        delimiter: str = "	",
        name: str | None = None,
        value_index_offset: int = 0,
    ) -> None:
        """Initialize `TextFileInitializer`.

        Args:
            filename: Path to vocabulary/lookup text file.
            key_dtype: Key data type.
            key_index: Column index for keys.
            value_dtype: Value data type.
            value_index: Column index for values.
            vocab_size: Optional maximum number of entries to read.
            delimiter: Field delimiter string.
            name: Optional initializer name.
            value_index_offset: Offset added to `LINE_NUMBER` values.
        """
        super().__init__([], [], key_dtype=key_dtype, value_dtype=value_dtype, name=name)
        self.filename = filename
        self.key_index = key_index
        self.value_index = value_index
        self.vocab_size = vocab_size
        self.delimiter = delimiter
        self.value_index_offset = value_index_offset


class _BaseHashTable:
    """Shared functional hash table backing `StaticHashTable`, `DenseHashTable`, and `MutableHashTable`."""

    def __init__(
        self,
        *args: KeyValueTensorInitializer | TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `_BaseHashTable`.

        Args:
            *args: Positional table arguments.
            **kwargs: Keyword table arguments.
        """
        self.args = args
        self.kwargs = kwargs
        self._table: dict[ScalarValue, ScalarValue] = {}
        default_cand = args[1] if len(args) >= 2 else kwargs.get("default_value")
        self._default_value: ScalarValue = (
            default_cand if isinstance(default_cand, (int, float, bool, str, bytes)) else -1
        )
        if args and isinstance(args[0], KeyValueTensorInitializer):
            for k, v in zip(args[0].keys, args[0].values):
                self._table[k] = v

    def insert(self, keys: TensorOperand, values: TensorOperand, name: str | None = None) -> None:
        """Insert `keys` and `values` into the hash table.

        Args:
            keys: Keys tensor or sequence.
            values: Values tensor or sequence.
            name: Optional operation name.
        """
        del name
        for k, v in zip(_to_flat_keys(keys), _to_flat_keys(values)):
            self._table[k] = v

    def remove(self, keys: TensorOperand, name: str | None = None) -> None:
        """Remove `keys` from the hash table.

        Args:
            keys: Keys tensor or sequence to remove.
            name: Optional operation name.
        """
        del name
        for k in _to_flat_keys(keys):
            self._table.pop(k, None)

    def lookup(self, keys: TensorOperand, name: str | None = None) -> Tensor:
        """Look up `keys` in the table, returning `default_value` for missing keys.

        Args:
            keys: Query keys tensor or sequence.
            name: Optional operation name.

        Returns:
            Tensor: Looked-up values tensor.
        """
        del name
        flat = _to_flat_keys(keys)
        results = [self._table.get(k, self._default_value) for k in flat]
        return Tensor(results)

    def size(self, name: str | None = None) -> Tensor:
        """Return the number of elements currently stored in the table.

        Args:
            name: Optional operation name.

        Returns:
            Tensor: Scalar integer size tensor.
        """
        del name
        return Tensor(len(self._table))

    def export(self, name: str | None = None) -> tuple[Tensor, Tensor]:
        """Return `(keys_tensor, values_tensor)` of all items in the table.

        Args:
            name: Optional operation name.

        Returns:
            tuple[Tensor, Tensor]: Tuple of `(keys, values)` tensors.
        """
        del name
        return (Tensor(list(self._table.keys())), Tensor(list(self._table.values())))


class StaticHashTable(_BaseHashTable):
    """Immutable generic hash table initialized from a `KeyValueTensorInitializer`."""

    def __init__(
        self,
        initializer: KeyValueTensorInitializer | TensorOperand = None,
        default_value: TensorOperand = -1,
        name: str | None = None,
        experimental_is_anonymous: bool = False,
    ) -> None:
        """Initialize `StaticHashTable`.

        Args:
            initializer: Table initializer (`KeyValueTensorInitializer` or `TextFileInitializer`).
            default_value: Value to return when a key is missing from the table.
            name: Optional name for the operation.
            experimental_is_anonymous: Whether to allocate the table in anonymous mode.
        """
        super().__init__(initializer, default_value)
        self.initializer = initializer
        self.default_value = default_value
        self.name = name
        self.experimental_is_anonymous = experimental_is_anonymous


class StaticVocabularyTable(_BaseHashTable):
    """String-to-ID lookup table with out-of-vocabulary bucket hashing."""

    def __init__(
        self,
        initializer: KeyValueTensorInitializer | TensorOperand = None,
        num_oov_buckets: int = 1,
        lookup_key_dtype: TensorOperand = None,
        name: str | None = None,
        experimental_is_anonymous: bool = False,
    ) -> None:
        """Initialize `StaticVocabularyTable`.

        Args:
            initializer: Table initializer (`KeyValueTensorInitializer` or `TextFileInitializer`).
            num_oov_buckets: Number of out-of-vocabulary buckets.
            lookup_key_dtype: Data type of keys passed to `lookup`.
            name: Optional name for the operation.
            experimental_is_anonymous: Whether to allocate the table in anonymous mode.
        """
        super().__init__(initializer, -1)
        self.initializer = initializer
        self.num_oov_buckets = num_oov_buckets
        self.lookup_key_dtype = lookup_key_dtype
        self.name = name
        self.experimental_is_anonymous = experimental_is_anonymous


class experimental:
    """Experimental hash table lookup classes."""

    class DenseHashTable(_BaseHashTable):
        """Mutable hash table using dense open addressing."""

    class MutableHashTable(_BaseHashTable):
        """Generic mutable hash table implementation."""
