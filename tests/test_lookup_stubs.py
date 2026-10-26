"""Unit tests for `zero_tensorflow.lookup` hash tables and initializers."""

from __future__ import annotations

from zero_tensorflow import Tensor, debugging
from zero_tensorflow.lookup import (
    KeyValueTensorInitializer,
    StaticHashTable,
    StaticVocabularyTable,
    TextFileIndex,
    TextFileInitializer,
    _to_flat_keys,
    experimental,
)


def test_lookup() -> None:
    """Verify `KeyValueTensorInitializer`, `StaticHashTable`, `DenseHashTable`, and `MutableHashTable`."""
    assert TextFileIndex.WHOLE_LINE == -2
    assert TextFileIndex.LINE_NUMBER == -1

    init = KeyValueTensorInitializer(Tensor(["a", "b"]), Tensor([10, 20]))
    table = StaticHashTable(init, default_value=-1)
    looked_up = table.lookup(["a", "c", "b"])
    debugging.assert_near(looked_up, [10, -1, 20])
    assert table.size().numpy() == 2

    t_init = TextFileInitializer(filename="vocab.txt")
    v_table = StaticVocabularyTable(t_init, 0)
    assert v_table.size().numpy() == 0

    dense = experimental.DenseHashTable()
    dense.insert(["k1", "k2"], [100, 200])
    assert dense.size().numpy() == 2
    debugging.assert_near(dense.lookup(["k1", "missing"]), [100, -1])
    dense.remove(["k1"])
    assert dense.size().numpy() == 1
    keys_t, vals_t = dense.export()
    assert keys_t.shape == (1,)
    debugging.assert_near(vals_t, [200])

    mutable = experimental.MutableHashTable(None, 99)
    mutable.insert(1, 42)
    debugging.assert_near(mutable.lookup([1, 2]), [42, 99])

    # Exercise _to_flat_keys fallback on custom object and nested sequence
    assert _to_flat_keys(dense) == [str(dense)]
    assert _to_flat_keys([[1, 2]]) == ["[1, 2]"]
