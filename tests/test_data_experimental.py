"""Unit tests for `test_data_experimental`."""

from __future__ import annotations

from zero_tensorflow.data_experimental import SqlDataset


def test_sql_dataset() -> None:
    """Verify `SqlDataset` initialization and iteration."""
    ds = SqlDataset(
        driver_name="sqlite",
        data_source_name="db.sqlite",
        query="SELECT * FROM table",
        output_types="int",
    )
    assert ds.driver_name == "sqlite"
    assert ds.data_source_name == "db.sqlite"
    assert ds.query == "SELECT * FROM table"
    assert ds.output_types == "int"
    assert list(ds) == [()]


def test_sql_dataset_kwargs() -> None:
    """Verify `SqlDataset` stores extra keyword options in `ds.options`."""
    ds = SqlDataset("a", "b", "c", "d", extra=True)
    assert ds.options["extra"] is True
