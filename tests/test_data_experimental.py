from zero_tensorflow.data_experimental import SqlDataset


def test_sql_dataset():
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


def test_sql_dataset_kwargs():
    ds = SqlDataset("a", "b", "c", "d", extra=True)
    assert ds.extra is True
