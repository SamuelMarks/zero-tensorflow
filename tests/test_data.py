import pytest
from zero_tensorflow import data


def test_threading_options():
    opt = data.ThreadingOptions(max_intra_op_parallelism=2, private_threadpool_size=4)
    assert opt.max_intra_op_parallelism == 2
    assert opt.private_threadpool_size == 4


def test_options():
    opt = data.Options(
        autotune=True,
        deterministic=True,
        experimental_deterministic=True,
        experimental_distribute="dist",
        experimental_external_state_policy="policy",
        experimental_optimization="opt",
        experimental_slack=True,
        experimental_symbolic_checkpoint=True,
        experimental_service="service",
        experimental_threading="threading",
        experimental_warm_start=True,
        dataset_name="name",
        framework_type=["type"],
        threading=data.ThreadingOptions(2, 4),
    )
    assert opt.autotune is True
    assert opt.deterministic is True
    assert opt.experimental_deterministic is True
    assert opt.experimental_distribute == "dist"
    assert opt.experimental_external_state_policy == "policy"
    assert opt.experimental_optimization == "opt"
    assert opt.experimental_slack is True
    assert opt.experimental_symbolic_checkpoint is True
    assert opt.experimental_service == "service"
    assert opt.experimental_threading == "threading"
    assert opt.experimental_warm_start is True
    assert opt.dataset_name == "name"
    assert opt.framework_type == ["type"]
    assert opt.threading.max_intra_op_parallelism == 2

    # Test default threading
    opt2 = data.Options()
    assert opt2.threading is not None


def test_dataset_spec():
    spec = data.DatasetSpec(element_spec="int", dataset_shape=(10,))
    assert spec._element_spec == "int"
    assert spec._dataset_shape == (10,)


def test_iterator_spec():
    spec = data.IteratorSpec(element_spec="int")
    assert spec._element_spec == "int"


def test_iterator():
    ds = data.Dataset([1, 2, 3])
    it = data.Iterator(ds)
    assert next(it) == 1
    assert next(iter(it)) == 2


def test_array_iterator():
    ds = data.Dataset([1, 2, 3])
    it = data.ArrayIterator(ds)
    assert next(it) == 1
    assert next(iter(it)) == 2


def test_dataset():
    # from_tensor_slices
    ds1 = data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
    assert list(iter(ds1)) == [1, 2, 3, 4, 5]

    ds2 = data.Dataset.from_tensor_slices("hello")
    assert list(iter(ds2)) == ["h", "e", "l", "l", "o"]

    # batch
    ds_batched = ds1.batch(2)
    assert list(ds_batched) == [[1, 2], [3, 4], [5]]

    with pytest.raises(ValueError):
        ds1.batch(0)

    # map
    ds_mapped = ds1.map(lambda x: x * 2)
    assert list(ds_mapped) == [2, 4, 6, 8, 10]

    # shuffle
    ds_shuffled = ds1.shuffle(10)
    assert set(list(ds_shuffled)) == {1, 2, 3, 4, 5}

    ds_shuffled_zero = ds1.shuffle(0)
    assert list(ds_shuffled_zero) == [1, 2, 3, 4, 5]


def test_fixed_length_record_dataset():
    ds = data.FixedLengthRecordDataset(
        filenames="file.bin",
        record_bytes=10,
        header_bytes=1,
        footer_bytes=2,
        buffer_size=100,
        compression_type="GZIP",
        num_parallel_reads=4,
        name="my_dataset",
    )
    assert ds.filenames == "file.bin"
    assert ds.record_bytes == 10
    assert ds.header_bytes == 1
    assert ds.footer_bytes == 2
    assert ds.buffer_size == 100
    assert ds.compression_type == "GZIP"
    assert ds.num_parallel_reads == 4
    assert ds.name == "my_dataset"


def test_tf_record_dataset():
    ds = data.TFRecordDataset(
        filenames=["f1.tfrecord", "f2.tfrecord"],
        compression_type="ZLIB",
        buffer_size=200,
        num_parallel_reads=2,
        name="tfrec_ds",
    )
    assert ds.filenames == ["f1.tfrecord", "f2.tfrecord"]
    assert ds.compression_type == "ZLIB"
    assert ds.buffer_size == 200
    assert ds.num_parallel_reads == 2
    assert ds.name == "tfrec_ds"


def test_text_line_dataset():
    ds = data.TextLineDataset(
        filenames="text.txt",
        compression_type=None,
        buffer_size=300,
        num_parallel_reads=1,
        name="text_ds",
    )
    assert ds.filenames == "text.txt"
    assert ds.compression_type is None
    assert ds.buffer_size == 300
    assert ds.num_parallel_reads == 1
    assert ds.name == "text_ds"
