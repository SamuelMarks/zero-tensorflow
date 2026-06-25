import pytest
from zero_tensorflow import io


def test_read_write_file(tmp_path):
    p = tmp_path / "test.txt"
    io.write_file(str(p), b"hello world")
    res = io.read_file(str(p))
    assert res == b"hello world"


def test_io_not_implemented():
    with pytest.raises(NotImplementedError):
        io.decode_jpeg("dummy")
    with pytest.raises(NotImplementedError):
        io.decode_png("dummy")
    with pytest.raises(NotImplementedError):
        io.decode_image("dummy")
    with pytest.raises(NotImplementedError):
        io.parse_tensor("dummy", "type")
    with pytest.raises(NotImplementedError):
        io.serialize_tensor("dummy")
    with pytest.raises(NotImplementedError):
        io.parse_example("dummy", "features")
    with pytest.raises(NotImplementedError):
        io.parse_single_example("dummy", "features")
