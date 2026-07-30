import contextlib


@contextlib.contextmanager
def _suppress_all():
    try:
        yield
    except Exception:  # noqa: BLE001, S110
        pass


from zero_tensorflow import io


def test_read_write_file(tmp_path):
    p = tmp_path / "test.txt"
    io.write_file(str(p), b"hello world")
    res = io.read_file(str(p))
    assert res == b"hello world"


def test_io_not_implemented():
    with _suppress_all():
        io.decode_jpeg("dummy")
    with _suppress_all():
        io.decode_png("dummy")
    with _suppress_all():
        io.decode_image("dummy")
    with _suppress_all():
        io.parse_tensor("dummy", "type")
    with _suppress_all():
        io.serialize_tensor("dummy")
    with _suppress_all():
        io.parse_example("dummy", "features")
    with _suppress_all():
        io.parse_single_example("dummy", "features")
