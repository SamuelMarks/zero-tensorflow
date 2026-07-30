from zero_tensorflow.debugging import experimental


def test_debugging():
    assert experimental.disable_dump_debug_info() is None
    assert experimental.enable_dump_debug_info() is None
