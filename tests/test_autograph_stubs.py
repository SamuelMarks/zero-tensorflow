from zero_tensorflow.autograph import experimental


def test_autograph():
    assert experimental.Feature() is not None
    assert experimental.do_not_convert() is not None
    assert experimental.set_loop_options() is None
