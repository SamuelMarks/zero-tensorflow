from zero_tensorflow.test import experimental


def test_test():
    assert experimental.sync_devices() is None
