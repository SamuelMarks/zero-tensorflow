from zero_tensorflow.saved_model import (
    Asset,
    LoadOptions,
    SaveOptions,
    contains_saved_model,
    experimental,
    load,
    save,
)


def test_saved_model():
    assert Asset() is not None
    assert LoadOptions() is not None
    assert SaveOptions() is not None
    assert contains_saved_model() is None

    # test experimental
    assert experimental.Fingerprint() is not None
    assert experimental.TrackableResource() is not None
    assert experimental.VariablePolicy() is not None
    assert experimental.read_fingerprint() is None

    # test save load
    save(None, "path")

    def dummy_call():
        pass

    # Just checking it doesn't crash
    save(dummy_call, "path")
    assert load("invalid_path") is None
