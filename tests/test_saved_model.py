import pytest
from zero_tensorflow import saved_model


def test_saved_model():
    with pytest.raises(NotImplementedError):
        saved_model.save("obj", "dir")
    with pytest.raises(NotImplementedError):
        saved_model.load("dir")
