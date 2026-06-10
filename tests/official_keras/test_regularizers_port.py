import pytest
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class RegularizersTest(test_util.TensorFlowTestCase):
    @pytest.mark.skip(reason="zero-keras does not implement regularizers fully yet")
    def test_l2_basic(self):
        from tensorflow.keras import regularizers

        reg = regularizers.L2(0.01)
        self.assertIsNotNone(reg)


if __name__ == "__main__":
    test.main()
