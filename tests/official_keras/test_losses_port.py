import pytest
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class LossesTest(test_util.TensorFlowTestCase):
    @pytest.mark.skip(reason="zero-keras does not implement losses fully yet")
    def test_mse_basic(self):
        from tensorflow.keras import losses

        mse = losses.MeanSquaredError()
        self.assertIsNotNone(mse)


if __name__ == "__main__":
    test.main()
