import pytest
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class OptimizersTest(test_util.TensorFlowTestCase):
    @pytest.mark.skip(reason="zero-keras does not implement optimizers yet")
    def test_adam_basic(self):
        from tensorflow.keras import optimizers

        opt = optimizers.Adam(learning_rate=0.01)
        self.assertEqual(opt.learning_rate, 0.01)


if __name__ == "__main__":
    test.main()
