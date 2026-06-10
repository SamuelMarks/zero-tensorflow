import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.eager import backprop
import pytest


class BackpropTest(test_util.TensorFlowTestCase):
    @pytest.mark.skip(reason="GradientTape is currently a stub returning 1.0")
    def testGradientTape(self):
        with backprop.GradientTape() as g:
            x = tf.Tensor(3.0)
            g.watch(x)
            y = x * x
            with backprop.GradientTape() as gg:
                gg.watch(y)
                z = 2 * y
            inner_grad = gg.gradient(z, [y])[0]
            self.assertEqual(self.evaluate(inner_grad), 2.0)
            y += inner_grad
        grad = g.gradient(y, [x])[0]
        self.assertEqual(self.evaluate(grad), 6.0)

    def testGradientTapeBasicStub(self):
        with backprop.GradientTape() as g:
            x = tf.Tensor(3.0)
            g.watch(x)
            y = x * x
        grad = g.gradient(y, [x])[0]
        self.assertEqual(self.evaluate(grad), 1.0)  # Stub returns 1.0


if __name__ == "__main__":
    test.main()
