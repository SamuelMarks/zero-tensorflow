from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
import numpy as np


class ConstantOpTest(test_util.TensorFlowTestCase):
    def test_np_array_memory_not_shared(self):
        # An arbitrarily large loop number to test memory sharing
        for _ in range(100):
            x = np.arange(10)
            xt = constant_op.constant(x)
            x[3] = 42
            # Changing the input array after `xt` is created should not affect `xt`
            self.assertEqual(xt.numpy()[3], 3)


if __name__ == "__main__":
    test.main()
