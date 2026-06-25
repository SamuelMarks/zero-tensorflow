from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class InitializersTest(test_util.TensorFlowTestCase):
    def test_random_normal(self):
        from tensorflow.keras import initializers

        init = initializers.RandomNormal()
        self.assertIsNotNone(init)


if __name__ == "__main__":
    test.main()
