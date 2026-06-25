from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class MetricsTest(test_util.TensorFlowTestCase):
    def test_accuracy_basic(self):
        from tensorflow.keras import metrics

        acc = metrics.Accuracy()
        self.assertIsNotNone(acc)


if __name__ == "__main__":
    test.main()
