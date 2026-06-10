import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class DatasetTest(test_util.TensorFlowTestCase):
    def testFromTensorSlices(self):
        ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
        ds = ds.batch(2).map(lambda x: x).shuffle(10)
        self.assertTrue(isinstance(ds, tf.data.Dataset))
        # We aren't doing graph iteration for now as the current zero_tensorflow Dataset API is just a stub for tracking.


if __name__ == "__main__":
    test.main()
