import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class VariableTest(test_util.TensorFlowTestCase):
    def testBasic(self):
        v = tf.Variable(1.0)
        self.assertEqual(v.value.numpy(), 1.0)

    def testAssign(self):
        v = tf.Variable(1.0)
        v.assign(3.0)
        self.assertEqual(v.value.numpy(), 3.0)

    def testAssignAdd(self):
        v = tf.Variable(1.0)
        v.assign_add(2.0)
        self.assertEqual(v.value.numpy(), 3.0)

    def testAssignSub(self):
        v = tf.Variable(3.0)
        v.assign_sub(1.0)
        self.assertEqual(v.value.numpy(), 2.0)


if __name__ == "__main__":
    test.main()
