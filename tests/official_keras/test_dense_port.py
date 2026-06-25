import numpy as np
from tensorflow.python.framework import test_util
from tensorflow.keras import layers
from tensorflow.python.platform import test


class DenseTest(test_util.TensorFlowTestCase):
    def test_dense_correctness(self):
        # With bias and activation.
        layer = layers.Dense(
            units=2,
            activation="relu",
            use_bias=True,
        )
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([0.5, -0.5]),
            ]
        )
        inputs = np.array([[1.0, 1.0]])
        expected_output = np.array([[4.5, 5.5]])

        # zero-tensorflow uses __call__ like keras, but we return a Tensor
        outputs = layer(inputs)

        self.assertAllClose(outputs, expected_output)

    def test_dense_api_signature(self):
        layer = layers.Dense(units=2, activation="relu", use_bias=True)
        layer.build((1, 2))
        self.assertTrue(layer.built)
        self.assertEqual(layer.units, 2)

    def assertAllClose(self, a, b):
        a = a.numpy() if hasattr(a, "numpy") else a
        np.testing.assert_allclose(a, b)


if __name__ == "__main__":
    test.main()
