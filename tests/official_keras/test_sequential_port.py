import pytest
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class SequentialTest(test_util.TensorFlowTestCase):
    @pytest.mark.skip(reason="zero-keras does not implement models.Sequential yet")
    def test_sequential_basic(self):
        from tensorflow.keras import models, layers

        model = models.Sequential([layers.Dense(2, activation="relu"), layers.Dense(1)])

        # Build manually since zero-keras is an API shell
        model.build((None, 2))
        self.assertEqual(len(model.layers), 2)
        self.assertTrue(model.built)


if __name__ == "__main__":
    test.main()
