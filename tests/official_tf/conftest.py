import sys
import zero_tensorflow as tf
import types
import pytest
import ml_switcheroo_compiler


@pytest.fixture(autouse=True)
def switcheroo_config():
    with ml_switcheroo_compiler.EagerMode():
        yield


# Create a fake tensorflow module that points to zero_tensorflow
class Dummy:
    pass


class FakeTensorFlowModule(types.ModuleType):
    def __getattr__(self, name):
        if hasattr(tf, name):
            return getattr(tf, name)
        # Mock some common submodules
        if name in ("python", "framework", "ops", "core", "eager", "platform"):
            return self
        if name == "test_util":
            import unittest

            class FakeTestUtil:
                class TensorFlowTestCase(unittest.TestCase):
                    def evaluate(self, tensor):
                        if hasattr(tensor, "numpy"):
                            return tensor.numpy()
                        return tensor

                @staticmethod
                def run_all_in_graph_and_eager_modes(cls=None):
                    if cls is None:
                        return lambda c: c
                    return cls

                @staticmethod
                def device(use_gpu=False):
                    from contextlib import contextmanager

                    @contextmanager
                    def mock_device():
                        yield

                    return mock_device()

                def __getattr__(self, name):
                    def dummy_decorator(*args, **kwargs):
                        if len(args) == 1 and callable(args[0]) and not kwargs:
                            return args[0]

                        def real_decorator(func):
                            return func

                        return real_decorator

                    return dummy_decorator

            return FakeTestUtil()

        if name == "dtypes":
            import numpy as np

            class FakeDTypes:
                float16 = np.float16
                float32 = np.float32
                float64 = np.float64
                int32 = np.int32
                int64 = np.int64
                complex64 = np.complex64
                complex128 = np.complex128
                bool = np.bool_
                uint8 = np.uint8
                int8 = np.int8
                int16 = np.int16
                uint16 = np.uint16
                uint32 = np.uint32
                uint64 = np.uint64
                bfloat16 = np.float32  # mock

            return FakeDTypes()

        if name == "math_ops":
            return tf.math

        if name == "constant_op":

            class FakeConstantOp:
                constant = tf.Tensor

            return FakeConstantOp()

        if name == "backprop":

            class FakeBackprop:
                GradientTape = tf.GradientTape

            return FakeBackprop()

        if name == "def_function":

            class FakeDefFunction:
                function = staticmethod(tf.function)

            return FakeDefFunction()

        if name == "nn_ops":
            return tf.nn

        # Return a dummy object for anything else
        return Dummy()


sys.modules["tensorflow"] = FakeTensorFlowModule("tensorflow")

# Populate common descendent modules dynamically
for sub in [
    "tensorflow.python",
    "tensorflow.python.framework",
    "tensorflow.python.eager",
    "tensorflow.python.ops",
    "tensorflow.python.ops.ragged",
    "tensorflow.python.platform",
    "tensorflow.core",
    "tensorflow.core.framework",
]:
    sys.modules[sub] = sys.modules["tensorflow"]

sys.modules["tensorflow.python.framework.test_util"] = sys.modules["tensorflow"]

# Also add keras to tensorflow.keras
import zero_keras as keras

sys.modules["tensorflow.keras"] = keras
