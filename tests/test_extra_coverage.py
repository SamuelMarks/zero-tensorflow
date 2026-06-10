from zero_tensorflow import _to_tensor
from zero_tensorflow.nn import softmax
import numpy as np


def test_missing_branches():
    # for __init__.py 57->61
    from ml_switcheroo.tracing import ProxyTensor

    pt = ProxyTensor("test", ())
    pt.dtype = None
    _to_tensor(pt)

    # for nn.py 76->78
    softmax(np.array([1.0, 2.0]), axis=0)
