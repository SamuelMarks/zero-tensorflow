import numpy as np
from zero_tensorflow import Tensor as ZTensor
from ml_switcheroo_compiler.ops import control_flow
import ml_switcheroo_compiler.core.config as config


def test_control_flow_eager():
    config.eager_mode = True

    x = ZTensor(np.array([[1.0, 2.0], [3.0, 4.0]]))._tensor

    def true_fn():
        return x

    def false_fn():
        return x

    control_flow.cond(ZTensor(True)._tensor, true_fn, false_fn)
    control_flow.cond(ZTensor(False)._tensor, true_fn, false_fn)

    # HIT WHILE LOOP BODY
    ctr = [0]

    def cond_fn(val):
        ctr[0] += 1
        return ZTensor(ctr[0] < 2)._tensor

    def body_fn(val):
        return val

    control_flow.while_loop(cond_fn, body_fn, x)

    # HIT SCAN ELSE BRANCH
    # If xs is empty, it returns empty array
    class TrickArray(np.ndarray):
        @property
        def data(self):
            raise AttributeError()

    def scan_fn_trick(c, xs):
        arr = np.array([1.0, 2.0]).view(TrickArray)
        return c, arr

    control_flow.scan(scan_fn_trick, x, x)

    def scan_fn_empty(c, xs):
        return c, 1.0

    empty_x = ZTensor(np.array([]))._tensor
    control_flow.scan(scan_fn_empty, x, empty_x)

    def vmap_fn(val):
        return val

    vmapped = control_flow.vmap(vmap_fn)
    vmapped(x)

    pmapped = control_flow.pmap(vmap_fn)
    pmapped(x)

    control_flow.stop_gradient(x)
