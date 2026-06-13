import numpy as np
from zero_tensorflow import Tensor as ZTensor
from ml_switcheroo_compiler.ops.shape import frontend as shape_frontend
import ml_switcheroo_compiler.core.config as config


def test_shape_eager():
    config.eager_mode = True

    x = ZTensor(np.array([[1.0, 2.0], [3.0, 4.0]]))._tensor

    # line 71: reshape
    shape_frontend.reshape(x, (4,))

    # line 210:
    # let's see what is line 210
    shape_frontend.broadcast_to(x, (2, 2, 2))

    # dynamic_slice
    shape_frontend.dynamic_slice(x, [ZTensor(0)._tensor], [1])
    shape_frontend.dynamic_slice(x, [0], [1])

    # update_slice
    shape_frontend.update_slice(
        x, ZTensor(np.array([[5.0]]))._tensor, [ZTensor(0)._tensor, ZTensor(0)._tensor]
    )
    shape_frontend.update_slice(x, ZTensor(np.array([[5.0]]))._tensor, [0, 0])

    # strided_slice
    shape_frontend.strided_slice(x, [0], [1], [1])
