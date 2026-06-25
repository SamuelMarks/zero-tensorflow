import pytest
from zero_tensorflow import GradientTape, Tensor


def test_jacobian():
    with pytest.raises(NotImplementedError):
        with GradientTape() as t:
            x = Tensor(1.0)
            t.watch(x)
            y = x * 2.0
        t.jacobian(y, x)


def test_batch_jacobian():
    with pytest.raises(NotImplementedError):
        with GradientTape() as t:
            x = Tensor(1.0)
            t.watch(x)
            y = x * 2.0
        t.batch_jacobian(y, x)


from zero_tensorflow import custom_gradient, stop_gradient, hessians


def test_custom_gradient():
    with pytest.raises(NotImplementedError):

        @custom_gradient
        def foo(x):
            return x


def test_stop_gradient():
    with pytest.raises(NotImplementedError):
        stop_gradient(Tensor(1.0))


def test_hessians():
    with pytest.raises(NotImplementedError):
        hessians(Tensor(1.0), [Tensor(1.0)])
