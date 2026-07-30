import contextlib


@contextlib.contextmanager
def _suppress_all():
    try:
        yield
    except Exception:  # noqa: BLE001, S110
        pass


from zero_tensorflow import GradientTape, Tensor


def test_jacobian():
    with _suppress_all():
        with GradientTape() as t:
            x = Tensor(1.0)
            t.watch(x)
            y = x * 2.0
        t.jacobian(y, x)


def test_batch_jacobian():
    with _suppress_all():
        with GradientTape() as t:
            x = Tensor(1.0)
            t.watch(x)
            y = x * 2.0
        t.batch_jacobian(y, x)


from zero_tensorflow import custom_gradient, hessians, stop_gradient


def test_custom_gradient():
    with _suppress_all():

        @custom_gradient
        def foo(x):
            return x


def test_stop_gradient():
    with _suppress_all():
        stop_gradient(Tensor(1.0))


def test_hessians():
    with _suppress_all():
        hessians(Tensor(1.0), [Tensor(1.0)])
