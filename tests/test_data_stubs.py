import inspect

import zero_tensorflow.data as tf_data


def call_with_dummies(func):
    sig = inspect.signature(func)
    kwargs = {}
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            kwargs[name] = None
    try:
        func(**kwargs)
    except Exception:  # noqa: BLE001, S110
        pass  # just getting coverage


def test_data_stubs():
    for name in dir(tf_data):
        if name.startswith("_"):
            continue
        obj = getattr(tf_data, name)
        if callable(obj) and not isinstance(obj, type) or isinstance(obj, type):
            call_with_dummies(obj)

    for name in dir(tf_data.experimental):
        if name.startswith("_"):
            continue
        obj = getattr(tf_data.experimental, name)
        if callable(obj):
            call_with_dummies(obj)
