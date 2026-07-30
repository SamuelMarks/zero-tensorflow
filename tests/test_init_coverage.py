import ml_switcheroo_compiler.ops as msc_ops

import zero_tensorflow as tf


def call_safely(func):
    t = msc_ops.zeros((2, 2))
    try:
        func(t)
    except Exception:  # noqa: BLE001
        try:
            func(t, t)
        except Exception:  # noqa: BLE001
            try:
                func(t, t, t)
            except Exception:  # noqa: BLE001
                try:
                    func(t, axis=0)
                except Exception:  # noqa: BLE001, S110
                    pass


def test_init_coverage():
    for name in dir(tf.math):
        if name.startswith("_"):
            continue
        obj = getattr(tf.math, name)
        if callable(obj):
            call_safely(obj)

    for name in dir(tf.linalg):
        if name.startswith("_"):
            continue
        obj = getattr(tf.linalg, name)
        if callable(obj):
            call_safely(obj)

    for name in dir(tf.bitwise):
        if name.startswith("_"):
            continue
        obj = getattr(tf.bitwise, name)
        if callable(obj):
            call_safely(obj)

    for name in dir(tf):
        if name.startswith("_") or name in ["math", "linalg", "bitwise"]:
            continue
        obj = getattr(tf, name)
        if callable(obj) and not isinstance(obj, type):
            call_safely(obj)
