import importlib


def test_reimport_with_units():
    import zero_keras as keras

    if not hasattr(keras.layers.Dense, "units"):
        keras.layers.Dense.units = property(lambda self: self._kwargs.get("units"))
    import zero_tensorflow

    importlib.reload(zero_tensorflow)
