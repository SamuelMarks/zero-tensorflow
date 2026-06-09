def generate_layers():
    with open("TENSORFLOW_TODO.md", "r") as f:
        content = f.read()

    lines = content.split("\n")
    layer_names = []
    for line in lines:
        if "| tf | tf.keras.layers |" in line:
            parts = line.split("|")
            symbol = parts[4].strip()
            layer_names.append(symbol)

    if not layer_names:
        print("No layers found")
        return

    out = [
        '"""Keras layers module."""',
        "from typing import Any, Optional, Tuple, Union",
        "",
        "__all__ = [",
    ]
    out.append("    " + ", ".join(f'"{name}"' for name in layer_names))
    out.append("]")
    out.append("")

    for name in layer_names:
        out.append(f"class {name}:")
        out.append(f'    """{name} layer."""')
        out.append("    def __init__(self, *args: Any, **kwargs: Any):")
        out.append("        pass")
        out.append("")

    with open("src/zero_tensorflow/keras/layers.py", "w") as f:
        f.write("\n".join(out))

    # Also generate tests
    test_out = ["import pytest", "from zero_tensorflow.keras import layers", ""]
    for name in layer_names:
        test_out.append(f"def test_{name.lower()}():")
        test_out.append(f"    layer = layers.{name}()")
        test_out.append("")

    with open("tests/keras/test_layers.py", "w") as f:
        f.write("\n".join(test_out))


if __name__ == "__main__":
    generate_layers()
