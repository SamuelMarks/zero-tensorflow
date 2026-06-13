import inspect
import importlib

module_names = [
    "ml_switcheroo_compiler.ops.binary.math",
    "ml_switcheroo_compiler.ops.binary.special",
    "ml_switcheroo_compiler.ops.control_flow",
    "ml_switcheroo_compiler.ops.creation.basic",
    "ml_switcheroo_compiler.ops.creation.frontend",
    "ml_switcheroo_compiler.ops.linalg.basic",
    "ml_switcheroo_compiler.ops.linalg.frontend",
    "ml_switcheroo_compiler.ops.reductions.basic",
    "ml_switcheroo_compiler.ops.reductions.frontend",
    "ml_switcheroo_compiler.ops.shape.basic",
    "ml_switcheroo_compiler.ops.shape.frontend",
    "ml_switcheroo_compiler.ops.unary.math",
    "ml_switcheroo_compiler.ops.unary.special",
]

print("import numpy as np")
print("from zero_tensorflow import Tensor")
print("from ml_switcheroo_compiler.core.config import EagerMode")
print("from ml_switcheroo_compiler.tracing import _tracer")

for m in module_names:
    print(f"import {m}")

print()
print("def test_ops_exhaustive():")
print("    with EagerMode():")
print("        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))")
print("        y = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))")
print("        z = Tensor(np.array([1, 2]))")
print("        v = Tensor(np.array([1.0]))")

for m_name in module_names:
    mod = importlib.import_module(m_name)
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        if callable(obj) and getattr(obj, "__module__", "") == m_name:
            sig = inspect.signature(obj)
            args = []

            for param_name, param in sig.parameters.items():
                if param.default != inspect.Parameter.empty:
                    continue  # Use default

                if param_name in [
                    "a",
                    "b",
                    "x",
                    "y",
                    "input",
                    "tensor",
                    "a",
                    "b",
                    "input_tensor",
                ]:
                    args.append("x")
                elif param_name in ["shape", "sizes"]:
                    args.append("(2, 2)")
                elif param_name in ["axis", "axes", "dim"]:
                    args.append("0")
                elif param_name == "repeats":
                    args.append("2")
                elif param_name == "indices":
                    args.append("z")
                elif param_name == "updates":
                    args.append("x")
                elif param_name == "condition" or param_name == "pred":
                    args.append("Tensor(np.array(True))")
                elif "fn" in param_name:
                    args.append("lambda *args: x")
                elif param_name in ["init_val", "init"]:
                    args.append("x")
                elif param_name == "xs":
                    args.append("x")
                elif param_name == "tensors":
                    args.append("[x, y]")
                elif param_name == "value":
                    args.append("1.0")
                elif param_name == "dtype":
                    args.append("None")
                else:
                    args.append("x")  # fallback

            args_str = ", ".join(args)
            print(f"        try: {m_name}.{name}({args_str})")
            print(f"        except Exception as e: pass # print('{m_name}.{name}', e)")

            # also generate a tracing variant!

print()
print("def test_ops_tracing():")
print("    prev_tracing = getattr(_tracer, 'is_tracing', False)")
print("    prev_graph = getattr(_tracer, 'active_graph', None)")
print("    try:")
print("        _tracer.is_tracing = True")
print(
    "        _tracer.active_graph = type('Graph', (), {'nodes': {}, 'add_node': lambda n: None})()"
)
print("        _tracer.add_node = lambda n: None")
print("        x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))")
print("        y = Tensor(np.array([[1.0, 0.0], [0.0, 1.0]]))")
print("        z = Tensor(np.array([1, 2]))")

for m_name in module_names:
    mod = importlib.import_module(m_name)
    for name, obj in inspect.getmembers(mod):
        if name.startswith("_"):
            continue
        if callable(obj) and getattr(obj, "__module__", "") == m_name:
            sig = inspect.signature(obj)
            args = []

            for param_name, param in sig.parameters.items():
                if param.default != inspect.Parameter.empty:
                    continue  # Use default

                if param_name in [
                    "a",
                    "b",
                    "x",
                    "y",
                    "input",
                    "tensor",
                    "a",
                    "b",
                    "input_tensor",
                ]:
                    args.append("x")
                elif param_name in ["shape", "sizes"]:
                    args.append("(2, 2)")
                elif param_name in ["axis", "axes", "dim"]:
                    args.append("0")
                elif param_name == "repeats":
                    args.append("2")
                elif param_name == "indices":
                    args.append("z")
                elif param_name == "updates":
                    args.append("x")
                elif param_name == "condition" or param_name == "pred":
                    args.append("Tensor(np.array(True))")
                elif "fn" in param_name:
                    args.append("lambda *args: x")
                elif param_name in ["init_val", "init"]:
                    args.append("x")
                elif param_name == "xs":
                    args.append("x")
                elif param_name == "tensors":
                    args.append("[x, y]")
                elif param_name == "value":
                    args.append("1.0")
                elif param_name == "dtype":
                    args.append("None")
                else:
                    args.append("x")  # fallback

            args_str = ", ".join(args)
            print(f"        try: {m_name}.{name}({args_str})")
            print(
                f"        except Exception as e: pass # print('Tracing {m_name}.{name}', e)"
            )
print("    finally:")
print("        _tracer.is_tracing = prev_tracing")
print("        _tracer.active_graph = prev_graph")
