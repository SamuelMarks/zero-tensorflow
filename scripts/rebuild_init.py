import ast
import json

with open("snapshots/tf_api.json") as f:
    api = json.load(f)

with open("src/zero_tensorflow/__init__.py") as f:
    lines = f.readlines()
    source = "".join(lines)

tree = ast.parse(source)

math_class = None
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == "math":
        math_class = node
        break

start_line = math_class.lineno - 1
end_line = math_class.end_lineno

methods = {}
for node in math_class.body:
    if isinstance(node, ast.FunctionDef):
        m_start = (
            node.decorator_list[0].lineno - 1
            if node.decorator_list
            else node.lineno - 1
        )
        m_end = node.end_lineno
        method_lines = lines[m_start:m_end]
        methods[node.name] = "".join(method_lines)


def get_tf_paths(name):
    overrides = {
        "all": ["math.reduce_all", "reduce_all"],
        "any": ["math.reduce_any", "reduce_any"],
        "arange": ["range"],
        "binary": [],
        "bitwise_not": ["bitwise.invert"],
        "concatenate": ["concat"],
        "creation": [],
        "empty": [],
        "float_power": ["math.pow", "pow"],
        "full": ["fill"],
        "inv": ["linalg.inv"],
        "linalg": [],
        "matrix_power": [],
        "power": ["math.pow", "pow"],
        "prod": ["math.reduce_prod", "reduce_prod"],
        "reductions": [],
        "unary": [],
        "variance": ["math.reduce_variance"],
        "broadcast_to": ["broadcast_to"],
        "cast": ["cast"],
        "expand": ["expand_dims"],
        "unsqueeze": ["expand_dims"],
        "flatten": ["reshape"],
        "gather": ["gather"],
        "gather_nd": ["gather_nd"],
        "identity": ["identity"],
        "linspace": ["linspace"],
        "ones": ["ones"],
        "ones_like": ["ones_like"],
        "repeat": ["repeat"],
        "reshape": ["reshape"],
        "roll": ["roll"],
        "scatter_nd": ["scatter_nd"],
        "shape": ["shape"],
        "slice": ["slice"],
        "split": ["split"],
        "squeeze": ["squeeze"],
        "stack": ["stack"],
        "strided_slice": ["strided_slice"],
        "tile": ["tile"],
        "transpose": ["transpose"],
        "unstack": ["unstack"],
        "where": ["where"],
        "zeros": ["zeros"],
        "zeros_like": ["zeros_like"],
        "std": ["math.reduce_std"],
        "sum": ["math.reduce_sum", "reduce_sum"],
        "left_shift": ["bitwise.left_shift"],
        "right_shift": ["bitwise.right_shift"],
    }

    if name in overrides:
        return overrides[name]

    paths = []

    if name in api:
        paths.append(name)
    if name in api["math"]["contents"]:
        paths.append("math." + name)
    if name in api["linalg"]["contents"]:
        paths.append("linalg." + name)
    if name in api["bitwise"]["contents"]:
        paths.append("bitwise." + name)

    return paths


out_modules = {"math": {}, "linalg": {}, "bitwise": {}, "top_level": {}}

for name, method_src in methods.items():
    paths = get_tf_paths(name)

    for path in paths:
        parts = path.split(".")
        if len(parts) == 1:
            mod = "top_level"
            func_name = parts[0]
        else:
            mod = parts[0]
            func_name = parts[1]

        # Avoid overriding multiple definitions with the same target
        # E.g. 'power' and 'float_power' both map to 'pow'. We just keep the first one we process.
        if func_name in out_modules[mod]:
            continue

        new_src = method_src.replace(f"def {name}(", f"def {func_name}(")

        if mod == "top_level":
            new_src = new_src.replace("    @staticmethod\n", "")
            new_src = new_src.replace("@staticmethod\n", "")
            new_src = (
                "\n".join(
                    line[4:] if line.startswith("    ") else line
                    for line in new_src.splitlines()
                )
                + "\n"
            )
            out_modules["top_level"][func_name] = new_src
        else:
            if mod in out_modules:
                out_modules[mod][func_name] = new_src

new_math = (
    "class math:\n" + "".join(out_modules["math"].values())
    if out_modules["math"]
    else ""
)
new_linalg = (
    "class linalg:\n" + "".join(out_modules["linalg"].values())
    if out_modules["linalg"]
    else ""
)
new_bitwise = (
    "class bitwise:\n" + "".join(out_modules["bitwise"].values())
    if out_modules["bitwise"]
    else ""
)

new_classes = "\n\n".join(filter(None, [new_math, new_linalg, new_bitwise]))
new_top = "".join(out_modules["top_level"].values())

before_math = "".join(lines[:start_line])
after_math = "".join(lines[end_line:])

with open("src/zero_tensorflow/__init__.py", "w") as f:
    f.write(before_math)
    f.write(new_classes)
    f.write("\n\n")
    f.write(new_top)
    f.write(after_math)

print("Done rebuilding.")
