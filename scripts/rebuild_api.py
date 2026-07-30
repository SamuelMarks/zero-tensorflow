import ast
import json

with open("snapshots/tf_api.json") as f:
    api = json.load(f)


def find_paths(obj, name, current_path="tf", depth=0):
    if depth > 2:
        return []
    paths = []
    for k, v in obj.items():
        if k == name:
            paths.append(current_path + "." + k)
        if (
            v["type"] == "module"
            and "contents" in v
            and (
                "keras" not in current_path
                and "compat" not in current_path
                and "experimental" not in current_path
            )
        ):
            paths.extend(
                find_paths(v["contents"], name, current_path + "." + k, depth + 1)
            )
    return paths


# Get methods from zero_tensorflow/__init__.py
with open("src/zero_tensorflow/__init__.py") as f:
    source = f.read()

tree = ast.parse(source)
math_class = next(
    node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "math"
)

methods = {}
for node in math_class.body:
    if isinstance(node, ast.FunctionDef):
        # Extract the source code for this function
        start_line = node.lineno - 1
        # Account for decorators
        if node.decorator_list:
            start_line = node.decorator_list[0].lineno - 1
        end_line = node.end_lineno
        lines = source.splitlines()[start_line:end_line]
        # Re-indent (remove 4 spaces)
        method_source = "\n".join(line.removeprefix("    ") for line in lines)
        methods[node.name] = method_source

# Mapping
mappings = {}
for name, source_code in methods.items():
    paths = find_paths(api, name)
    if not paths:
        # Try finding aliases if any. E.g. all -> reduce_all
        if name == "all":
            paths = find_paths(api, "reduce_all")
        elif name == "any":
            paths = find_paths(api, "reduce_any")
        elif name == "arange":
            paths = find_paths(api, "range")
        elif name == "binary":
            paths = []
        elif name == "concatenate":
            paths = find_paths(api, "concat")
        elif name == "creation":
            paths = []
        elif name == "dot":
            paths = find_paths(api, "tensordot")
        elif name == "empty":
            paths = []
        elif name == "float_power":
            paths = find_paths(api, "pow")
        elif name == "full":
            paths = find_paths(api, "fill")
        elif name == "inv":
            paths = find_paths(api, "linalg.inv")
        elif name == "linalg" or name == "matrix_power":
            paths = []
        elif name == "power":
            paths = find_paths(api, "pow")
        elif name == "prod":
            paths = find_paths(api, "reduce_prod")
        elif name == "reductions" or name == "unary":
            paths = []
        elif name == "variance":
            paths = find_paths(api, "math.reduce_variance")

    for p in paths:
        parts = p.split(".")
        if len(parts) == 2:
            module = "top_level"
            func_name = parts[1]
        elif len(parts) == 3:
            module = parts[1]
            func_name = parts[2]
        else:
            continue

        if module not in mappings:
            mappings[module] = {}
        # Replace the function name in the source code if it's aliased
        new_source = source_code.replace(f"def {name}(", f"def {func_name}(")
        mappings[module][func_name] = new_source

print({k: len(v) for k, v in mappings.items()})
