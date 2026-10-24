"""Rebuild `math`, `linalg`, `bitwise`, and top-level op definitions from `math`."""

from __future__ import annotations

import ast
import copy
import json
from collections.abc import Mapping

from typing_extensions import TypeAlias

SnapshotEntry: TypeAlias = "Mapping[str, str | Mapping[str, SnapshotEntry]]"

_OVERRIDES: dict[str, list[str]] = {
    "all": ["math.reduce_all", "reduce_all"],
    "any": ["math.reduce_any", "reduce_any"],
    "arange": ["range"],
    "binary": [],
    "bitwise_not": ["bitwise.invert"],
    "broadcast_to": ["broadcast_to"],
    "cast": ["cast"],
    "concatenate": ["concat"],
    "creation": [],
    "empty": [],
    "expand": ["expand_dims"],
    "flatten": ["reshape"],
    "float_power": ["math.pow", "pow"],
    "full": ["fill"],
    "gather": ["gather"],
    "gather_nd": ["gather_nd"],
    "identity": ["identity"],
    "inv": ["linalg.inv"],
    "left_shift": ["bitwise.left_shift"],
    "linalg": [],
    "linspace": ["linspace"],
    "matrix_power": [],
    "ones": ["ones"],
    "ones_like": ["ones_like"],
    "power": ["math.pow", "pow"],
    "prod": ["math.reduce_prod", "reduce_prod"],
    "reductions": [],
    "repeat": ["repeat"],
    "reshape": ["reshape"],
    "right_shift": ["bitwise.right_shift"],
    "roll": ["roll"],
    "scatter_nd": ["scatter_nd"],
    "shape": ["shape"],
    "slice": ["slice"],
    "split": ["split"],
    "squeeze": ["squeeze"],
    "stack": ["stack"],
    "std": ["math.reduce_std"],
    "strided_slice": ["strided_slice"],
    "sum": ["math.reduce_sum", "reduce_sum"],
    "tile": ["tile"],
    "transpose": ["transpose"],
    "unary": [],
    "unsqueeze": ["expand_dims"],
    "unstack": ["unstack"],
    "variance": ["math.reduce_variance"],
    "where": ["where"],
    "zeros": ["zeros"],
    "zeros_like": ["zeros_like"],
}


def _get_subcontents(
    api: Mapping[str, SnapshotEntry],
    mod_name: str,
) -> Mapping[str, SnapshotEntry]:
    """Return the `contents` mapping for `mod_name` in `api`.

    Args:
        api: Loaded `tf_api.json` snapshot dictionary.
        mod_name: Submodule name (`'math'`, `'linalg'`, `'bitwise'`).

    Returns:
        Mapping[str, SnapshotEntry]: Submodule contents mapping.
    """
    entry = api.get(mod_name)
    contents = entry.get("contents") if entry is not None else None
    return contents if isinstance(contents, Mapping) else {}


def get_tf_paths(name: str, api: Mapping[str, SnapshotEntry]) -> list[str]:
    """Determine target TensorFlow API paths for a `math` operation `name`.

    Args:
        name: Operation method name on `math`.
        api: Loaded `tf_api.json` snapshot dictionary.

    Returns:
        list[str]: Target relative paths (e.g., `['math.abs', 'abs']`).
    """
    if name in _OVERRIDES:
        return list(_OVERRIDES[name])

    paths: list[str] = []
    if name in api:
        paths.append(name)
    for submod in ("math", "linalg", "bitwise"):
        if name in _get_subcontents(api, submod):
            paths.append(f"{submod}.{name}")
    return paths


def _rename_method_ast(
    node: ast.FunctionDef,
    func_name: str,
    is_top_level: bool,
) -> str:
    """Clone `node`, rename it to `func_name`, and unparse it to source code.

    Args:
        node: Original AST `FunctionDef` node.
        func_name: New function/method name.
        is_top_level: If True, strip `@staticmethod` decorators and return top-level source.

    Returns:
        str: Formatted Python source code for the function or method.
    """
    cloned = copy.deepcopy(node)
    cloned.name = func_name
    nl = chr(10)
    if is_top_level:
        cloned.decorator_list = []
        return ast.unparse(cloned) + nl
    unparsed = ast.unparse(cloned)
    return nl.join(f"    {line}" if line else "" for line in unparsed.splitlines()) + nl


def rebuild_init_source(
    source: str,
    api: Mapping[str, SnapshotEntry],
) -> str:
    """Rebuild the `math`, `linalg`, `bitwise`, and top-level ops section of `source`.

    Args:
        source: Full Python source code of `src/zero_tensorflow/math_ops.py` or `__init__.py`.
        api: Loaded `tf_api.json` snapshot dictionary.

    Returns:
        str: Updated Python source code.
    """
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)

    math_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "math"),
        None,
    )
    if math_class is None:
        return source

    start_line = math_class.lineno - 1
    end_line = math_class.end_lineno or math_class.lineno

    method_nodes: dict[str, ast.FunctionDef] = {}
    for node in math_class.body:
        if isinstance(node, ast.FunctionDef):
            method_nodes[node.name] = node

    out_modules: dict[str, dict[str, str]] = {
        "math": {},
        "linalg": {},
        "bitwise": {},
        "top_level": {},
    }

    nl = chr(10)
    for name, m_node in method_nodes.items():
        for path in get_tf_paths(name, api):
            parts = path.split(".")
            mod = "top_level" if len(parts) == 1 else parts[0]
            func_name = parts[0] if len(parts) == 1 else parts[1]
            out_modules[mod][func_name] = _rename_method_ast(
                m_node,
                func_name,
                is_top_level=(mod == "top_level"),
            )

    class_blocks: list[str] = []
    for mod_name in ("math", "linalg", "bitwise"):
        if out_modules[mod_name]:
            header = f'class {mod_name}:{nl}    """TensorFlow `{mod_name}` namespace."""{nl}{nl}'
            class_blocks.append(header + nl.join(out_modules[mod_name].values()))

    new_classes = f"{nl}{nl}".join(class_blocks)
    new_top = nl.join(out_modules["top_level"].values())

    before_math = "".join(lines[:start_line])
    after_math = "".join(lines[end_line:])
    return f"{before_math}{new_classes}{nl}{nl}{new_top}{nl}{after_math}"


def main(
    snapshot_path: str = "snapshots/tf_api.json",
    init_path: str = "src/zero_tensorflow/math_ops.py",
    write: bool = False,
) -> str:
    """Load `init_path` and `snapshot_path`, rebuild the source, and optionally write it.

    Args:
        snapshot_path: Path to `snapshots/tf_api.json`.
        init_path: Path to `src/zero_tensorflow/math_ops.py`.
        write: Whether to overwrite `init_path` on disk.

    Returns:
        str: Rebuilt Python source string.
    """
    with open(snapshot_path, encoding="utf-8") as f:
        api: Mapping[str, SnapshotEntry] = json.load(f)
    with open(init_path, encoding="utf-8") as f:
        source = f.read()

    rebuilt = rebuild_init_source(source, api)
    if write:
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(rebuilt)
    print("Done rebuilding.")
    return rebuilt


if __name__ == "__main__":
    main(write=True)
