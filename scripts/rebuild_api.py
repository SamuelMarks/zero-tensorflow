"""Map `zero_tensorflow.math` operations onto top-level and submodule TensorFlow API paths."""

from __future__ import annotations

import ast
import copy
import json
from collections.abc import Mapping

from typing_extensions import TypeAlias

SnapshotEntry: TypeAlias = "Mapping[str, str | Mapping[str, SnapshotEntry]]"

_ALIAS_FALLBACKS: dict[str, str] = {
    "all": "reduce_all",
    "any": "reduce_any",
    "arange": "range",
    "concatenate": "concat",
    "dot": "tensordot",
    "float_power": "pow",
    "full": "fill",
    "inv": "linalg.inv",
    "power": "pow",
    "prod": "reduce_prod",
    "variance": "math.reduce_variance",
}

_SKIPPED_NAMES: set[str] = {
    "binary",
    "creation",
    "empty",
    "linalg",
    "matrix_power",
    "reductions",
    "unary",
}


def find_paths(
    obj: Mapping[str, SnapshotEntry],
    name: str,
    current_path: str = "tf",
    depth: int = 0,
) -> list[str]:
    """Recursively locate all API paths matching `name` within `obj`.

    Args:
        obj: Snapshot dictionary to search.
        name: Symbol name to find.
        current_path: Current dotted API prefix.
        depth: Current recursion depth (capped at `2`).

    Returns:
        list[str]: Matching dotted API paths.
    """
    if depth > 2:
        return []
    paths: list[str] = []
    for k, v in obj.items():
        if k == name:
            paths.append(f"{current_path}.{k}")
        contents = v.get("contents")
        if (
            v.get("type") == "module"
            and isinstance(contents, Mapping)
            and "keras" not in current_path
            and "compat" not in current_path
            and "experimental" not in current_path
        ):
            paths.extend(find_paths(contents, name, f"{current_path}.{k}", depth + 1))
    return paths


def extract_math_methods(init_path: str = "src/zero_tensorflow/math_ops.py") -> dict[str, str]:
    """Extract method source blocks from `class math` in `init_path`.

    Args:
        init_path: Path to `src/zero_tensorflow/math_ops.py`.

    Returns:
        dict[str, str]: Mapping from method name to unindented method source code.
    """
    with open(init_path, encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)
    math_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "math"),
        None,
    )
    if math_class is None:
        return {}

    methods: dict[str, str] = {}
    for node in math_class.body:
        if isinstance(node, ast.FunctionDef):
            methods[node.name] = ast.unparse(node)
    return methods


def _rename_func_source(source_code: str, func_name: str) -> str:
    """Rename a parsed function definition to `func_name` via AST transformation.

    Args:
        source_code: Original function source code.
        func_name: Target function name.

    Returns:
        str: Unparsed function source code with updated name.
    """
    mod_tree = ast.parse(source_code)
    first_node = mod_tree.body[0]
    if isinstance(first_node, ast.FunctionDef):
        cloned = copy.deepcopy(first_node)
        cloned.name = func_name
        return ast.unparse(cloned)
    return source_code


def build_api_mappings(
    snapshot_path: str = "snapshots/tf_api.json",
    init_path: str = "src/zero_tensorflow/math_ops.py",
) -> dict[str, dict[str, str]]:
    """Compute per-module method source mappings from `math` to TensorFlow namespaces.

    Args:
        snapshot_path: Path to `snapshots/tf_api.json`.
        init_path: Path to `src/zero_tensorflow/math_ops.py`.

    Returns:
        dict[str, dict[str, str]]: Mapping of `{module_name: {func_name: source_code}}`.
    """
    with open(snapshot_path, encoding="utf-8") as f:
        api: Mapping[str, SnapshotEntry] = json.load(f)

    methods = extract_math_methods(init_path)
    mappings: dict[str, dict[str, str]] = {}

    for name, source_code in methods.items():
        if name in _SKIPPED_NAMES:
            continue
        paths = find_paths(api, name)
        if not paths and name in _ALIAS_FALLBACKS:
            target = _ALIAS_FALLBACKS[name]
            paths = [f"tf.{target}"] if "." in target else find_paths(api, target)

        for p in paths:
            parts = p.split(".")
            module = "top_level" if len(parts) == 2 else parts[1]
            func_name = parts[1] if len(parts) == 2 else parts[2]

            mod_map = mappings.setdefault(module, {})
            mod_map[func_name] = _rename_func_source(source_code, func_name)

    return mappings


def main(
    snapshot_path: str = "snapshots/tf_api.json",
    init_path: str = "src/zero_tensorflow/math_ops.py",
) -> dict[str, int]:
    """Build API mappings and print summary counts per module.

    Args:
        snapshot_path: Path to `snapshots/tf_api.json`.
        init_path: Path to `src/zero_tensorflow/math_ops.py`.

    Returns:
        dict[str, int]: Count of mapped functions per module.
    """
    mappings = build_api_mappings(snapshot_path, init_path)
    counts = {k: len(v) for k, v in mappings.items()}
    print(counts)
    return counts


if __name__ == "__main__":
    main()
