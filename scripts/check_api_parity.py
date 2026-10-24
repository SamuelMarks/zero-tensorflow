"""Verify API parity between `zero_tensorflow` and official TensorFlow snapshots."""

from __future__ import annotations

import inspect
import json
import os
import sys
import types
from collections.abc import Mapping

import zero_tensorflow as ztf
from zero_tensorflow._typing import TensorOperand

from typing_extensions import TypeAlias

SnapshotNode: TypeAlias = "Mapping[str, str | Mapping[str, SnapshotNode]]"

_IGNORED_SYMBOLS: set[str] = {
    "ArrayIterator",
    "ArrayLikeProtocol",
    "AttributeValue",
    "Callable",
    "CallableOperandProtocol",
    "DTypeLike",
    "DatasetElement",
    "Future",
    "IntTuple",
    "Iterable",
    "Mapping",
    "NestedTensorData",
    "NumpyConvertibleProtocol",
    "Protocol",
    "ProxyTensor",
    "PyIterator",
    "ScalarNumeric",
    "ScalarValue",
    "Sequence",
    "ShapeLike",
    "ShapeTuple",
    "StringTensorLike",
    "TensorConfig",
    "TensorLike",
    "TensorOperand",
    "TensorOrScalar",
    "ThreadPoolExecutor",
    "TracebackType",
    "TracerTape",
    "TrackableContainerProtocol",
    "TrackableLike",
    "TwoTensorTuple",
    "TypeAlias",
    "TypeGuard",
    "TypeVar",
    "ValuesHolderProtocol",
    "as_operand",
    "as_tensor_obj",
    " autodiff",
    "bitwise_ops",
    "builtins",
    "contextlib",
    "core_stubs",
    "core_tensor",
    "data_experimental",
    "dt",
    "enum",
    "functools",
    "initializers",
    "keras",
    "k_activations",
    "k_ops",
    "legacy_mod",
    "linalg_ops",
    "losses",
    "math_ops",
    "metrics",
    "msc_core",
    "msc_dtype",
    "msc_ops",
    "optimizers",
    "os",
    "queue",
    "raw_ops",
    "re",
    "runtime_checkable",
    "sys",
    "threading",
    "to_dtype",
    "uuid",
    "zk_random",
}


def _get_module_or_class_members(
    target: types.ModuleType | type,
) -> list[tuple[str, types.ModuleType | type | TensorOperand]]:
    """Extract public members from a module or class using `inspect.getmembers` and `__all__`.

    Args:
        target: Module or class object to inspect.

    Returns:
        list[tuple[str, types.ModuleType | type | TensorOperand]]: List of `(name, value)` pairs.
    """
    all_members = dict(inspect.getmembers(target))
    all_list = all_members.get("__all__")
    if isinstance(all_list, list):
        return [
            (str(name), all_members[name])
            for name in all_list
            if isinstance(name, str) and name in all_members
        ]
    return [(name, val) for name, val in all_members.items() if not name.startswith("_")]


def check_parity(
    ztf_obj: types.ModuleType | type,
    tf_snapshot: Mapping[str, SnapshotNode],
    path: str = "tf",
) -> list[str]:
    """Compare exported symbols on `ztf_obj` against `tf_snapshot`.

    Args:
        ztf_obj: `zero_tensorflow` module or namespace class.
        tf_snapshot: Dictionary representing official TensorFlow API hierarchy.
        path: Dotted API path prefix (e.g., `'tf'`).

    Returns:
        list[str]: List of parity error descriptions (empty if 100% compliant).
    """
    errors: list[str] = []
    members = _get_module_or_class_members(ztf_obj)

    for name, obj in members:
        if name.startswith("_") or name in _IGNORED_SYMBOLS:
            continue

        if inspect.ismodule(obj) and not obj.__name__.startswith("zero_tensorflow"):
            continue

        if name not in tf_snapshot:
            errors.append(f"Missing in official TF: {path}.{name}")
            continue

        tf_info = tf_snapshot[name]
        contents_raw = tf_info.get("contents")
        contents: Mapping[str, SnapshotNode] = (
            contents_raw if isinstance(contents_raw, Mapping) else {}
        )

        if inspect.ismodule(obj) or inspect.isclass(obj):
            if inspect.isclass(obj) and name in ("math", "nn", "data", "linalg", "bitwise"):
                for sub_name, sub_obj in inspect.getmembers(obj):
                    if sub_name.startswith("_") or sub_name == "pow":
                        continue
                    if callable(sub_obj) and sub_name not in contents:
                        errors.append(f"Missing in official TF: {path}.{name}.{sub_name}")
            elif tf_info.get("type") == "module":
                errors.extend(check_parity(obj, contents, f"{path}.{name}"))

    return errors


def load_ecosystem_snapshot_paths(
    cache_path: str | None = None,
) -> set[str]:
    """Load `api_path` entries from `ml-ecosystem-snapshots` (v0.0.4) if available.

    Args:
        cache_path: Optional explicit path to a `tensorflow_v*.json` snapshot file.

    Returns:
        set[str]: Set of ground-truth `tf.*` API paths from `ml-ecosystem-snapshots`.
    """
    resolved_path = cache_path or os.path.expanduser(
        "~/.cache/ml_ecosystem_snapshots/tensorflow_v2.21.0.json"
    )
    if not os.path.exists(resolved_path):
        return set()

    with open(resolved_path, encoding="utf-8") as f:
        data = json.load(f)

    api_paths: set[str] = set()
    categories = data.get("categories", {})
    for items in categories.values():
        for item in items:
            p = item.get("api_path")
            if isinstance(p, str):
                api_paths.add(p)
    return api_paths


def main(snapshot_path: str = "snapshots/tf_api.json") -> int:
    """Run the API parity check and return `0` on success or `1` on failure.

    Args:
        snapshot_path: Path to the JSON API snapshot file.

    Returns:
        int: Exit code (`0` for success, `1` for failure).
    """
    try:
        with open(snapshot_path, encoding="utf-8") as f:
            tf_snapshot: Mapping[str, SnapshotNode] = json.load(f)
    except FileNotFoundError:
        print(f"{snapshot_path} not found. Run generate_api_snapshot.py first.")
        return 1

    errors = check_parity(ztf, tf_snapshot)
    if errors:
        print(f"API Parity check failed with {len(errors)} errors:")
        for err in errors:
            print("  " + err)
        return 1

    print("API Parity check passed. All zero-tensorflow APIs exist in official TensorFlow.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
