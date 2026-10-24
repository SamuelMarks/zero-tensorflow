"""Generate `snapshots/tf_api.json` from official TensorFlow or `ml-ecosystem-snapshots`."""

from __future__ import annotations

import inspect
import json
import os
import types
from collections.abc import Mapping

from zero_tensorflow._typing import TensorOperand

from typing_extensions import TypeAlias

ApiTree: TypeAlias = "dict[str, str | dict[str, ApiTree]]"


def safe_name(obj: types.ModuleType | type | TensorOperand) -> str:
    """Return the `__name__` of `obj` if present, else an empty string.

    Args:
        obj: Candidate module, class, or object.

    Returns:
        str: Module or type name string.
    """
    if isinstance(obj, (types.ModuleType, type)):
        return obj.__name__
    return ""


def get_public_api(
    module: types.ModuleType,
    prefix: str = "tf",
    depth: int = 0,
) -> dict[str, ApiTree]:
    """Recursively introspect a live TensorFlow module into a nested API dictionary.

    Args:
        module: Live module to introspect.
        prefix: Current dotted API path prefix.
        depth: Current recursion depth (capped at `2`).

    Returns:
        dict[str, ApiTree]: Nested dictionary of public symbols and their types.
    """
    if depth > 2:
        return {}
    api: dict[str, ApiTree] = {}
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if inspect.ismodule(obj):
            mod_name = safe_name(obj)
            if mod_name.startswith("tensorflow") or mod_name.startswith("keras"):
                api[name] = {
                    "type": "module",
                    "contents": get_public_api(obj, f"{prefix}.{name}", depth + 1),
                }
        elif inspect.isclass(obj):
            api[name] = {"type": "class"}
        elif callable(obj):
            api[name] = {"type": "function"}
        else:
            api[name] = {"type": "value"}
    return api


def build_api_from_ecosystem_snapshot(
    snapshot_file: str,
) -> dict[str, ApiTree]:
    """Build a nested `tf_api.json` tree from an `ml-ecosystem-snapshots` v0.0.4 file.

    Args:
        snapshot_file: Path to `tensorflow_v*.json` from `ml-ecosystem-snapshots`.

    Returns:
        dict[str, ApiTree]: Nested API tree constructed from GhostRef `api_path` entries.
    """
    with open(snapshot_file, encoding="utf-8") as f:
        data = json.load(f)

    root: dict[str, ApiTree] = {}
    categories = data.get("categories", {})
    if not isinstance(categories, Mapping):
        return root

    for items in categories.values():
        for item in items:
            api_path = item.get("api_path")
            kind = item.get("kind", "function")
            if not isinstance(api_path, str) or not api_path.startswith("tf."):
                continue
            parts = api_path.split(".")[1:]
            curr = root
            for part in parts[:-1]:
                sub_contents: dict[str, ApiTree] = {}
                node = curr.setdefault(part, {"type": "module", "contents": sub_contents})
                existing = node.get("contents")
                curr = existing if isinstance(existing, dict) else sub_contents
            leaf = parts[-1]
            curr[leaf] = {"type": str(kind) if kind in ("class", "function") else "value"}

    return root


def generate_snapshot(
    output_path: str = "snapshots/tf_api.json",
    ecosystem_snapshot_path: str | None = None,
) -> dict[str, ApiTree]:
    """Generate and write the TensorFlow API snapshot JSON file.

    Args:
        output_path: Destination JSON file path.
        ecosystem_snapshot_path: Optional path to an `ml-ecosystem-snapshots` JSON file.

    Returns:
        dict[str, ApiTree]: Generated API dictionary.
    """
    default_eco = ecosystem_snapshot_path or os.path.expanduser(
        "~/.cache/ml_ecosystem_snapshots/tensorflow_v2.21.0.json"
    )
    if ecosystem_snapshot_path is not None and os.path.exists(ecosystem_snapshot_path):
        api = build_api_from_ecosystem_snapshot(ecosystem_snapshot_path)
    else:
        try:
            import importlib

            tf_mod = importlib.import_module("tensorflow")
            if not tf_mod.__name__.startswith("tensorflow"):
                raise TypeError("Invalid tensorflow module")
            api = get_public_api(tf_mod)
        except (ImportError, TypeError):
            api = (
                build_api_from_ecosystem_snapshot(default_eco)
                if os.path.exists(default_eco)
                else {}
            )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(api, f, indent=2)
    return api


if __name__ == "__main__":
    generate_snapshot()
