"""Verify API parity between `zero_tensorflow` and official TensorFlow snapshots."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import types
from collections.abc import Mapping, Sequence

import zero_tensorflow as ztf
from zero_tensorflow._typing import TensorOperand

import zero_keras
from ml_ecosystem_snapshots.cli.resolver import resolve_snapshot_path
from ml_switcheroo_ir.schema.ghost import GhostRef, SnapshotEnvelope
from typing_extensions import TypeAlias

SnapshotNode: TypeAlias = "Mapping[str, str | Mapping[str, SnapshotNode]]"

_SKIP_CLASS_FIELDS: set[str] = {
    "__composite_gradient__",
    "__init__",
    "__nonzero__",
    "__slots__",
    "args",
    "kwargs",
}

_IGNORED_SYMBOLS: set[str] = {
    "ArrayIterator",
    "ArrayLikeProtocol",
    "AttributeValue",
    "Callable",
    "CallableOperandProtocol",
    "CondBranchProtocol",
    "CustomGradientFnProtocol",
    "DTypeLike",
    "DatasetElement",
    "DatasetMapFnProtocol",
    "Future",
    "GradFnProtocol",
    "IntTuple",
    "Iterable",
    "Mapping",
    "NestedSequence",
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
    "TensorCompatible",
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
    "WhileBodyProtocol",
    "WhileCondProtocol",
    "array_ops",
    "as_operand",
    "as_tensor_obj",
    " autodiff",
    "bitwise_ops",
    "builtins",
    "contextlib",
    "control_flow_ops",
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
    "spec_types",
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
                    if sub_name.startswith("_") or sub_name in (
                        "pow",
                        "dawsn",
                        "fresnel_cos",
                        "Fresnel_cos",
                        "fresnel_sin",
                        "spence",
                    ):
                        continue
                    if callable(sub_obj) and sub_name not in contents:
                        errors.append(f"Missing in official TF: {path}.{name}.{sub_name}")
            elif tf_info.get("type") == "module":
                errors.extend(check_parity(obj, contents, f"{path}.{name}"))

    return errors


def check_signature_parity(
    ztf_obj: types.ModuleType | type,
    tf_snapshot: Mapping[str, SnapshotNode],
    path: str = "tf",
) -> dict[str, int]:
    """Audit callable signatures and return annotations against `tf_snapshot`.

    Args:
        ztf_obj: `zero_tensorflow` module or namespace class.
        tf_snapshot: Dictionary representing official TensorFlow API hierarchy.
        path: Dotted API path prefix.

    Returns:
        dict[str, int]: Summary metrics (`total_callables`, `typed_returns`,
            `none_return_stubs`, `explicit_params`, `varargs_only_stubs`).
    """
    metrics: dict[str, int] = {
        "total_callables": 0,
        "typed_returns": 0,
        "none_return_stubs": 0,
        "explicit_params": 0,
        "varargs_only_stubs": 0,
    }
    members = _get_module_or_class_members(ztf_obj)
    for name, obj in members:
        if name not in tf_snapshot:
            continue

        tf_info = tf_snapshot[name]
        contents_raw = tf_info.get("contents")
        contents: Mapping[str, SnapshotNode] = (
            contents_raw if isinstance(contents_raw, Mapping) else {}
        )

        if (inspect.ismodule(obj) or inspect.isclass(obj)) and tf_info.get("type") == "module":
            sub_metrics = check_signature_parity(obj, contents, f"{path}.{name}")
            for k, v in sub_metrics.items():
                metrics[k] += v
        elif callable(obj) and not inspect.isclass(obj):
            metrics["total_callables"] += 1
            sig = inspect.signature(obj)
            ret = sig.return_annotation
            if ret in ("None", None):
                metrics["none_return_stubs"] += 1
            else:
                metrics["typed_returns"] += 1
            param_kinds = [p.kind for p in sig.parameters.values()]
            if param_kinds == [
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ]:
                metrics["varargs_only_stubs"] += 1
            else:
                metrics["explicit_params"] += 1

    return metrics


def _resolve_v005_snapshot_file(cache_path: str | None = None) -> str:
    """Resolve the path to `tensorflow_v2.21.0.json` from `ml-ecosystem-snapshots` (`v0.0.5`).

    Args:
        cache_path: Optional explicit path to a `tensorflow_v*.json` snapshot file.

    Returns:
        str: Resolved filesystem path to `tensorflow_v2.21.0.json`.
    """
    if cache_path is not None:
        return cache_path
    default_cache = os.path.expanduser("~/.cache/ml_ecosystem_snapshots/tensorflow_v2.21.0.json")
    return (
        default_cache
        if os.path.exists(default_cache)
        else resolve_snapshot_path("tensorflow_v2.21.0.json", must_exist=False)
    )


def load_ecosystem_snapshot_paths(
    cache_path: str | None = None,
) -> set[str]:
    """Load `api_path` entries from `ml-ecosystem-snapshots` (v0.0.5) if available.

    Args:
        cache_path: Optional explicit path to a `tensorflow_v*.json` snapshot file.

    Returns:
        set[str]: Set of ground-truth `tf.*` API paths from `ml-ecosystem-snapshots`.
    """
    resolved_path = _resolve_v005_snapshot_file(cache_path)
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


def parse_todo_unchecked(todo_plan_path: str = "TODO_PLAN.md") -> set[str]:
    """Extract unchecked `- [ ]` canonical `tf.*` API paths from `TODO_PLAN.md`.

    Args:
        todo_plan_path: Path to `TODO_PLAN.md`.

    Returns:
        set[str]: Set of `tf.*` API paths marked `- [ ]` in `TODO_PLAN.md`.
    """
    if not os.path.exists(todo_plan_path):
        return set()

    unchecked: set[str] = set()
    with open(todo_plan_path, encoding="utf-8") as f:
        for line in f:
            if not re.match(r"^\s*-\s*\[\s\]\s+", line):
                continue
            for match in re.finditer(r"`(tf\.[A-Za-z0-9_\.]+)`", line):
                unchecked.add(match.group(1))
    return unchecked


def resolve_tf_symbol(
    root: types.ModuleType | type,
    api_path: str,
) -> tuple[bool, types.ModuleType | type | TensorOperand]:
    """Resolve a dotted `tf.*` API path on `root` without `getattr` reflection.

    Args:
        root: Root `zero_tensorflow` module or synthetic test module.
        api_path: Dotted `tf.*` API path string.

    Returns:
        tuple[bool, types.ModuleType | type | TensorOperand]: `(resolved, target_obj)`.
    """
    parts = api_path.split(".")
    if parts[0] != "tf":
        return False, None
    cur: types.ModuleType | type | TensorOperand = root
    for part in parts[1:]:
        members = dict(inspect.getmembers(cur))
        if part not in members:
            return False, None
        cur = members[part]
    return True, cur


def verify_v005_ref(
    root: types.ModuleType | type,
    ref: GhostRef,
) -> list[str]:
    """Verify a single `GhostRef` from `ml-ecosystem-snapshots` (`v0.0.5`) against `root`.

    Args:
        root: Root `zero_tensorflow` module.
        ref: Hydrated `GhostRef` specification from `tensorflow_v2.21.0.json`.

    Returns:
        list[str]: List of violation descriptions (empty if compliant).
    """
    ok, target = resolve_tf_symbol(root, ref.api_path)
    if not ok:
        return [f"Missing minimum v0.0.5 API: {ref.api_path}"]

    violations: list[str] = []
    is_cls = inspect.isclass(target)
    if ref.kind == "class" and not is_cls:
        return [f"{ref.api_path}: expected class, got {type(target).__name__}"]
    if not callable(target):
        return [f"{ref.api_path}: expected callable ({ref.kind}), got {type(target).__name__}"]

    if not inspect.getdoc(target):
        violations.append(f"{ref.api_path}: missing docstring")

    for alias in ref.aliases or ():
        if not alias.startswith("tf.keras"):
            alias_ok, _ = resolve_tf_symbol(root, alias)
            if not alias_ok:
                violations.append(f"{ref.api_path}: missing alias {alias}")

    cls_members = dict(inspect.getmembers(target)) if is_cls else {}
    sig = inspect.signature(target)
    act_params = {k: v for k, v in sig.parameters.items() if k not in ("self", "cls")}

    for exp_p in ref.params:
        pname = exp_p.name
        if pname in ("self", "cls") or pname in _SKIP_CLASS_FIELDS:
            continue
        if pname in act_params:
            act_p = act_params[pname]
            if act_p.annotation is inspect.Parameter.empty:
                violations.append(f"{ref.api_path}({pname}): missing type annotation")
            if (
                exp_p.default is not None
                and exp_p.kind.value not in ("VAR_POSITIONAL", "VAR_KEYWORD")
                and act_p.default is inspect.Parameter.empty
            ):
                violations.append(f"{ref.api_path}({pname}): missing default value")
        elif is_cls and pname in cls_members:
            continue
        else:
            violations.append(f"{ref.api_path}: missing parameter/field '{pname}'")

    return violations


def check_v005_minimum_compliance(
    ztf_obj: types.ModuleType | type = ztf,
    ecosystem_snapshot_path: str | None = None,
    todo_plan_path: str = "TODO_PLAN.md",
) -> tuple[int, int, list[str]]:
    """Validate guaranteed minimum API existence and signatures against `ml-ecosystem-snapshots` (`v0.0.5`).

    Additional helper functions, classes, and modules in `zero_tensorflow` are allowed,
    and `tf.keras.*` is delegated to `../zero-keras`.

    Args:
        ztf_obj: `zero_tensorflow` root module to validate.
        ecosystem_snapshot_path: Optional path to `tensorflow_v2.21.0.json`.
        todo_plan_path: Path to `TODO_PLAN.md` tracking pending `- [ ]` signature migrations.

    Returns:
        tuple[int, int, list[str]]: `(conformant_count, total_non_keras_count, violations)`.
    """
    resolved_snap = _resolve_v005_snapshot_file(ecosystem_snapshot_path)
    if not os.path.exists(resolved_snap):
        return 0, 0, []

    with open(resolved_snap, encoding="utf-8") as f:
        raw_data = json.load(f)

    envelope = SnapshotEnvelope.model_validate(raw_data)
    unchecked = parse_todo_unchecked(todo_plan_path)

    violations: list[str] = []
    root_members = dict(inspect.getmembers(ztf_obj))
    if "keras" in root_members and root_members["keras"] is not zero_keras:
        violations.append("tf.keras must delegate directly to zero_keras")

    seen_paths: set[str] = set()
    conformant = 0
    total = 0

    for refs in envelope.categories.values():
        for raw_ref in refs:
            ref = GhostRef.model_validate(raw_ref)
            if ref.api_path in seen_paths or ref.api_path.startswith("tf.keras"):
                continue
            seen_paths.add(ref.api_path)
            total += 1
            if ref.api_path in unchecked:
                continue
            ref_errs = verify_v005_ref(ztf_obj, ref)
            if ref_errs:
                violations.extend(ref_errs)
            else:
                conformant += 1

    return conformant, total, violations


def main(
    snapshot_path: str = "snapshots/tf_api.json",
    argv: Sequence[str] | None = None,
) -> int:
    """Run the API parity check and return `0` on success or `1` on failure.

    Args:
        snapshot_path: Default path to the JSON API snapshot file.
        argv: Optional CLI argument list (`--check`, `--snapshot`).

    Returns:
        int: Exit code (`0` for success, `1` for failure).
    """
    check_mode = False
    resolved_snapshot = snapshot_path
    if argv is not None:
        parser = argparse.ArgumentParser(description="Verify TensorFlow API and signature parity.")
        parser.add_argument(
            "--check",
            action="store_true",
            help="Also report signature and return-annotation parity.",
        )
        parser.add_argument(
            "--snapshot",
            default=snapshot_path,
            help="Path to snapshots/tf_api.json.",
        )
        parsed = parser.parse_args(list(argv))
        check_mode = bool(parsed.check)
        resolved_snapshot = str(parsed.snapshot)

    try:
        with open(resolved_snapshot, encoding="utf-8") as f:
            tf_snapshot: Mapping[str, SnapshotNode] = json.load(f)
    except FileNotFoundError:
        print(f"{resolved_snapshot} not found. Run generate_api_snapshot.py first.")
        return 1

    errors = check_parity(ztf, tf_snapshot)
    if resolved_snapshot == "snapshots/tf_api.json":
        _, _, v005_violations = check_v005_minimum_compliance(ztf)
        errors.extend(v005_violations)

    if check_mode:
        sig_stats = check_signature_parity(ztf, tf_snapshot)
        print(
            "Signature Parity Summary: "
            f"total={sig_stats['total_callables']}, "
            f"typed_returns={sig_stats['typed_returns']}, "
            f"none_return_stubs={sig_stats['none_return_stubs']}, "
            f"explicit_params={sig_stats['explicit_params']}, "
            f"varargs_only_stubs={sig_stats['varargs_only_stubs']}"
        )

    if errors:
        print(f"API Parity check failed with {len(errors)} errors:")
        for err in errors:
            print("  " + err)
        return 1

    print("API Parity check passed. All zero-tensorflow APIs exist in official TensorFlow.")
    return 0


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv[1:]))
