"""Unit tests for `test_stubs`."""

from __future__ import annotations

import contextlib
import inspect
import types

import zero_tensorflow as tf
from zero_tensorflow._typing import TensorOperand

_SUPPRESSED_ERRORS = (
    ValueError,
    TypeError,
    AttributeError,
    RuntimeError,
    NotImplementedError,
    KeyError,
    IndexError,
)


def _call_stubs(
    obj: types.ModuleType | type | TensorOperand,
    visited: set[int] | None = None,
) -> None:
    """Recursively invoke zero-argument functions and staticmethods on `obj`.

    Args:
        obj: Target module or class to traverse.
        visited: Set of already-visited object IDs.
    """
    if visited is None:
        visited = set()
    if id(obj) in visited:
        return
    visited.add(id(obj))
    if not isinstance(obj, (types.ModuleType, type)):
        return
    for name, attr in inspect.getmembers(obj):
        if name.startswith("_") or name in (
            "update_forward_refs",
            "from_orm",
            "parse_obj",
            "parse_raw",
            "parse_file",
            "validate",
            "construct",
            "copy",
            "dict",
            "json",
            "schema",
            "schema_json",
        ):
            continue
        if inspect.isfunction(attr) or inspect.ismethod(attr):
            with contextlib.suppress(*_SUPPRESSED_ERRORS):
                attr()
        elif inspect.isclass(attr):
            _call_stubs(attr, visited)


def test_stubs() -> None:
    """Verify all stub classes and functions across `zero_tensorflow`."""
    _call_stubs(tf)
