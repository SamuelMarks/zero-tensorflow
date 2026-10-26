"""Unit tests for `test_autograph_stubs`."""

from __future__ import annotations

import enum

from zero_tensorflow import autograph
from zero_tensorflow.autograph import experimental


def _sample_add(x: int, y: int) -> int:
    """Return `x + y` for AutoGraph AST inspection testing.

    Args:
        x: First integer operand.
        y: Second integer operand.

    Returns:
        int: Sum `x + y`.
    """
    return x + y


def test_autograph() -> None:
    """Verify `autograph.experimental.Feature` enum, `to_code`, `to_graph`, `set_verbosity`, and `trace`."""
    assert issubclass(experimental.Feature, enum.Enum)
    assert experimental.Feature.ALL.value == "ALL"
    assert experimental.Feature.AUTO_CONTROL_DEPS.value == "AUTO_CONTROL_DEPS"
    assert experimental.Feature.ASSERT_STATEMENTS.value == "ASSERT_STATEMENTS"
    assert experimental.Feature.BUILTIN_FUNCTIONS.value == "BUILTIN_FUNCTIONS"
    assert experimental.Feature.EQUALITY_OPERATORS.value == "EQUALITY_OPERATORS"
    assert experimental.Feature.LISTS.value == "LISTS"
    assert experimental.Feature.NAME_SCOPES.value == "NAME_SCOPES"

    assert experimental.do_not_convert() is not None
    assert experimental.set_loop_options() is None

    autograph.set_verbosity(1, alsologtostderr=True)
    autograph.trace("msg")

    code = autograph.to_code(_sample_add)
    assert "def _sample_add" in code
    assert autograph.to_code(None) == ""
    assert autograph.to_code(int) == ""

    converted = autograph.to_graph(_sample_add)
    assert converted is not None
    assert converted(3, 4) == 7
    assert converted.graph is not None
    assert "def _sample_add" in converted.source_code
    assert autograph.to_graph(None) is None
