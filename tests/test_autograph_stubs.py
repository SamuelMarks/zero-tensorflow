"""Unit tests for `test_autograph_stubs`."""

from __future__ import annotations

import enum
from collections.abc import Callable

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


def _sample_if_while(x: int) -> int:
    """Test function containing `if`/`else` and `while` statements for AutoGraph lowering.

    Args:
        x: Input integer.

    Returns:
        int: Computed output.
    """
    while x < 10:
        x = x + 1
    while x < 5:
        pass
    if x > 20:
        x = x - 1
    if x > 0:
        return x * 2
    else:
        return -x


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

    dec: Callable[[Callable[[int], int]], Callable[[int], int]] = experimental.do_not_convert()  # type: ignore[assignment]
    assert callable(dec)

    def _no_conv_fn(a: int) -> int:
        """Function excluded from AutoGraph conversion.

        Args:
            a: Input value.

        Returns:
            int: Value `a` or `-a`.
        """
        if a > 0:
            return a
        else:
            return -a

    dec(_no_conv_fn)
    assert experimental.do_not_convert(_no_conv_fn) is _no_conv_fn
    assert "tf.cond" not in autograph.to_code(_no_conv_fn)
    nc_graph = autograph.to_graph(_no_conv_fn)
    assert nc_graph is not None and nc_graph(5) == 5
    nc_builtin = autograph.to_graph(experimental.do_not_convert(int))
    assert nc_builtin is not None and nc_builtin("3") == 3

    assert experimental.set_loop_options() is None

    autograph.set_verbosity(0, alsologtostderr=False)
    autograph.trace("silent_msg")
    autograph.set_verbosity(1, alsologtostderr=True)
    autograph.trace("msg")

    code = autograph.to_code(_sample_add)
    assert "def _sample_add" in code
    lowered_code = autograph.to_code(_sample_if_while)
    assert "tf.cond" in lowered_code
    assert "tf.while_loop" in lowered_code
    assert autograph.to_code(None) == ""
    assert autograph.to_code(int) == ""

    converted = autograph.to_graph(_sample_add)
    assert converted is not None
    assert converted(3, 4) == 7
    assert converted.graph is not None
    assert "def _sample_add" in converted.source_code
    assert autograph.to_graph(None) is None
