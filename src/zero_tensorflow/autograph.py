"""TensorFlow AutoGraph module."""

from __future__ import annotations

import ast
import enum
import inspect
import textwrap
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

from zero_tensorflow._typing import TensorOperand

from ml_switcheroo_compiler.backends.ast_to_ir import parse_ast_to_ir
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.tracing.autograph import set_loop_options
from typing_extensions import ParamSpec

_P = ParamSpec("_P")
_R = TypeVar("_R")
F = TypeVar("F")

__all__ = [
    "experimental",
    "set_verbosity",
    "to_code",
    "to_graph",
    "trace",
]

_VERBOSITY_LEVEL: int = 0
_ALSOLOGTOSTDERR: bool = False
_TRACE_LOG: list[tuple[TensorOperand, ...]] = []
_DO_NOT_CONVERT_IDS: set[int] = set()


class _FeatureEnum(enum.Enum):
    """Enumeration of AutoGraph conversion feature flags."""

    ALL = "ALL"
    AUTO_CONTROL_DEPS = "AUTO_CONTROL_DEPS"
    ASSERT_STATEMENTS = "ASSERT_STATEMENTS"
    BUILTIN_FUNCTIONS = "BUILTIN_FUNCTIONS"
    EQUALITY_OPERATORS = "EQUALITY_OPERATORS"
    LISTS = "LISTS"
    NAME_SCOPES = "NAME_SCOPES"


class _AutoGraphNodeTransformer(ast.NodeTransformer):
    """AST transformer rewriting Python `if`/`else` and `while` into `tf.cond` and `tf.while_loop`."""

    def visit_If(self, node: ast.If) -> ast.AST:
        """Transform `if cond: return a else: return b` or simple assignments into `tf.cond`.

        Args:
            node: Input `ast.If` statement node.

        Returns:
            ast.AST: Rewritten AST node using `tf.cond` or visited `ast.If`.
        """
        self.generic_visit(node)
        if (
            len(node.body) == 1
            and len(node.orelse) == 1
            and isinstance(node.body[0], ast.Return)
            and isinstance(node.orelse[0], ast.Return)
            and node.body[0].value is not None
            and node.orelse[0].value is not None
        ):
            true_lambda = ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=node.body[0].value,
            )
            false_lambda = ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=node.orelse[0].value,
            )
            cond_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="tf", ctx=ast.Load()),
                    attr="cond",
                    ctx=ast.Load(),
                ),
                args=[node.test, true_lambda, false_lambda],
                keywords=[],
            )
            return ast.copy_location(ast.Return(value=cond_call), node)
        return node

    def visit_While(self, node: ast.While) -> ast.AST:
        """Transform single-variable `while cond: x = expr` into `x = tf.while_loop(...)`.

        Args:
            node: Input `ast.While` statement node.

        Returns:
            ast.AST: Rewritten AST node using `tf.while_loop` or visited `ast.While`.
        """
        self.generic_visit(node)
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Assign)
            and len(node.body[0].targets) == 1
            and isinstance(node.body[0].targets[0], ast.Name)
        ):
            var_name = node.body[0].targets[0].id
            cond_lambda = ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=var_name)],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=node.test,
            )
            body_lambda = ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg=var_name)],
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                ),
                body=ast.Tuple(elts=[node.body[0].value], ctx=ast.Load()),
            )
            loop_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="tf", ctx=ast.Load()),
                    attr="while_loop",
                    ctx=ast.Load(),
                ),
                args=[
                    cond_lambda,
                    body_lambda,
                    ast.List(elts=[ast.Name(id=var_name, ctx=ast.Load())], ctx=ast.Load()),
                ],
                keywords=[],
            )
            assign_stmt = ast.Assign(
                targets=[ast.Tuple(elts=[ast.Name(id=var_name, ctx=ast.Store())], ctx=ast.Store())],
                value=loop_call,
            )
            return ast.copy_location(assign_stmt, node)
        return node


class experimental:
    """Experimental AutoGraph APIs."""

    Feature = _FeatureEnum

    @staticmethod
    def do_not_convert(
        func: F | None = None,
    ) -> F | Callable[[F], F]:
        """Mark a function as excluded from AutoGraph conversion.

        Args:
            func: Optional target function to decorate.

        Returns:
            F | Callable[[F], F]: Decorated function or decorator callable.
        """
        if func is not None:
            _DO_NOT_CONVERT_IDS.add(id(func))
            return func

        def _decorator(f: F) -> F:
            """Register `f` in `_DO_NOT_CONVERT_IDS` and return `f`.

            Args:
                f: Function to exclude from AutoGraph conversion.

            Returns:
                F: Unmodified function `f`.
            """
            _DO_NOT_CONVERT_IDS.add(id(f))
            return f

        return _decorator

    @staticmethod
    def set_loop_options(
        parallel_iterations: int | None = None,
        swap_memory: bool | None = None,
        maximum_iterations: int | None = None,
        shape_invariants: tuple[int | None, ...] | None = None,
    ) -> None:
        """Specify loop options for the enclosing loop in AutoGraph.

        Args:
            parallel_iterations: Maximum number of parallel loop iterations.
            swap_memory: Whether to enable memory swapping.
            maximum_iterations: Hard limit on loop iterations.
            shape_invariants: Shape invariants for loop variables.
        """
        set_loop_options(
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            maximum_iterations=maximum_iterations,
            shape_invariants=shape_invariants,
        )


class _ConvertedEntity(Generic[_P, _R]):
    """Callable wrapper produced by `autograph.to_graph` holding the parsed `IRGraph`."""

    def __init__(
        self,
        fn: Callable[_P, _R],
        graph: IRGraph,
        source_code: str,
    ) -> None:
        """Initialize `_ConvertedEntity`.

        Args:
            fn: Underlying Python callable.
            graph: Parsed `IRGraph` from `ml_switcheroo_compiler`.
            source_code: Normalized Python source string.
        """
        self._fn = fn
        self.graph = graph
        self.source_code = source_code

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Invoke the converted callable.

        Args:
            *args: Positional arguments forwarded to the underlying callable.
            **kwargs: Keyword arguments forwarded to the underlying callable.

        Returns:
            _R: Result of calling `self._fn(*args, **kwargs)`.
        """
        return self._fn(*args, **kwargs)


def set_verbosity(level: int = 0, alsologtostderr: bool = False) -> None:
    """Set the AutoGraph verbosity level.

    Args:
        level: Integer verbosity level (`0` to `10`).
        alsologtostderr: Whether to also log messages to standard error.
    """
    global _VERBOSITY_LEVEL, _ALSOLOGTOSTDERR
    _VERBOSITY_LEVEL = level
    _ALSOLOGTOSTDERR = alsologtostderr


def to_code(
    entity: Callable[_P, _R] | type | None = None,
    recursive: bool = True,
    experimental_optional_features: (_FeatureEnum | Sequence[_FeatureEnum] | None) = None,
) -> str:
    """Return the source code generated by AutoGraph for `entity`.

    Args:
        entity: Python callable or class to convert.
        recursive: Whether to recursively convert functions called by `entity`.
        experimental_optional_features: Optional `Feature` flags to enable.

    Returns:
        str: Normalized Python source code string for `entity`.
    """
    del recursive, experimental_optional_features
    if entity is None:
        return ""
    try:
        raw_source = textwrap.dedent(inspect.getsource(entity))
    except (OSError, TypeError):
        return ""
    parsed_ast = ast.parse(raw_source)
    if id(entity) not in _DO_NOT_CONVERT_IDS:
        parsed_ast = ast.fix_missing_locations(_AutoGraphNodeTransformer().visit(parsed_ast))
    _ = parse_ast_to_ir(raw_source)
    return ast.unparse(parsed_ast)


def to_graph(
    entity: Callable[_P, _R] | None = None,
    recursive: bool = True,
    experimental_optional_features: (_FeatureEnum | Sequence[_FeatureEnum] | None) = None,
) -> _ConvertedEntity[_P, _R] | None:
    """Convert a Python entity into a TensorFlow graph-backed callable.

    Args:
        entity: Python callable to convert.
        recursive: Whether to recursively convert functions called by `entity`.
        experimental_optional_features: Optional `Feature` flags to enable.

    Returns:
        _ConvertedEntity[_P, _R] | None: Converted callable with `.graph` attribute, or `None`.
    """
    if entity is None or not callable(entity):
        return None
    if id(entity) in _DO_NOT_CONVERT_IDS:
        try:
            raw_src = textwrap.dedent(inspect.getsource(entity))
        except (OSError, TypeError):
            raw_src = ""
        return _ConvertedEntity(entity, IRGraph(), raw_src)
    code = to_code(
        entity,
        recursive=recursive,
        experimental_optional_features=experimental_optional_features,
    )
    ir_graph = parse_ast_to_ir(code) if code else IRGraph()
    return _ConvertedEntity(entity, ir_graph, code)


def trace(*args: TensorOperand) -> None:
    """Trace debugging information at execution time in AutoGraph.

    Args:
        *args: Arguments to trace.
    """
    if _VERBOSITY_LEVEL > 0:
        _TRACE_LOG.append(args)
    return None
