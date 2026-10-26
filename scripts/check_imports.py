"""Verify that source, script, and non-parity test files only import allowed dependencies."""

from __future__ import annotations

import ast
import glob
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path

ALLOWED_3RD_PARTY: set[str] = {
    "cdd",
    "cdd_python",
    "ml_ecosystem_snapshots",
    "ml_switcheroo_compiler",
    "ml_switcheroo_ir",
    "pydantic",
    "typing_extensions",
    "zero_keras",
}

ALLOWED_TEST_MODULES: set[str] = {
    "pytest",
    "scripts",
    "tests",
}

STDLIB: set[str] = {
    "__future__",
    "abc",
    "argparse",
    "ast",
    "asyncio",
    "builtins",
    "collections",
    "concurrent",
    "contextlib",
    "copy",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "glob",
    "gzip",
    "hashlib",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "pprint",
    "queue",
    "random",
    "re",
    "shutil",
    "socket",
    "sqlite3",
    "string",
    "struct",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "traceback",
    "types",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "zlib",
}


def is_stdlib(module_name: str) -> bool:
    """Check whether `module_name` belongs to the Python standard library.

    Args:
        module_name: Top-level module name to inspect.

    Returns:
        bool: `True` if `module_name` is a standard library module.
    """
    if module_name in sys.builtin_module_names or module_name in STDLIB:
        return True

    spec = importlib.util.find_spec(module_name)
    if spec is not None and spec.origin is not None:
        return not ("site-packages" in spec.origin or "dist-packages" in spec.origin)

    return False


def check_module(base_module: str, is_test_or_script: bool = False) -> bool:
    """Determine whether `base_module` is an allowed import.

    Args:
        base_module: Top-level package/module name.
        is_test_or_script: Whether the file being checked is in `tests/` or `scripts/`.

    Returns:
        bool: `True` if the module is permitted.
    """
    if base_module == "zero_tensorflow" or base_module.startswith("zero_"):
        return True
    if base_module in ALLOWED_3RD_PARTY:
        return True
    if is_test_or_script and base_module in ALLOWED_TEST_MODULES:
        return True
    return is_stdlib(base_module)


def _is_dynamic_import_call(node: ast.Call) -> bool:
    """Return True if `node` represents a dynamic import or `sys.modules.get` call.

    Args:
        node: AST Call node to inspect.

    Returns:
        bool: True if the call dynamically resolves or imports a module.
    """
    if isinstance(node.func, ast.Name) and node.func.id == "__import__":
        return True
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr in ("__import__", "import_module"):
        return True
    return (
        node.func.attr == "get"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "modules"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "sys"
    )


def _is_sys_modules_subscript(node: ast.Subscript) -> bool:
    """Return True if `node` is a `sys.modules[...]` load or store subscript.

    Args:
        node: AST Subscript node to inspect.

    Returns:
        bool: True if `node` reads from or writes to `sys.modules[...]`.
    """
    return (
        isinstance(node.value, ast.Attribute)
        and node.value.attr == "modules"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "sys"
        and isinstance(node.ctx, (ast.Load, ast.Store))
    )


def check_file(filepath: str) -> list[tuple[int, str]]:
    """Scan a single Python source file for disallowed third-party imports or reflection.

    Args:
        filepath: Path to the Python file to analyze.

    Returns:
        list[tuple[int, str]]: List of `(lineno, module_name)` violations.
    """
    with open(filepath, encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return [(0, "SyntaxError")]

    disallowed: list[tuple[int, str]] = []
    parts = Path(filepath).parts
    is_test_or_script = "tests" in parts or "scripts" in parts

    def verify_module(lineno: int, name: str | None) -> None:
        """Record a violation if `name` is None or not an allowed module.

        Args:
            lineno: Line number of the import statement or expression.
            name: Imported module name, or None if unresolved dynamic import.
        """
        if name is None:
            disallowed.append((lineno, "dynamic_import"))
            return

        base_module = name.split(".")[0]
        if not check_module(base_module, is_test_or_script=is_test_or_script):
            disallowed.append((lineno, name))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                verify_module(node.lineno, alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:
                verify_module(node.lineno, node.module)
        elif isinstance(node, ast.Call) and _is_dynamic_import_call(node):
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                verify_module(node.lineno, node.args[0].value)
            else:
                verify_module(node.lineno, None)
        elif isinstance(node, ast.Subscript) and _is_sys_modules_subscript(node):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                verify_module(node.lineno, node.slice.value)
            elif isinstance(node.slice, ast.Name) and node.slice.id == "__name__":
                verify_module(node.lineno, "zero_tensorflow")
            else:
                verify_module(node.lineno, None)
        elif isinstance(node, ast.Attribute) and node.attr == "__dict__":
            disallowed.append((node.lineno, "__dict__"))
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "builtins"
            and node.attr == "object"
        ):
            disallowed.append((node.lineno, "builtins.object"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("vars", "dir", "eval", "exec")
        ):
            disallowed.append((node.lineno, node.func.id))
        elif isinstance(node, ast.Name) and node.id in (
            "Any",
            "object",
            "cast",
            "getattr",
            "setattr",
            "hasattr",
            "delattr",
        ):
            disallowed.append((node.lineno, node.id))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in (
            "__getattr__",
            "__setattr__",
            "__getattribute__",
        ):
            disallowed.append((node.lineno, node.name))

    return disallowed


def main(
    search_pattern: str | Sequence[str] = (
        "src/**/*.py",
        "scripts/**/*.py",
        "tests/**/*.py",
    ),
) -> int:
    """Check all matching source, script, and test files and return exit code `0` or `1`.

    Args:
        search_pattern: Glob pattern or sequence of glob patterns of Python files to check.

    Returns:
        int: `0` if all imports are allowed, `1` if disallowed imports exist.
    """
    has_errors = False
    patterns = [search_pattern] if isinstance(search_pattern, str) else list(search_pattern)
    matched_files: list[str] = []
    for pat in patterns:
        matched_files.extend(sorted(glob.glob(pat, recursive=True)))

    for filepath in sorted(set(matched_files)):
        errors = check_file(filepath)
        if errors:
            has_errors = True
            for lineno, module in errors:
                print(
                    f"{filepath}:{lineno} Disallowed import: '{module}'. "
                    "Only standard library and allowed 3rd-party dependencies are permitted."
                )

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
