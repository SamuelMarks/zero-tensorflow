import ast
import glob
import sys

# Allowed 3rd party dependencies based on the rule
ALLOWED_3RD_PARTY = {
    "pydantic",
    "cdd",
    "ml_switcheroo_ir",
    "ml_switcheroo_compiler",
    "zero_keras",
}

# Standard library modules (a reasonably comprehensive set for typical use cases to avoid false positives)
import sysconfig

stdlib_modules = set(sys.builtin_module_names) | set(sysconfig.get_paths()["stdlib"])


def is_stdlib(module_name):
    if module_name in sys.builtin_module_names:
        return True

    # Try using imp or importlib to see if it's stdlib
    try:
        import importlib.util

        spec = importlib.util.find_spec(module_name)
        if spec is not None and spec.origin is not None:
            return not (
                "site-packages" in spec.origin or "dist-packages" in spec.origin
            )
    except Exception:  # noqa: BLE001, S110
        pass

    return False


# Hardcoded stdlib for Python to be safe
STDLIB = {
    "abc",
    "argparse",
    "ast",
    "asyncio",
    "collections",
    "contextlib",
    "copy",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "glob",
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
    "random",
    "re",
    "shutil",
    "socket",
    "sqlite3",
    "string",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "traceback",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "builtins",
}


def check_file(filepath):
    with open(filepath, "r") as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return []

    disallowed = []

    def verify_module(lineno, name):
        if name is None:
            disallowed.append((lineno, "dynamic_import"))
            return

        base_module = name.split(".")[0]
        if not check_module(base_module):
            disallowed.append((lineno, name))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                verify_module(node.lineno, name.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:  # not relative
                verify_module(node.lineno, node.module)
        elif isinstance(node, ast.Call):
            if (
                (isinstance(node.func, ast.Name) and node.func.id == "__import__")
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "__import__"
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "modules"
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "sys"
                )
            ):
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    verify_module(node.lineno, node.args[0].value)
                else:
                    verify_module(node.lineno, None)

        elif (
            isinstance(node, ast.Subscript)
            and (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "modules"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "sys"
            )
            and isinstance(node.ctx, ast.Load)
        ):
            if isinstance(node.slice, ast.Constant) and isinstance(
                node.slice.value, str
            ):
                verify_module(node.lineno, node.slice.value)
            else:
                verify_module(node.lineno, None)

    return disallowed


def check_module(base_module):
    # Local project modules
    if base_module in ["zero_tensorflow"]:
        return True

    if base_module in STDLIB:
        return True

    if base_module in ALLOWED_3RD_PARTY:
        return True

    # Check dynamically if it's stdlib
    return bool(is_stdlib(base_module))


def main():
    has_errors = False

    src_files = glob.glob("src/**/*.py", recursive=True)

    for filepath in src_files:
        errors = check_file(filepath)
        if errors:
            has_errors = True
            for lineno, module in errors:
                print(
                    f"{filepath}:{lineno} Disallowed import: '{module}'. Only standard library and allowed 3rd-party dependencies are permitted in non-test code."
                )

    if has_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
