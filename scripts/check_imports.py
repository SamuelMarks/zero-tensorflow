import ast
import sys
import glob

# Allowed 3rd party dependencies based on the rule
ALLOWED_3RD_PARTY = {
    "numpy",
    "pydantic",
    "cdd",  # for ../cdd-python, it's typically imported as cdd or similar?
    "ml_switcheroo_ir",  # ../ml-switcheroo-ir
    "ml_switcheroo",  # ../ml-switcheroo-compiler
    "zero_jax",  # ../zero-jax
    "zero_keras",  # ../zero-keras
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
            if "site-packages" in spec.origin or "dist-packages" in spec.origin:
                return False
            return True
    except Exception:
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

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                base_module = name.name.split(".")[0]
                if not check_module(base_module):
                    disallowed.append((node.lineno, name.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0:  # not relative
                base_module = node.module.split(".")[0]
                if not check_module(base_module):
                    disallowed.append((node.lineno, node.module))
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
    if is_stdlib(base_module):
        return True

    return False


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
