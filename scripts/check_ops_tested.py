import json
import sys
import subprocess


def main():
    print("Checking test coverage for ml_switcheroo_compiler.ops...")
    # We must run pytest with coverage over ml_switcheroo_compiler.ops
    # We assume pytest has already been run and coverage.json is generated,
    # OR we run it here if coverage.json is not present or we want to be safe.

    # Actually, running pytest inside pre-commit can be slow.
    # Usually pytest-cov generates coverage.json if configured.
    subprocess.run(
        ["pytest", "--cov=ml_switcheroo_compiler.ops", "--cov-report=json", "tests/"],
        capture_output=True,
    )

    try:
        with open("coverage.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("coverage.json not found. Run pytest with --cov-report=json first.")
        sys.exit(1)

    ops_files = {
        k: v
        for k, v in data.get("files", {}).items()
        if "ml_switcheroo_compiler/ops" in k
    }

    if not ops_files:
        print("No coverage data found for ml_switcheroo_compiler.ops.")
        sys.exit(1)

    missing = False
    for filename, file_data in ops_files.items():
        # Ignore __init__.py and base.py which may not have 100% execution easily
        if (
            filename.endswith("__init__.py")
            or filename.endswith("base.py")
            or filename.endswith("state.py")
        ):
            continue

        cov_pct = file_data.get("summary", {}).get("percent_covered", 0.0)
        if cov_pct < 100.0:
            print(f"FAIL: {filename} has {cov_pct}% coverage. 100% is required.")
            missing = True

    if missing:
        print(
            "Error: The test suite does not have 100% coverage for ml-switcheroo-compiler operations."
        )
        sys.exit(1)

    print("SUCCESS: 100% coverage achieved for ml-switcheroo-compiler operations.")


if __name__ == "__main__":
    main()
