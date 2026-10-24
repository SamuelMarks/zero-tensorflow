"""Compute test, documentation, and API parity metrics and synchronize `README.md` badges."""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys
from collections.abc import Sequence


def get_color(pct: float) -> str:
    """Return the Shields.io badge color corresponding to percentage `pct`.

    Args:
        pct: Coverage or parity percentage (`0.0` to `100.0`).

    Returns:
        str: Shields.io color name.
    """
    if pct >= 100.0:
        return "brightgreen"
    if pct >= 90.0:
        return "green"
    if pct >= 80.0:
        return "yellowgreen"
    if pct >= 70.0:
        return "yellow"
    if pct >= 60.0:
        return "orange"
    return "red"


def format_cov(cov: float) -> str:
    """Format a coverage percentage float into a compact badge string.

    Args:
        cov: Coverage percentage (`0.0` to `100.0`).

    Returns:
        str: Integer string if whole number, else 1-decimal string.
    """
    if int(cov) == cov:
        return str(int(cov))
    return f"{cov:.1f}"


def get_test_coverage(cov_json_path: str = "coverage.json") -> float:
    """Compute or read test coverage percentage from `coverage.json`.

    Args:
        cov_json_path: Path to the JSON coverage report.

    Returns:
        float: Test coverage percentage (`0.0` to `100.0`).
    """
    if not os.path.exists(cov_json_path):
        return 100.0
    try:
        with open(cov_json_path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "totals" not in data:
            return 0.0
        totals = data["totals"]
        if not isinstance(totals, dict) or "percent_covered" not in totals:
            return 0.0
        pct = totals["percent_covered"]
        return float(pct) if isinstance(pct, (int, float)) else 0.0
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0


def get_doc_coverage(paths: Sequence[str] = ("src", "scripts", "tests")) -> float:
    """Compute AST docstring coverage percentage across `paths`.

    Args:
        paths: Directories or file paths to scan for Python docstrings.

    Returns:
        float: Documentation coverage percentage (`0.0` to `100.0`).
    """
    total = 0
    covered = 0
    files: list[str] = []
    for p in paths:
        if os.path.isfile(p) and p.endswith(".py"):
            files.append(p)
        elif os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "*.py"), recursive=True)))

    for filepath in files:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        if os.path.basename(filepath) == "__init__.py" and not content.strip():
            continue
        try:
            tree = ast.parse(content, filename=filepath)
        except SyntaxError:
            total += 1
            continue

        total += 1
        if ast.get_docstring(tree) is not None:
            covered += 1

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                total += 1
                if ast.get_docstring(node) is not None:
                    covered += 1

    if total == 0:
        return 100.0
    return (covered / total) * 100.0


def get_official_test_pass_rate(report_path: str = ".report.json") -> float:
    """Read the official reference test pass rate from `report_path` if present.

    Args:
        report_path: Path to a pytest JSON report file.

    Returns:
        float: Pass rate percentage (`0.0` to `100.0`).
    """
    try:
        if not os.path.exists(report_path):
            return 100.0
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        total = int(summary.get("total", 0))
        passed = int(summary.get("passed", 0))
        if total <= 0:
            return 100.0
        return (passed / total) * 100.0
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0


def render_readme_badges(
    content: str,
    test_cov: float,
    doc_cov: float,
    official_cov: float,
) -> str:
    """Replace or insert coverage and parity badges in `README.md` markdown content.

    Args:
        content: Original `README.md` text.
        test_cov: Test coverage percentage.
        doc_cov: Documentation coverage percentage.
        official_cov: Official API parity percentage.

    Returns:
        str: Updated `README.md` text with current badges.
    """
    test_str = format_cov(test_cov)
    doc_str = format_cov(doc_cov)
    official_str = format_cov(official_cov)

    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)
    official_color = get_color(official_cov)

    test_re = re.compile(
        r"\[?\!\[Test Coverage\]\(https://img\.shields\.io/badge/(?:[tT]est_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = test_re.sub(
        f"[![Test Coverage](https://img.shields.io/badge/test_coverage-{test_str}%25-{test_color}.svg)](#)",
        content,
    )

    doc_re = re.compile(
        r"\[?\!\[Doc Coverage\]\(https://img\.shields\.io/badge/(?:[dD]oc_)?(?:[cC]overage)-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    content = doc_re.sub(
        f"[![Doc Coverage](https://img.shields.io/badge/doc_coverage-{doc_str}%25-{doc_color}.svg)](#)",
        content,
    )

    official_re = re.compile(
        r"\[?\!\[Official API Parity\]\(https://img\.shields\.io/badge/[oO]fficial_[aA]pi_[pP]arity-[0-9.]+%25-[a-z]+\.svg\)\]?(?:\(#\))?"
    )
    official_badge = f"[![Official API Parity](https://img.shields.io/badge/official_api_parity-{official_str}%25-{official_color}.svg)](#)"
    if official_re.search(content):
        content = official_re.sub(official_badge, content)
    else:
        nl = chr(10)
        lines = content.split(nl)
        for i, line in enumerate(lines):
            if "[![Doc Coverage]" in line:
                lines.insert(i + 1, official_badge)
                break
        content = nl.join(lines)

    return content


def update_readme(
    readme_path: str = "README.md",
    cov_json_path: str = "coverage.json",
    report_path: str = ".report.json",
) -> bool:
    """Update `readme_path` with current coverage and parity badges.

    Args:
        readme_path: Path to `README.md`.
        cov_json_path: Path to `coverage.json`.
        report_path: Path to `.report.json`.

    Returns:
        bool: `True` if `README.md` was already up to date, `False` if modified.
    """
    if not os.path.exists(readme_path):
        return True

    test_cov = get_test_coverage(cov_json_path)
    doc_cov = get_doc_coverage()
    official_cov = get_official_test_pass_rate(report_path)

    with open(readme_path, encoding="utf-8") as f:
        original = f.read()

    updated = render_readme_badges(original, test_cov, doc_cov, official_cov)
    if updated != original:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(updated)
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for updating or enforcing `README.md` badges.

    Args:
        argv: Optional list of CLI arguments.

    Returns:
        int: `0` on success, `1` if `--enforce` fails due to stale or sub-100% badges.
    """
    parser = argparse.ArgumentParser(description="Update README coverage badges.")
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Verify badges without silently passing on <100% coverage.",
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        help="Path to README.md file.",
    )
    parser.add_argument(
        "--cov-json",
        default="coverage.json",
        help="Path to coverage.json file.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    test_cov = get_test_coverage(str(args.cov_json))
    doc_cov = get_doc_coverage()
    unchanged = update_readme(readme_path=str(args.readme), cov_json_path=str(args.cov_json))

    if args.enforce and (test_cov < 100.0 or doc_cov < 100.0 or not unchanged):
        print(
            f"Coverage enforcement failed: test={test_cov:.1f}%, doc={doc_cov:.1f}%, unchanged={unchanged}"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
