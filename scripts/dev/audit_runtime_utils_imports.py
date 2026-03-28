#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Audit runtime imports that depend on shared project utils modules."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path

TARGETS = [
    "run_dl.py",
    "exp",
    "data_provider",
    "models",
    "layers",
]


def iter_python_files(repo_root: Path):
    for target in TARGETS:
        target_path = repo_root / target
        if target_path.is_file():
            yield target_path
            continue
        for file_path in sorted(target_path.rglob("*.py")):
            yield file_path


def scan_utils_imports(repo_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for file_path in iter_python_files(repo_root):
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "utils" or alias.name.startswith("utils."):
                        records.append(
                            {
                                "file": str(file_path.relative_to(repo_root)),
                                "module": alias.name,
                                "names": alias.asname or "*",
                            }
                        )
            elif isinstance(node, ast.ImportFrom):
                if not node.module:
                    continue
                if node.module == "utils" or node.module.startswith("utils."):
                    imported_names = ", ".join(alias.name for alias in node.names)
                    records.append(
                        {
                            "file": str(file_path.relative_to(repo_root)),
                            "module": node.module,
                            "names": imported_names,
                        }
                    )
    return records


def print_summary(records: list[dict[str, str]]):
    by_module: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_module[record["module"]].add(record["file"])

    print("Runtime utils import audit")
    print("==========================")
    print(f"Files with runtime utils imports: {len({record['file'] for record in records})}")
    print(f"Unique utils modules referenced: {len(by_module)}")
    print("")
    for module in sorted(by_module):
        print(f"- {module}: {len(by_module[module])} file(s)")


def print_full_report(records: list[dict[str, str]]):
    print("| File | Utils Module | Imported Names |")
    print("| --- | --- | --- |")
    for record in records:
        print(f"| `{record['file']}` | `{record['module']}` | `{record['names']}` |")


def main():
    parser = argparse.ArgumentParser(description="Audit runtime imports from shared project utils modules.")
    parser.add_argument("--summary", action="store_true", help="print a compact summary instead of the full table")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    records = scan_utils_imports(repo_root)

    if args.summary:
        print_summary(records)
    else:
        print_summary(records)
        print("")
        print_full_report(records)


if __name__ == "__main__":
    main()
