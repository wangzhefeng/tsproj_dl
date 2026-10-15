#!/usr/bin/env python3
"""静态盘点运行代码的绝对 utils 导入；不导入重依赖模型，也不声称依赖可安装。"""
import argparse
import ast
from collections import defaultdict
from pathlib import Path


TARGETS = ("run.py", "exp", "data_provider", "models", "layers")


def iter_python_files(repo_root):
    for target in TARGETS:
        path = repo_root / target
        if not path.exists():
            raise FileNotFoundError(f"审计目标缺失: {path}")
        if path.is_file():
            yield path
        else:
            yield from sorted(path.rglob("*.py"))


def scan_utils_imports(repo_root):
    records = []
    for path in iter_python_files(repo_root):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [(alias.name, alias.asname or "*") for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports = [(node.module, ", ".join(alias.name for alias in node.names))]
            else:
                continue
            for module, names in imports:
                if module == "utils" or module.startswith("utils."):
                    records.append({"file": path.relative_to(repo_root).as_posix(),
                                    "line": node.lineno, "module": module, "names": names})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="只显示按模块汇总的导入引用")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    records = scan_utils_imports(root)
    by_module = defaultdict(set)
    for record in records:
        by_module[record["module"]].add(record["file"])
    print("Runtime utils import audit (static references only)")
    print(f"Files with utils imports: {len({r['file'] for r in records})}")
    print(f"Unique utils modules referenced: {len(by_module)}")
    for module, files in sorted(by_module.items()):
        print(f"- {module}: {len(files)} file(s)")
    if not args.summary:
        print("\n| File:Line | Utils Module | Imported Names |")
        print("| --- | --- | --- |")
        for record in records:
            print(f"| {record['file']}:{record['line']} | {record['module']} | {record['names']} |")


if __name__ == "__main__":
    main()
