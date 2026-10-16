"""运行代码身份；哈希用于审计，不替代模型计算兼容合同。"""
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

from data_provider.spatial_temporal.schema import canonical_hash


ROOT = Path(__file__).resolve().parents[1]


def capture_source(root=ROOT):
    root = Path(root).resolve()
    patterns = ("run.py", "exp/exp_basic.py", "exp/*gslaformer*.py",
                "models/transformer/GSLAformer.py", "layers/gslaformer/*.py",
                "data_provider/spatial_temporal/*.py", "utils/*.py",
                "scripts/dev/*gslaformer*.py", "scripts/smoke/*gslaformer*",
                "pyproject.toml", "uv.lock")
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for pattern in patterns for p in sorted(root.glob(pattern)) if p.is_file()}
    git_sha, dirty, status = None, None, "unavailable"
    try:
        # 无 Git 的导出副本不能误认上层仓库；不读取或输出凭据/环境文件。
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        if Path(top).resolve() == root:
            git_sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                     capture_output=True, text=True, check=True, timeout=10).stdout.strip()
            dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                       capture_output=True, text=True, check=True, timeout=10).stdout.strip())
            status = "available"
    except (OSError, subprocess.SubprocessError):
        pass  # 导出源码无 Git 时仍以完整文件清单审计，不声称 clean。
    payload = {"version": 1, "git_sha": git_sha, "git_dirty": dirty, "git_status": status,
               "files": files, "source_fingerprint": canonical_hash(files)}
    return {**payload, "fingerprint": canonical_hash(payload)}


def validate_source(source):
    if (not isinstance(source, dict) or source.get("version") != 1
            or not isinstance(source.get("files"), dict) or not source["files"]
            or source.get("source_fingerprint") != canonical_hash(source["files"])
            or source.get("fingerprint") != canonical_hash({k: v for k, v in source.items() if k != "fingerprint"})):
        raise ValueError("源码溯源缺失或指纹错误")


def checkpoint_sources(saved):
    """旧计算版本 2 可读取，但不为旧训练虚构源码来源。"""
    if "provenance_version" not in saved:
        return {"training_source": {"status": "legacy_missing"}, "execution_history": []}
    if saved["provenance_version"] != 1:
        raise ValueError("未知源码溯源版本")
    source = saved.get("training_source")
    if source != {"status": "legacy_missing"}:
        validate_source(source)
    history = saved.get("execution_history")
    if not isinstance(history, list) or not history:
        raise ValueError("checkpoint 缺源码执行历史")
    for item in history:
        validate_source(item)
    return {"training_source": source, "execution_history": history}


def execution_environment(device, *, source=None, **extra):
    return {"provenance_version": 1, "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "device": str(device), "torch_threads": torch.get_num_threads(),
            "execution_source": capture_source() if source is None else source, **extra}
