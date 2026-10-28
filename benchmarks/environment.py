"""Run-wide facts recorded in every results header, so a result can be traced to what produced it."""

import platform
import subprocess
from datetime import UTC, datetime
from typing import Any

import torch


def _run(*cmd: str) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata() -> dict[str, Any]:
    commit = _run("git", "rev-parse", "--short", "HEAD")
    dirty = bool(_run("git", "status", "--porcelain", "--untracked-files=no"))
    gpu = _run("nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit", "--format=csv,noheader")
    driver, max_sm_clock, power_limit = gpu.splitlines()[0].split(", ") if gpu else (None, None, None)
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": f"{commit}{'-dirty' if dirty else ''}" if commit else None,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "driver": driver,
        "max_sm_clock": max_sm_clock,
        "power_limit": power_limit,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }
