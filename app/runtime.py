"""解析 watermark-strip 运行时 Python（默认本项目 .venv）。"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")


@lru_cache(maxsize=1)
def python_executable() -> str:
    configured = (os.getenv("WATERMARK_STRIP_PYTHON") or "").strip()
    if configured:
        path = Path(configured)
        if path.is_file():
            return str(path.resolve())
        raise FileNotFoundError(f"WATERMARK_STRIP_PYTHON not found: {configured}")

    for candidate in (_ROOT / ".venv" / "Scripts" / "python.exe",):
        if candidate.is_file():
            return str(candidate.resolve())

    found = shutil.which("python")
    if found:
        return found
    raise RuntimeError("no Python runtime found; set WATERMARK_STRIP_PYTHON in .env")


def cli_executable() -> str:
    py = Path(python_executable())
    cli = py.parent / "remove-ai-watermarks.exe"
    if cli.is_file():
        return str(cli)
    return "remove-ai-watermarks"
