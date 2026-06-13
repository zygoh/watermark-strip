"""从 .env 读取 strip 默认参数。"""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    return int(raw) if raw else default


def _env_float_optional(name: str) -> float | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    return float(raw)


def strip_settings() -> dict:
    return {
        "default_mode": (os.getenv("WATERMARK_STRIP_DEFAULT_MODE") or "all").strip(),
        "steps": _env_int("WATERMARK_STRIP_STEPS", 50),
        "strength": _env_float_optional("WATERMARK_STRIP_STRENGTH"),
        "adaptive_polish": _env_bool("WATERMARK_STRIP_ADAPTIVE_POLISH", True),
        "post_denoise": _env_bool("WATERMARK_STRIP_POST_DENOISE", True),
        "max_resolution": _env_int("WATERMARK_STRIP_MAX_RESOLUTION", 1536),
    }
