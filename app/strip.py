"""remove-ai-watermarks 封装：进程内常驻引擎，批量复用同一 pipeline。"""

from __future__ import annotations

import os
from pathlib import Path

from app.engine import get_runtime


def strip_image(
    input_path: Path,
    output_path: Path,
    *,
    mode: str | None = None,
    device: str | None = None,
    max_resolution: int | None = None,
    pipeline: str = "controlnet",
    steps: int | None = None,
    strength: float | None = None,
) -> dict:
    """mode=all：可见 + 隐形(SynthID) + 元数据。"""
    result = get_runtime().strip_image(
        input_path,
        output_path,
        mode=mode,
        device=device,
        max_resolution=max_resolution,
        pipeline=pipeline,
        steps=steps,
        strength=strength,
    )
    if not result["ok"]:
        raise RuntimeError(f"strip failed: {input_path}")
    return result


def batch_strip_directory(
    source_dir: Path,
    *,
    output_subdir: str = "cleaned",
    patterns: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.webp"),
    wait_ready: bool = True,
    warm_timeout: float | None = None,
    **kwargs,
) -> list[dict]:
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)

    runtime = get_runtime()
    runtime.start_preload()
    if wait_ready:
        timeout = warm_timeout
        if timeout is None:
            timeout = float(os.getenv("WATERMARK_STRIP_WARM_TIMEOUT") or "900")
        runtime.wait_ready(timeout=timeout)

    out_dir = source_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(source_dir.glob(pat)))
    files = [f for f in files if f.parent == source_dir]

    if not files:
        raise FileNotFoundError(f"no images in {source_dir}")

    results: list[dict] = []
    for src in files:
        dst = out_dir / src.name
        results.append(strip_image(src, dst, **kwargs))
    return results
