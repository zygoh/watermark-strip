"""remove-ai-watermarks 封装：进程内常驻引擎，批量复用同一 pipeline。"""

from __future__ import annotations

import os
import time
from pathlib import Path

from app.engine import get_runtime
from app.log import log_info


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
    if wait_ready:
        runtime.start_preload()
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

    total = len(files)
    log_info(f"批量处理 {total} 张 → {out_dir}")

    results: list[dict] = []
    for index, src in enumerate(files, start=1):
        dst = out_dir / src.name
        log_info(f"({index}/{total}) 开始 {src.name} …")
        t0 = time.monotonic()
        meta = strip_image(src, dst, **kwargs)
        elapsed = time.monotonic() - t0
        log_info(
            f"({index}/{total}) 完成 {src.name} "
            f"({elapsed:.0f}s, strength={meta.get('strength')})"
        )
        results.append(meta)
    return results


def batch_strip_directories(
    source_dirs: list[Path | str],
    *,
    wait_ready: bool = True,
    **kwargs,
) -> dict[str, list[dict]]:
    """多目录串行处理：只预加载一次，各目录仍写入各自 cleaned/。"""
    dirs = [Path(d).resolve() for d in source_dirs]
    if not dirs:
        raise ValueError("source_dirs is empty")

    log_info(f"共 {len(dirs)} 个目录待处理（GPU 串行，勿并行起第二个进程）")
    by_dir: dict[str, list[dict]] = {}
    for index, source_dir in enumerate(dirs, start=1):
        log_info(f"===== 目录 ({index}/{len(dirs)}) {source_dir} =====")
        by_dir[str(source_dir)] = batch_strip_directory(
            source_dir,
            wait_ready=wait_ready and index == 1,
            **kwargs,
        )
    return by_dir
