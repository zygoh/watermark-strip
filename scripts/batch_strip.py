#!/usr/bin/env python3
"""批量对目录内图片做隐形移除（all = visible + invisible + metadata）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.runtime import python_executable  # noqa: E402
from app.log import log_info
from app.strip import batch_strip_directories, batch_strip_directory  # noqa: E402

if Path(python_executable()) != Path(sys.executable):
    print(
        f"hint: run with configured python:\n"
        f'  "{python_executable()}" "{Path(__file__).resolve()}" ...',
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch strip AI watermarks from one or more directories")
    parser.add_argument(
        "source_dirs",
        type=Path,
        nargs="+",
        help="one or more directories containing source images (processed serially on GPU)",
    )
    parser.add_argument("--output-subdir", default="cleaned", help="subdir for cleaned images")
    parser.add_argument("--mode", default="all", choices=["all", "invisible", "visible", "metadata"])
    parser.add_argument("--device", default=None, help="cuda | cpu (default: env or cuda)")
    parser.add_argument("--max-resolution", type=int, default=1536)
    parser.add_argument("--pipeline", default="controlnet", choices=["controlnet", "sdxl"])
    parser.add_argument(
        "--no-wait-ready",
        action="store_true",
        help="do not block until diffusion pipeline is preloaded",
    )
    args = parser.parse_args()

    try:
        if len(args.source_dirs) == 1:
            results = batch_strip_directory(
                args.source_dirs[0],
                output_subdir=args.output_subdir,
                mode=args.mode,
                device=args.device,
                max_resolution=args.max_resolution,
                pipeline=args.pipeline,
                wait_ready=not args.no_wait_ready,
            )
            payload = {"ok": True, "count": len(results), "results": results}
        else:
            by_dir = batch_strip_directories(
                args.source_dirs,
                output_subdir=args.output_subdir,
                mode=args.mode,
                device=args.device,
                max_resolution=args.max_resolution,
                pipeline=args.pipeline,
                wait_ready=not args.no_wait_ready,
            )
            total = sum(len(v) for v in by_dir.values())
            payload = {"ok": True, "directories": len(by_dir), "count": total, "by_dir": by_dir}
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    count = payload.get("count", 0)
    dirs = payload.get("directories", 1)
    log_info(f"全部完成：{count} 张图，{dirs} 个目录")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
