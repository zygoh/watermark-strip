"""本地 GPU 去水印 HTTP 服务（供穿透后云端调用）。"""

from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import strip_settings
from app.engine import get_runtime

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

_cfg = strip_settings()
API_KEY = (os.getenv("WATERMARK_STRIP_API_KEY") or "").strip()
MAX_RESOLUTION = _cfg["max_resolution"]
DEFAULT_MODE = _cfg["default_mode"]
DEFAULT_DEVICE = (os.getenv("WATERMARK_STRIP_DEVICE") or "cuda").strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = get_runtime()
    print("[watermark-strip] HTTP 服务已启动，后台预加载线程即将运行 …", flush=True)
    runtime.start_preload()
    app.state.runtime = runtime
    yield


app = FastAPI(title="watermark-strip", version="0.2.0", lifespan=lifespan)


def _check_auth(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


@app.get("/health")
def health():
    info = get_runtime().status()
    # 兼容旧字段 + 新状态机
    return {
        **info,
        "device_default": info["device"],
        "ready": info["status"] == "ready",
    }


@app.post("/strip")
async def strip_upload(
    image: UploadFile = File(...),
    mode: str = Form(DEFAULT_MODE),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """上传单张图，返回处理后的文件。mode 默认 all（可见+隐形+元数据）。"""
    _check_auth(x_api_key)
    if mode not in {"all", "invisible", "visible", "metadata"}:
        raise HTTPException(status_code=400, detail=f"unsupported mode: {mode}")

    runtime = get_runtime()
    if mode in {"all", "invisible"}:
        st = runtime.status()
        if st["status"] == "warming":
            raise HTTPException(status_code=503, detail="model warming; retry shortly")
        if st["status"] == "failed":
            raise HTTPException(status_code=503, detail=f"model unavailable: {st.get('error')}")

    suffix = Path(image.filename or "image.png").suffix or ".png"
    with tempfile.TemporaryDirectory(prefix="wm-strip-") as tmp:
        inp = Path(tmp) / f"input{suffix}"
        out = Path(tmp) / f"output{suffix}"
        inp.write_bytes(await image.read())
        try:
            meta = await asyncio.to_thread(
                runtime.strip_image,
                inp,
                out,
                mode=mode,
                device=DEFAULT_DEVICE,
                max_resolution=MAX_RESOLUTION,
            )
            body = out.read_bytes()
        except RuntimeError as exc:
            msg = str(exc)
            if "warming" in msg.lower():
                raise HTTPException(status_code=503, detail=msg) from exc
            raise HTTPException(status_code=500, detail=msg) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    media_type = image.content_type or "image/png"
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="clean{suffix}"',
            "X-Strip-Device": meta["device"],
            "X-Strip-Mode": meta["mode"],
            "X-Strip-Strength": str(meta.get("strength") or ""),
            "X-Strip-Post-Denoise": "1" if meta.get("post_denoise") else "0",
        },
    )
