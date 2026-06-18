"""进程内常驻去水印引擎：启动预加载、单 worker 串行推理。"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

import app.runtime  # noqa: F401 — load_dotenv on import
from app.config import strip_settings
from app.log import log_info
from app.postprocess import apply_flat_region_denoise

logger = logging.getLogger(__name__)

StripState = Literal["idle", "warming", "ready", "failed"]


def _notify(msg: str) -> None:
    """终端可见提示（uvicorn 默认不显示 app logger）。"""
    log_info(msg)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_device(requested: str | None) -> str:
    device = (requested or os.getenv("WATERMARK_STRIP_DEVICE") or "cuda").strip()
    if device == "cuda" and not cuda_available():
        logger.warning("CUDA unavailable, falling back to cpu (very slow)")
        return "cpu"
    return device


def _preload_enabled() -> bool:
    flag = (os.getenv("WATERMARK_STRIP_PRELOAD") or "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


class StripRuntime:
    """单例运行时：InvisibleEngine 常驻显存，threading.Lock 串行处理。"""

    def __init__(self) -> None:
        self._state: StripState = "idle"
        self._error: str | None = None
        self._inv_engine: Any = None
        self._warm_thread: threading.Thread | None = None
        self._infer_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._device = _resolve_device(None)
        self._pipeline = (os.getenv("WATERMARK_STRIP_PIPELINE") or "controlnet").strip()

    def _set_state(self, state: StripState, error: str | None = None) -> None:
        with self._state_lock:
            self._state = state
            self._error = error

    def status(self) -> dict[str, Any]:
        cfg = strip_settings()
        with self._state_lock:
            return {
                "status": self._state,
                "model_loaded": self._state == "ready",
                "cuda": cuda_available(),
                "device": self._device,
                "pipeline": self._pipeline,
                "preload_enabled": _preload_enabled(),
                "error": self._error,
                **cfg,
            }

    def start_preload(self) -> None:
        if not _preload_enabled():
            self._set_state("idle")
            _notify("预加载已关闭 (WATERMARK_STRIP_PRELOAD=0)，首张 /strip 请求时再加载模型")
            return
        with self._state_lock:
            if self._state in {"warming", "ready"}:
                return
            if self._warm_thread and self._warm_thread.is_alive():
                return
        thread = threading.Thread(target=self._warm_worker, name="wm-strip-preload", daemon=True)
        self._warm_thread = thread
        thread.start()

    def _warm_worker(self) -> None:
        self._set_state("warming")
        t0 = time.monotonic()
        _notify(f"正在预加载模型 (device={self._device}, pipeline={self._pipeline}) …")
        try:
            self._get_inv_engine(preload=True)
            self._set_state("ready")
            elapsed = time.monotonic() - t0
            _notify(f"预加载完成 ({elapsed:.0f}s)，服务就绪 — GET /health → status=ready")
        except Exception as exc:
            logger.exception("preload failed")
            self._set_state("failed", str(exc))
            _notify(f"预加载失败: {exc}")

    def wait_ready(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._state_lock:
                state = self._state
                err = self._error
            if state == "ready":
                return
            if state == "failed":
                raise RuntimeError(f"model preload failed: {err}")
            if not _preload_enabled():
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"model not ready after {timeout}s (state={state})")
            if state == "idle":
                self.start_preload()
            time.sleep(0.5)

    def _get_inv_engine(self, *, preload: bool = False) -> Any:
        from remove_ai_watermarks.invisible_engine import InvisibleEngine, is_available

        if not is_available():
            raise RuntimeError("GPU dependencies not installed (remove-ai-watermarks[gpu])")
        if self._inv_engine is None:
            device_arg = None if self._device == "auto" else self._device
            self._inv_engine = InvisibleEngine(
                device=device_arg,
                pipeline=self._pipeline,
            )
        if preload:
            self._inv_engine.preload()
        return self._inv_engine

    def _require_ready_for_invisible(self) -> None:
        if not _preload_enabled():
            return
        with self._state_lock:
            state = self._state
            err = self._error
        if state == "warming":
            raise RuntimeError("model still warming; retry /health until status=ready")
        if state == "failed":
            raise RuntimeError(f"model preload failed: {err}")

    def strip_image(
        self,
        input_path: Path,
        output_path: Path,
        *,
        mode: str | None = None,
        device: str | None = None,
        max_resolution: int | None = None,
        pipeline: str | None = None,
        steps: int | None = None,
        strength: float | None = None,
        adaptive_polish: bool | None = None,
        post_denoise: bool | None = None,
    ) -> dict[str, Any]:
        cfg = strip_settings()
        mode = mode or cfg["default_mode"]
        max_resolution = max_resolution if max_resolution is not None else cfg["max_resolution"]
        steps = steps if steps is not None else cfg["steps"]
        strength = strength if strength is not None else cfg["strength"]
        adaptive_polish = cfg["adaptive_polish"] if adaptive_polish is None else adaptive_polish
        post_denoise = cfg["post_denoise"] if post_denoise is None else post_denoise
        input_path = input_path.resolve()
        output_path = output_path.resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"input not found: {input_path}")
        if mode not in {"all", "invisible", "visible", "metadata"}:
            raise ValueError(f"unsupported mode: {mode}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_device = _resolve_device(device)
        if pipeline and pipeline != self._pipeline:
            logger.warning("per-request pipeline=%s ignored; using runtime pipeline=%s", pipeline, self._pipeline)

        with self._infer_lock:
            if mode in {"invisible", "all"}:
                self._require_ready_for_invisible()
                engine = self._get_inv_engine(preload=True)
            else:
                engine = None

            denoised, used_strength = self._run_strip(
                input_path,
                output_path,
                mode=mode,
                engine=engine,
                max_resolution=max_resolution,
                steps=steps,
                strength=strength,
                adaptive_polish=adaptive_polish,
                post_denoise=post_denoise,
                device=resolved_device,
            )

        return {
            "input": str(input_path),
            "output": str(output_path),
            "mode": mode,
            "device": resolved_device,
            "steps": steps,
            "strength": used_strength,
            "adaptive_polish": adaptive_polish,
            "post_denoise": denoised,
            "ok": output_path.is_file(),
        }

    def _run_strip(
        self,
        input_path: Path,
        output_path: Path,
        *,
        mode: str,
        engine: Any,
        max_resolution: int,
        steps: int,
        strength: float | None,
        adaptive_polish: bool,
        post_denoise: bool,
        device: str,
    ) -> tuple[bool, float | None]:
        from remove_ai_watermarks.cli import (
            _read_bgr_and_alpha,
            _remove_visible_auto,
            _write_bgr_with_alpha,
        )
        from remove_ai_watermarks.noai.watermark_profiles import resolve_strength, vendor_for_strength

        saved_alpha = None
        ran_diffusion = False
        resolved_strength: float | None = None

        if mode in {"visible", "all"}:
            image, alpha = _read_bgr_and_alpha(input_path)
            if image is None:
                raise RuntimeError(f"failed to read image: {input_path}")
            result, _ = _remove_visible_auto(image, inpaint=True)
            _write_bgr_with_alpha(output_path, result, alpha)
            saved_alpha = alpha

        if mode in {"invisible", "all"}:
            assert engine is not None
            vendor = vendor_for_strength(input_path)
            resolved_strength = strength if strength is not None else resolve_strength(None, vendor)
            engine.remove_watermark(
                input_path if mode == "invisible" else output_path,
                output_path,
                strength=resolved_strength,
                num_inference_steps=steps,
                max_resolution=max_resolution,
                adaptive_polish=adaptive_polish,
                humanize=0.0,
                unsharp=0.0,
                vendor=vendor,
            )
            ran_diffusion = True

        if mode in {"metadata", "all"}:
            from remove_ai_watermarks.metadata import remove_ai_metadata

            remove_ai_metadata(input_path if mode == "metadata" else output_path, output_path)

        if mode == "all" and saved_alpha is not None:
            final_bgr, _ = _read_bgr_and_alpha(output_path)
            if final_bgr is not None:
                _write_bgr_with_alpha(output_path, final_bgr, saved_alpha)

        if not output_path.is_file():
            raise RuntimeError(f"strip produced no output: {output_path}")

        if post_denoise and ran_diffusion:
            apply_flat_region_denoise(output_path)
            return True, resolved_strength
        return False, resolved_strength


_runtime: StripRuntime | None = None
_runtime_init_lock = threading.Lock()


def get_runtime() -> StripRuntime:
    global _runtime
    with _runtime_init_lock:
        if _runtime is None:
            _runtime = StripRuntime()
        return _runtime
