"""扩散去水印后的轻量后处理（压平涂区颗粒，保留文字边缘）。"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from remove_ai_watermarks import image_io
from remove_ai_watermarks.humanizer import _smooth_grain_mask


def _denoise_config() -> tuple[int, float, float]:
    d = int(os.getenv("WATERMARK_STRIP_DENOISE_D") or "9")
    sigma = float(os.getenv("WATERMARK_STRIP_DENOISE_SIGMA") or "40")
    return d, sigma, sigma


def apply_flat_region_denoise(path: Path) -> None:
    """仅在平滑区域混合双边滤波结果，减轻字块内白点颗粒。"""
    img = image_io.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"post_denoise: cannot read {path}")

    d, sigma_color, sigma_space = _denoise_config()
    flat_weight = _smooth_grain_mask(img)
    denoised = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    w = flat_weight[:, :, np.newaxis]
    blended = img.astype(np.float32) * (1.0 - w) + denoised.astype(np.float32) * w
    image_io.imwrite(str(path), np.clip(blended, 0, 255).astype(np.uint8))
