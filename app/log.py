"""终端可见日志（带 GMT+8 时间戳）。"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_GMT8 = timezone(timedelta(hours=8))


def log_info(msg: str) -> None:
    ts = datetime.now(_GMT8).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [watermark-strip] {msg}", flush=True)
    logger.info("%s", msg)
