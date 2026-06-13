"""Download SDXL + ControlNet with progress, retries, and Windows-safe locks."""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

OFFICIAL_ENDPOINT = "https://huggingface.co"
MIRROR_ENDPOINT = "https://hf-mirror.com"

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "7200")


def _use_mirror() -> bool:
    flag = (os.environ.get("HF_USE_MIRROR") or "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return True
    if flag in {"0", "false", "no"}:
        return False
    endpoint = (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/")
    return endpoint == MIRROR_ENDPOINT


def _hub_endpoint() -> str:
    if _use_mirror():
        return MIRROR_ENDPOINT
    custom = (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/")
    return custom or OFFICIAL_ENDPOINT


USE_MIRROR = _use_mirror()
HUB_ENDPOINT = _hub_endpoint()

if USE_MIRROR:
    os.environ["HF_ENDPOINT"] = MIRROR_ENDPOINT
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        os.environ.pop(key, None)
else:
    os.environ.pop("HF_ENDPOINT", None)

import httpx
from huggingface_hub.utils._http import close_session, hf_request_event_hook, set_client_factory


def _http_client_factory() -> httpx.Client:
    return httpx.Client(
        event_hooks={"request": [hf_request_event_hook]},
        follow_redirects=True,
        timeout=None,
        trust_env=not USE_MIRROR,
    )


close_session()
set_client_factory(_http_client_factory)
import huggingface_hub.file_download as fd
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.file_download import HfFileMetadata

CACHE_HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))

MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "xinsir/controlnet-canny-sdxl-1.0",
]

# diffusers 只需 PyTorch 组件；跳过 Flax/ONNX/OpenVINO/演示图，避免 50GB+ 整仓。
PYTORCH_ONLY_IGNORE = [
    "*.msgpack",
    "**/*flax*",
    "**/*.onnx",
    "**/*.onnx_data",
    "**/openvino_*",
    "**/*.bin",
    "**/*.h5",
    "**/*.ckpt",
    "*.webp",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "**/*.fp16.safetensors",
    "sd_xl_base*.safetensors",
    "sd_xl_offset*.safetensors",
]

MAX_RETRIES = int(os.environ.get("HF_DOWNLOAD_MAX_RETRIES", "30"))
PROGRESS_INTERVAL_S = 30
_orig_get_hf_file_metadata = fd.get_hf_file_metadata
_hf_api = HfApi(endpoint=HUB_ENDPOINT)
_commit_cache: dict[tuple[str, str], str] = {}


def _sanitize_etag(etag: str | None) -> str | None:
    if not etag or not isinstance(etag, str):
        return etag
    clean = etag.strip().strip('"').strip("'")
    clean = re.sub(r'[<>:"/\\|?*]', "_", clean)
    return clean or etag


def _proxy_label() -> str:
    return (
        os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
        or "none"
    )


def install_metadata_patch() -> None:
    def repo_commit(repo_id: str, revision: str) -> str:
        key = (repo_id, revision)
        if key not in _commit_cache:
            _commit_cache[key] = _hf_api.repo_info(repo_id, revision=revision).sha
        return _commit_cache[key]

    def patched_get_hf_file_metadata(
        url,
        token=None,
        timeout=10.0,
        user_agent=None,
        headers=None,
        endpoint=None,
        retry_on_errors=False,
    ):
        endpoint = endpoint or HUB_ENDPOINT
        url_s = str(url)
        try:
            meta = _orig_get_hf_file_metadata(
                url,
                token=token,
                timeout=timeout,
                user_agent=user_agent,
                headers=headers,
                endpoint=endpoint,
                retry_on_errors=retry_on_errors,
            )
            if meta.commit_hash and meta.etag and meta.size is not None:
                loc = meta.location or url_s
                if USE_MIRROR:
                    m = re.search(r"https?://[^/]+/(.+?)/resolve/([^/]+)/(.+)", url_s)
                    if m:
                        repo_id, revision, relpath = m.group(1), m.group(2), m.group(3)
                        loc = f"{MIRROR_ENDPOINT}/{repo_id}/resolve/{revision}/{relpath}"
                    else:
                        loc = loc.replace(OFFICIAL_ENDPOINT, MIRROR_ENDPOINT)
                return HfFileMetadata(
                    commit_hash=meta.commit_hash,
                    etag=_sanitize_etag(meta.etag),
                    location=loc,
                    size=meta.size,
                    xet_file_data=None,
                )
        except Exception:
            if not USE_MIRROR:
                raise

        m = re.search(r"https?://[^/]+/(.+?)/resolve/([^/]+)/(.+)", url_s)
        if not m:
            raise RuntimeError(f"Cannot parse hub url: {url_s}")
        repo_id, revision, relpath = m.group(1), m.group(2), m.group(3)
        commit = repo_commit(repo_id, revision)
        fetch_url = f"{HUB_ENDPOINT}/{repo_id}/resolve/{revision}/{relpath}"
        hdrs = dict(headers or {})
        with httpx.Client(
            follow_redirects=True,
            timeout=60.0,
            trust_env=not USE_MIRROR,
        ) as client:
            with client.stream("GET", fetch_url, headers=hdrs) as response:
                response.raise_for_status()
                loc = str(response.url)
                if USE_MIRROR:
                    loc = loc.replace(OFFICIAL_ENDPOINT, MIRROR_ENDPOINT)
                etag = response.headers.get("etag") or response.headers.get("x-linked-etag") or commit
                etag = _sanitize_etag(etag if isinstance(etag, str) else str(etag))
                cl = response.headers.get("content-length")
                size = int(cl) if cl else None
                if size is None:
                    linked = response.headers.get("x-linked-size")
                    size = int(linked) if linked else 0
        return HfFileMetadata(
            commit_hash=commit,
            etag=etag,
            location=loc,
            size=size,
            xet_file_data=None,
        )

    def patched_official_only(url, token=None, timeout=10.0, user_agent=None, headers=None, endpoint=None, retry_on_errors=False):
        meta = _orig_get_hf_file_metadata(
            url,
            token=token,
            timeout=timeout,
            user_agent=user_agent,
            headers=headers,
            endpoint=endpoint,
            retry_on_errors=retry_on_errors,
        )
        if meta.etag:
            return HfFileMetadata(
                commit_hash=meta.commit_hash,
                etag=_sanitize_etag(meta.etag),
                location=meta.location,
                size=meta.size,
                xet_file_data=meta.xet_file_data,
            )
        return meta

    fd.get_hf_file_metadata = patched_get_hf_file_metadata if USE_MIRROR else patched_official_only


install_metadata_patch()


def cache_stats() -> tuple[int, int]:
    if not CACHE_HUB.is_dir():
        return 0, 0
    total = 0
    count = 0
    for p in CACHE_HUB.rglob("*"):
        if p.is_file():
            count += 1
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return count, total


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    n = float(n)
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
    return f"{n:.2f} TB"


def model_cache_dir(repo_id: str) -> Path:
    return CACHE_HUB / ("models--" + repo_id.replace("/", "--"))


def clean_incomplete() -> None:
    if not CACHE_HUB.is_dir():
        return
    for inc in CACHE_HUB.rglob("*.incomplete"):
        try:
            inc.unlink()
            print(f"Removed incomplete blob: {inc.name}", flush=True)
        except OSError as e:
            print(f"WARN: could not remove {inc}: {e}", flush=True)


def clean_locks(repo_id: str | None = None) -> None:
    locks_root = CACHE_HUB / ".locks"
    if not locks_root.is_dir():
        return
    if repo_id:
        targets = [locks_root / ("models--" + repo_id.replace("/", "--"))]
    else:
        targets = [locks_root]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            print(f"Removed locks: {target.name}", flush=True)


class ProgressReporter:
    def __init__(self, label: str) -> None:
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_bytes = 0
        self._last_change = time.monotonic()

    def start(self) -> None:
        count, total = cache_stats()
        self._last_bytes = total
        self._last_change = time.monotonic()
        print(
            f"[progress] {self.label} start | hub {count} files, {fmt_bytes(total)}",
            flush=True,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.wait(PROGRESS_INTERVAL_S):
            count, total = cache_stats()
            delta = total - self._last_bytes
            if delta > 0:
                self._last_change = time.monotonic()
                self._last_bytes = total
            stale_s = time.monotonic() - self._last_change
            print(
                f"[progress] {self.label} | hub {count} files, {fmt_bytes(total)}"
                f" (+{fmt_bytes(max(delta, 0))} since last)"
                f" | no-growth {stale_s:.0f}s",
                flush=True,
            )


def _is_transient_download_error(err: BaseException) -> bool:
    msg = f"{type(err).__name__}: {err}".lower()
    needles = (
        "peer closed",
        "remoteprotocolerror",
        "read timed out",
        "readtimeout",
        "connection reset",
        "connection aborted",
        "incomplete message body",
        "errno 22",
        "timed out",
        "502",
        "503",
        "504",
    )
    return any(n in msg for n in needles)


def _retry_pause_seconds(attempt: int, err: BaseException) -> int:
    if _is_transient_download_error(err):
        return min(30 + 15 * attempt, 180)
    return min(10 * attempt, 60)


def download_repo(repo_id: str) -> Path:
    last_err: BaseException | None = None
    snap_kwargs: dict = {
        "repo_id": repo_id,
        "ignore_patterns": PYTORCH_ONLY_IGNORE,
        "max_workers": 1,
    }
    if USE_MIRROR:
        snap_kwargs["endpoint"] = MIRROR_ENDPOINT

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            clean_incomplete()
            clean_locks(repo_id)
            print(f"Resume {repo_id} (attempt {attempt}), keeping completed files", flush=True)
        reporter = ProgressReporter(f"{repo_id} try {attempt}/{MAX_RETRIES}")
        reporter.start()
        try:
            mode = "mirror" if USE_MIRROR else "official+proxy"
            print(
                f"Downloading {repo_id} (attempt {attempt}, {mode}, pytorch-only, 1 worker) ...",
                flush=True,
            )
            path = snapshot_download(**snap_kwargs)
            count, total = cache_stats()
            print(
                f"Done: {repo_id} -> {path} | hub total {count} files, {fmt_bytes(total)}",
                flush=True,
            )
            return Path(path)
        except Exception as e:
            last_err = e
            transient = _is_transient_download_error(e)
            print(
                f"FAIL {repo_id} attempt {attempt}/{MAX_RETRIES}"
                f" ({'transient, will resume' if transient else 'error'}): {e}",
                flush=True,
            )
            if attempt < MAX_RETRIES:
                pause = _retry_pause_seconds(attempt, e)
                print(f"Retry in {pause}s ...", flush=True)
                time.sleep(pause)
        finally:
            reporter.stop()
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {repo_id}") from last_err


def verify_repo(path: Path, repo_id: str) -> dict[str, object]:
    files = [p for p in path.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    names = {p.name.lower() for p in files}
    info: dict[str, object] = {
        "repo_id": repo_id,
        "path": str(path),
        "file_count": len(files),
        "total_bytes": total,
        "total_human": fmt_bytes(total),
    }
    if "stabilityai" in repo_id:
        paths_lower = [str(p).lower() for p in files]
        info["has_unet"] = any("unet" in p and p.endswith(".safetensors") for p in paths_lower)
        info["has_vae"] = any("vae" in p and p.endswith(".safetensors") for p in paths_lower)
        info["ok"] = bool(info["has_unet"] and info["has_vae"]) and total > 10_000_000_000
    elif "controlnet" in repo_id:
        info["has_safetensors"] = any(n.endswith(".safetensors") for n in names)
        info["ok"] = bool(info["has_safetensors"]) and total > 1_000_000_000
    else:
        info["ok"] = len(files) > 0
    return info


def main() -> int:
    print(f"mode={'mirror' if USE_MIRROR else 'official'}", flush=True)
    print(f"endpoint={HUB_ENDPOINT}", flush=True)
    print(f"proxy={_proxy_label()}", flush=True)
    print(f"max_retries={MAX_RETRIES}", flush=True)
    print(f"HF_HUB_CACHE={CACHE_HUB}", flush=True)
    clean_incomplete()
    clean_locks()
    paths: list[Path] = []
    for repo_id in MODELS:
        paths.append(download_repo(repo_id))
    print("\n=== Verification ===", flush=True)
    all_ok = True
    for repo_id, path in zip(MODELS, paths):
        v = verify_repo(path, repo_id)
        all_ok = all_ok and bool(v.get("ok"))
        print(v, flush=True)
    if not all_ok:
        print("VERIFICATION FAILED", flush=True)
        return 1
    print("ALL MODELS READY for remove-ai-watermarks (controlnet pipeline)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
