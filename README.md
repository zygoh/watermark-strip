# watermark-strip

> **简介**：本地 **GPU SynthID 去水印** HTTP 服务，封装 [remove-ai-watermarks](https://github.com/wiltodelta/remove-ai-watermarks)；支持可见/隐形水印去除与元数据剥离，单卡串行推理。

本地 GPU 去水印 HTTP 服务，封装 [remove-ai-watermarks](https://github.com/wiltodelta/remove-ai-watermarks) 的 **SynthID 隐形去除**（`mode=all` = 可见 + 扩散再生 + 元数据剥离）。

**边界（2026-06-13）**：**独立单项目**运行（本机常驻 + 穿透调用）；暂不接入外部发帖 flow。

Python 环境在 **`.venv`**（`torch 2.5.1+cu121` + `remove-ai-watermarks 0.11.0`）。

---

## 安装 / 补依赖

`.venv` 已存在时，仅补装缺失包：

```powershell
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
```

### 模型（首次 `all` / `invisible`）

```powershell
# 模型缓存写在 .env 的 HF_HUB_CACHE（默认 D:\huggingface\hub）
$env:HTTPS_PROXY = "http://127.0.0.1:10808"   # 按你的本地代理改
$env:HTTP_PROXY = "http://127.0.0.1:10808"
Remove-Item Env:HF_ENDPOINT -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe scripts\download_models.py
```

SDXL + ControlNet PyTorch 权重约 **13–15GB**（`scripts/download_models.py` 断点续传，跳过 Flax/ONNX）。

### xformers（可选）

略提速、降低推理峰值显存。Windows + torch 2.5.1 须从 **cu124** 源安装，**勿**重装 torch：

```powershell
$env:UV_HTTP_TIMEOUT='900'
uv pip install --python ".venv\Scripts\python.exe" "xformers==0.0.28.post3" `
  --index-url https://download.pytorch.org/whl/cu124 --no-deps
```

验证：`diffusers` 识别 xformers；`Triton is not available` 在 Windows 可忽略。

---

## HTTP 服务

```powershell
copy .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8787 --app-dir .
```

启动后后台**预加载** SDXL + ControlNet（约 40–90s）。终端打印 `[watermark-strip] 预加载完成 …`；或 `GET /health` → `status=ready`。

| 接口 | 说明 |
|------|------|
| `GET /health` | `status`: `warming` \| `ready` \| `failed`；`model_loaded`、`cuda`、`device`、strip 默认参数 |
| `POST /strip` | 表单 `image` + 可选 `mode`（默认 `all`）；`warming` → 503；穿透带 `X-API-Key` |

**约束**

- 请求**串行**（单 GPU worker），勿并发多张。
- `mode=all` 默认 **`adaptive_polish`** + **`post_denoise`**（平涂区双边滤波，减轻字块白点）。
- 去 SynthID 须 `all`/`invisible`；仅 `metadata` 不过 OpenAI verify。

**远端示例**

```powershell
curl http://127.0.0.1:8787/health

curl.exe -X POST "http://127.0.0.1:8787/strip" `
  -F "image=@path\to\01-cover.png" `
  -F "mode=all" `
  -o "path\to\cleaned\01-cover.png"
```

---

## 环境变量

见 [`.env.example`](.env.example)。常用：

| 变量 | 默认 | 说明 |
|------|------|------|
| `WATERMARK_STRIP_DEFAULT_MODE` | `all` | `all` \| `invisible` \| `visible` \| `metadata` |
| `WATERMARK_STRIP_STEPS` | `50` | 扩散步数 |
| `WATERMARK_STRIP_STRENGTH` | `0.20` | GenerateImage/OpenAI；清洗后 JPEG 无 C2PA 时勿留空 |
| `WATERMARK_STRIP_POST_DENOISE` | `1` | 扩散后平涂区降噪 |
| `WATERMARK_STRIP_MAX_RESOLUTION` | `1536` | 对齐 GenerateImage 出图 |

---

## 批量处理

```powershell
.\.venv\Scripts\python.exe scripts\batch_strip.py "path\to\images-dir"
```

输出写入源目录 `cleaned/`，原图不动。

---

## 资源

- **GPU**：NVIDIA ≥ 8GB VRAM；1536 图单张 `all` 约 **6–10 分钟**。
- **无 GPU**：仅可 `visible` + `metadata`，不可 `invisible`/`all`。

---

## GitHub About 简介

`本地 GPU SynthID 去水印 HTTP 服务，封装 remove-ai-watermarks，支持可见/隐形水印与元数据剥离。`
