# Unit 9: Video Intelligence: Transcribe, Translate, Summarise & OCR

Build a small **video-understanding pipeline** on a German [Planet Wissen](https://commons.wikimedia.org/wiki/Category:Planet_Wissen)
documentary clip using three open models, loaded one at a time so they never
share GPU memory:

| Stage | Model |
|-------|-------|
| Speech → text | [`openai/whisper-large-v3`](https://huggingface.co/openai/whisper-large-v3) |
| German → English translation | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (text-only) |
| Summary + on-screen-text OCR | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |

This unit is **self-contained**: it vendors a small `video_pipeline` package
(under `src/`) and there is no dependency on any external repo.

## Notebooks (run in order)

| Notebook | Purpose |
|----------|---------|
| `00_PrepData.ipynb` | `wget` the *Der Golfstrom* clip, convert WebM → MP4 with FFmpeg, transcribe the German speech with Whisper Large-V3 → `prepared/transcript.json`. |
| `01_Translate.ipynb` | Translate the transcript German → English with Qwen2.5-VL (text-only), segment by segment → `prepared/translation.json`. |
| `02_Summarize_OCR.ipynb` | Summarise the English transcript, and OCR on-screen text from frames sampled at 1 fps → `prepared/summary.json`, `prepared/ocr.json`. |

## Setup

FFmpeg is a **system** package — install it first:

```bash
sudo apt update && sudo apt install -y ffmpeg
```

Then the Python environment (managed with [uv](https://docs.astral.sh/uv/)):

```bash
uv sync                 # torch (CUDA 12.8), transformers, etc.
uv run jupyter lab      # run notebooks 00 → 01 → 02
```

> **GPU:** Whisper Large-V3 (~3 GB) and Qwen2.5-VL-7B (~16 GB fp16) are large; a
> CUDA GPU is strongly recommended.

## Command-line interface

The vendored package also installs a `video-pipeline` CLI:

```bash
uv run video-pipeline whisper   prepared/der_golfstrom.mp4 -l german -o prepared/transcript.json
uv run video-pipeline translate prepared/transcript.json   -s German -t English -o prepared/translation.json
uv run video-pipeline summarize prepared/translation.json  -o prepared/summary.json
uv run video-pipeline ocr       prepared/der_golfstrom.mp4 --fps 1 -o prepared/ocr.json
```

## Sources & credits

- **Video:** [*Der Golfstrom* — Planet Wissen](https://commons.wikimedia.org/wiki/File:Der_Golfstrom_-_Planet_Wissen.webm), Wikimedia Commons.
- **Whisper Large-V3:** Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, 2022. [HF](https://huggingface.co/openai/whisper-large-v3) · [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- **Qwen2.5-VL:** Qwen Team, 2025. [HF](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) · [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
- **FFmpeg:** <https://ffmpeg.org/>
