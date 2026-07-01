# Homework 5: Foundation Models (20 pts)

Apply the **Unit 9** pipeline — transcribe → translate → summarise + OCR — to a
German [Planet Wissen](https://commons.wikimedia.org/wiki/Category:Planet_Wissen)
documentary clip **of your own choice**.

This homework is **self-contained**: it vendors the same small `video_pipeline`
package (under `src/`).

## Exercises

* **`00_PrepData.ipynb`** (4 pts, Ex 1) — Choose a clip from the [Planet Wissen category](https://commons.wikimedia.org/wiki/Category:Planet_Wissen),
  `wget` it, convert WebM → MP4, and transcribe the German speech with Whisper Large-V3.
* **`01_Translate.ipynb`** (4 pts, Ex 2) — Translate the transcript German → English with Qwen2.5-VL.
* **`02_Summarize_OCR.ipynb`** (4 pts, Ex 3) — Summarise the English transcript and OCR the on-screen text at 1 fps.
* **`Change the models to `** (8 pts, Ex 4) — Swap the models with smaller counterparts and redo, compare runtime and memory usage (you might need to make changes within ./src/). Use e.g. [whisper-tiny](https://huggingface.co/openai/whisper-tiny) and [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

Look for `___` (blanks) and `# TODO:` comments — those are the parts you fill in
and run.

## Deliverables

1. The three notebooks executed end-to-end and exported as **HTML**.
2. A short **report** (≤1 page): which clip you used, the English summary, and a
   brief comment on the OCR quality (what on-screen text was / wasn't captured). 
   Analyze runtime and memory use.

## Setup

```bash
sudo apt update && sudo apt install -y ffmpeg   # system package
uv sync                                         # Python environment
uv run jupyter lab                              # run 00 → 01 → 02
```

> A CUDA GPU is strongly recommended: Whisper Large-V3 (~3 GB) and
> Qwen2.5-VL-7B (~16 GB fp16). The models are loaded one at a time.

## Sources & credits

- **Videos:** [Planet Wissen on Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Planet_Wissen).
- **Whisper Large-V3:** [HF](https://huggingface.co/openai/whisper-large-v3) · [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- **Qwen2.5-VL:** [HF](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) · [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
- **FFmpeg:** <https://ffmpeg.org/>
