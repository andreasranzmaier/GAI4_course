# Homework 3: pix2pixHD on BBBC010 (20 pts)

Train NVIDIA's pix2pixHD to translate binary foreground masks → brightfield *C. elegans*
brightfield images. Same approach as Unit 5, different dataset.

## Notebooks

* **`00_PrepData.ipynb`** (8 pts, Ex 1) — Download BBBC010, pair masks to brightfield, 80/20 split, save in pix2pixHD layout
* **`01_TrainModel.ipynb`** (2 pts, Ex 2) — Clone pix2pixHD, apply Python-3.11+ patches, train 40 epochs with milestone checkpoints at 5/10/20/40
* **`02_Evaluation.ipynb`** (10 pts, Ex 3) — SSIM at each milestone + side-by-side comparison; discussion

## Deliverables

1. 3 Jupyter notebooks exported as **HTML**
2. 1-page **PDF report**: dataset/pre-processing summary, SSIM trend across epochs,
   discussion

## Setup

```bash
uv sync
```

Run notebooks sequentially (00 → 01 → 02). Look for `___` (blanks) and `# TODO:` comments
are those parts that you need to update and run.
