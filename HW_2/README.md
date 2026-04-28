# Homework 2: OOD Detection — Ensemble vs Autoencoder (20 pts)

Compare two paradigms for out-of-distribution detection on the UC Merced Land Use dataset (in-distribution) with FIDS30 fruit images (OOD).

## Notebooks

* **`00_PrepData.ipynb`** (4 pts) — Download UC Merced + FIDS30, split 70/15/15, visualize samples
* **`01_DeepEnsemble_OOD.ipynb`** (8 pts) — Train 10 EfficientNet-B0 models, use variance across models as OOD score
* **`02_Autoencoder_OOD.ipynb`** (8 pts) — Train a conv autoencoder, use reconstruction MSE as OOD score

## Deliverables

1. 3 Jupyter notebooks exported as **HTML**
2. 1-page **PDF report**: compare the two methods (detection rate, false alarm rate, training cost, need for labels), discuss which wins here and why.

## Setup

```bash
uv sync
```

Run notebooks sequentially (00 → 01 → 02). The ensemble notebook takes a while (10 models × 5 epochs)
