# Homework 1: Scratch vs Pretrained Image Classification (UC Merced)

Train and compare EfficientNet-B0 models (from scratch vs ImageNet pretrained) on the UC Merced Land Use dataset (21 classes).

## Notebooks

* **`00_PrepData.ipynb`** (4 pts) — Load UC Merced from HuggingFace, split 70/15/15, visualize samples
* **`01_TrainModel.ipynb`** (12 pts) — Train from scratch (6 pts) + pretrained (6 pts), 20 epochs each
* **`02_Evaluation.ipynb`** (4 pts) — Side-by-side comparison, confusion matrices, discussion

## Deliverables

1. 3 Jupyter notebooks exported as **HTML**
2. 1-page **PDF report**: summarize results, comparison, and key observations

## Setup

```bash
uv sync
```

Run notebooks sequentially (00 → 01 → 02).
