# Homework 4: Segmentation on BBBC010

Inverse of HW 3: train a **U-Net (EfficientNet-B0 encoder)** with
[`segmentation_models_pytorch`](https://github.com/qubvel-org/segmentation_models.pytorch)
to predict the foreground mask from a brightfield image.

## Notebooks

* **`00_PrepData.ipynb`** Download BBBC010, pair brightfield with mask, 80/20 split, save for segmentation
* **`01_TrainModel.ipynb`** U-Net (EfficientNet-B0), BCE + Dice loss, 30 epochs, save best by val Dice
* **`02_Evaluation.ipynb`** Dice/IoU on test set + best/worst grid; Gaussian-noise robustness + discussion

## Deliverables

1. 3 Jupyter notebooks exported as **HTML**
2. 1-page **PDF report**: final Dice/IoU, best/worst comments, IoU-vs-noise curve and
   deployment-readiness discussion.

## Setup

```bash
uv sync
```

Run notebooks sequentially (00 → 01 → 02). Look for `___` (blanks) and `# TODO:` comments
— those are the parts you need to fill in.
