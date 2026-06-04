# Unit 6 — Nuclei Segmentation (BBBC039)

The inverse of Unit 5. There we trained a conditional GAN to **generate** fluorescence
microscopy images from binary masks. Here we go the other way: given a real fluorescence
image, predict the **binary mask** that segments the nuclei.

We use [`segmentation_models_pytorch`](https://github.com/qubvel-org/segmentation_models.pytorch)
to build a U-Net with an **EfficientNet-B0** encoder pretrained on ImageNet. Training is a
combination of **BCE** + **Dice** loss. We evaluate with **Dice** and **IoU**, then probe
**adversarial robustness** by adding Gaussian noise at increasing strengths and watching
the IoU collapse.

## Notebooks

| | Notebook | What it does |
|---|---|---|
| 0 | `00_PrepData_BBBC039.ipynb`   | Download BBBC039, binarize masks, resize, split into `train/` and `test/`. |
| 1 | `01_UNet_Training.ipynb`      | Train U-Net (EfficientNet-B0 backbone) with BCE + Dice loss. |
| 2 | `02_Evaluate_UNet.ipynb`      | Dice/IoU on the test set + adversarial robustness (Gaussian noise sweep). |

## Setup

```bash
uv sync
```

## Dataset

[BBBC039](https://bbbc.broadinstitute.org/BBBC039) — 200 fluorescence microscopy images of
U2OS cell nuclei with instance segmentation masks. We binarize the masks (any instance >0 →
foreground) and resize everything to 512×512. Split is 160 train / 40 test, same as Unit 5.

## Layout after data prep

```
datasets/bbbc039_seg/
├── train/
│   ├── images/   # fluorescence (RGB, 512×512)
│   └── masks/    # binary mask (L, 512×512, {0,255})
└── test/
    ├── images/
    └── masks/
```
