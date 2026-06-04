# Unit 5: pix2pixHD — Mask to Fluorescence image translation

GAN exercise using NVIDIA's pix2pixHD on the **BBBC039** microscopy dataset.
Task: translate binary nuclei masks into realistic fluorescence images at 512×512.

## Notebooks

* **`00_PrepData_BBBC039.ipynb`** — Download BBBC039, binarize instance masks, convert 16-bit TIFF → 8-bit RGB, resize to 512×512, split 160/40 train/test, and arrange in pix2pixHD format (`train_A`, `train_B`, ...).
* **`01_pix2pixHD_Training.ipynb`** — Guideline / command reference. Use the original NVIDIA repo, install deps, run `train.py` and `test.py` with the correct flags for `--label_nc 0`.
* **`02_Evaluate_pix2pixHD.ipynb`** — Load test outputs and evaluate with **SSIM**. Visual comparison, best/worst samples, and progress across epochs.

## Quick start

```bash
cd Unit_5_pix2pixHD
uv sync                              # install deps and run notebooks in order: 00 → 01 (commands to be run in terminal) → 02
```

Training itself happens in a terminal following the commands in `01_pix2pixHD_Training.ipynb`.
