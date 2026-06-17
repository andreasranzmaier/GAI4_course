# Unit 7 - Diffusion super-resolution on SR-CACO-2 (X4)

Train a **conditional diffusion model** (SR3-style) that turns a
low-resolution confocal-microscopy acquisition into its high-resolution
counterpart, then evaluate it with **SSIM**.

* **Task:** X4 super-resolution - real low-res patch (`LowRes128`, 128x128) -> high-res patch.
* **Model:** `diffusers.UNet2DModel` trained from scratch as a conditional DDPM. The LR image is concatenated with the noisy HR image (2 input channels); the UNet predicts the noise.
* **Data:** [SR-CACO-2](https://github.com/sbelharbi/sr-caco-2) (Belharbi et al., *NeurIPS 2024 Datasets & Benchmarks*), confocal fluorescence microscopy of the Caco-2 cell line. Single-channel 8-bit TIFF. The LR images are *genuine low-resolution optical scans*, not downsamples.

## Setup

```bash
cd Unit_7
uv sync
```

## Notebooks (run in order)

| Notebook | Purpose |
|----------|---------|
| `00_PrepData_SRCACO2.ipynb` | Pair `LowRes128` <-> `HighRes1024` patches, apply the official tile-based split and run a few visual sanity checks. |
| `01_Train.ipynb` | Train the conditional DDPM; saves `checkpoints/sr3_caco2_x4.pt`. |
| `02_Evaluate.ipynb` | Sample with DDIM and report SSIM |

Dataset: S. Belharbi et al., "SR-CACO-2: A Dataset for Confocal Fluorescence
Microscopy Image Super-Resolution", NeurIPS 2024. arXiv:2406.09168.
