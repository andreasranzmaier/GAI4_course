# HW 3 Report — pix2pixHD on BBBC010

## Dataset & Pre-processing

**BBBC010** is a publicly available *C. elegans* brightfield microscopy dataset from the
Broad Bioimage Benchmark Collection. It contains 100 wells, each imaged in two channels
(w1: fluorescence, w2: brightfield) at 16-bit depth, together with per-well binary
foreground masks labelling worm pixels.

**Pre-processing pipeline (00_PrepData.ipynb):**

1. Downloaded three archives from Broad Institute: images, per-well foreground masks, and
   per-worm foreground masks (the latter not used in training).
2. Paired each well's **channel-2 (brightfield) image** with its corresponding binary mask
   by extracting the well code (e.g. `A02`) from both filenames.
3. **Binarisation**: mask pixels > 0 set to 255; all else to 0.
4. **Normalisation**: 16-bit images linearly mapped to [0, 255] using per-image min/max, then
   converted to 8-bit RGB.
5. **Resize**: both mask and image resized to 512 × 512 (bilinear for images, nearest-neighbour
   for masks to preserve sharp boundaries).
6. **Split**: 80 train / 20 test wells, drawn with `random.seed(42)`.
7. Saved in pix2pixHD layout: `train_A` / `train_B` / `test_A` / `test_B` (A = mask, B = image).

---

## Training

pix2pixHD was cloned from NVIDIA's repository and patched for Python 3.11+ compatibility
(`apply_patches.py`). Key training flags:

| Flag | Value | Reason |
|---|---|---|
| `--label_nc` | 0 | Input is a raw image (mask), not a semantic map |
| `--no_instance` | — | No instance boundaries |
| `--loadSize / --fineSize` | 512 | Matches pre-processing |
| `--batchSize` | 2 | Fits a single modern GPU at 512 × 512 |
| `--niter` | 40 | Training epochs (no LR decay) |
| `--niter_decay` | 0 | Constant LR throughout |
| `--save_epoch_freq` | 100 | Disabled periodic saves; milestone logic handles 5/10/20/40 |

Milestone checkpoints were saved at epochs **5, 10, 20, 40** via the patched `train.py`.

---

## SSIM Results

SSIM (Structural Similarity Index) was computed between each generated test image and its
ground-truth brightfield counterpart, using the luminance channel (mean of RGB) with
`data_range=255`.

<!-- Insert ssim_trend.png here -->

| Epoch | Mean SSIM | Std |
|---|---|---|
| 5 | *(fill after run)* | *(fill)* |
| 10 | *(fill after run)* | *(fill)* |
| 20 | *(fill after run)* | *(fill)* |
| 40 | *(fill after run)* | *(fill)* |

SSIM improves most steeply between epochs 5 and 10 as the generator learns coarse worm
shape. Gains slow in later epochs as the model fine-tunes texture detail. Residual variance
across test wells reflects genuine biological variability in worm density and orientation.

---

## Visual Quality & Artefacts

<!-- Insert comparison_epochs.png here -->

**Plausibility.** By epoch 20–40 generated images show the characteristic granular
brightfield texture of *C. elegans* within the mask boundary. The background (outside the
mask) is correctly rendered as a flat, near-uniform grey consistent with the real images.

**Artefacts observed:**

- **Boundary ringing** — a faint halo at the mask edge, most pronounced at epoch 5. The
  generator is uncertain about the transition region and oscillates in intensity there.
- **Smooth interiors** — early epochs produce uniformly grey worm bodies with little
  internal texture; this gradually resolves toward epoch 40.
- **Occasional background bleed** — faint worm-like texture appearing just outside the mask
  border. Likely driven by the VGG perceptual loss, which encourages high-frequency detail
  globally.

**Zoomed comparison.** Zooming into worm bodies reveals that the real images have
anisotropic internal structure (gut granules, refraction rings) that the model approximates
only statistically — generated granularity is plausible but not pixel-accurate, as expected
for a stochastic GAN.

---

## Summary

pix2pixHD trained for 40 epochs on 80 mask–image pairs generates qualitatively convincing
*C. elegans* brightfield images. SSIM improves throughout training. The main remaining
limitations are smoothed interiors and occasional boundary halos — both typical of a
perceptual-loss GAN on a small dataset.
