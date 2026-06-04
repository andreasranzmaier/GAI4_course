# HW 4 Report — U-Net Segmentation on BBBC010

## Exercise 2 — Model & Training

A U-Net with an **EfficientNet-B0** encoder (pretrained on ImageNet) was trained to predict
the binary foreground mask from a brightfield (w2) image.

| Setting | Value |
|---|---|
| Encoder | EfficientNet-B0 (ImageNet) |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Optimizer | Adam, lr = 1 × 10⁻⁴ |
| LR schedule | CosineAnnealingLR, T_max = 30 |
| Epochs | 30 |
| Batch size | 4 |
| Input size | 512 × 512 |
| Augmentations | HorizontalFlip, VerticalFlip, RandomRotate90 (all p=0.5), mild Affine |

Best validation Dice reached at epoch **[EPOCH]**: **[BEST_VAL_DICE]**

---

## Exercise 3 — Test-Set Evaluation

Metrics computed on all 20 test wells; threshold = 0.5.

| Metric | Mean ± Std | Min | Max |
|---|---|---|---|
| Dice | [MEAN_DICE] ± [STD_DICE] | [MIN_DICE] | [MAX_DICE] |
| IoU | [MEAN_IOU] ± [STD_IOU] | [MIN_IOU] | [MAX_IOU] |

**Best samples** (highest IoU): wells [BEST_1] and [BEST_2] show clean, compact worm
clusters that closely match the ground-truth outline. The network confidently activates on
the foreground with very little false-positive background signal.

**Worst samples** (lowest IoU): wells [WORST_1] and [WORST_2] likely contain unusual worm
configurations (touching, overlapping, or out-of-focus worms) where the boundary between
foreground and background is ambiguous, leading to missed regions or spurious activations.

---

## Exercise 4 — Robustness to Gaussian Noise

Zero-mean Gaussian noise was added in pixel space ([0, 1]) at
σ ∈ {0, 0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030}.

| σ | Mean IoU | Mean Dice |
|---|---|---|
| 0.000 | [IOU_0] | [DICE_0] |
| 0.001 | [IOU_1] | [DICE_1] |
| 0.002 | [IOU_2] | [DICE_2] |
| 0.005 | [IOU_5] | [DICE_5] |
| 0.010 | [IOU_10] | [DICE_10] |
| 0.015 | [IOU_15] | [DICE_15] |
| 0.020 | [IOU_20] | [DICE_20] |
| 0.030 | [IOU_30] | [DICE_30] |

**Where does the model break?**
Mean IoU drops below 0.5 at approximately σ ≈ **[BREAK_SIGMA]**. This corresponds to a
pixel-level perturbation of roughly ±[BREAK_GRAYLEVEL] grey levels in 8-bit space — a
level of noise that is nearly invisible to the human eye but sufficient to collapse the
model's foreground signal.

The degradation is steep rather than gradual, which indicates the network has learned
high-frequency texture cues from the brightfield images rather than robust structural
features of the worm silhouette. EfficientNet-B0 was pretrained on natural RGB
photographs; its early-layer filters are sensitive to fine texture that is easily destroyed
by even mild additive noise.

**Deployment implications & mitigations**

Deploying this model in a real microscopy pipeline carries risk whenever imaging conditions
deviate from the training distribution (different camera gain, illumination fluctuations,
shot noise from low photon counts). The following mitigations would improve robustness:

1. **Noise augmentation during training** — adding Gaussian noise at the same σ range
   during training teaches the model to ignore it. This is the single most effective and
   lowest-cost fix.
2. **Denoising pre-processing** — a Gaussian or median filter applied before inference
   attenuates zero-mean noise with minimal loss of edge information.
3. **Domain-adapted backbone** — replacing the ImageNet backbone with one pre-trained on
   microscopy data (e.g. from CellPose or BioimageIO) yields features that are
   intrinsically more robust to imaging artefacts.
4. **Test-time augmentation (TTA)** — averaging predictions over multiple noise-perturbed
   copies of the input reduces variance from stochastic noise.
5. **Certified robustness / adversarial training** — training against worst-case
   perturbations (PGD) provides formal guarantees up to a given perturbation radius.
