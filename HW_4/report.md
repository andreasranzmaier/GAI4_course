# HW 4 Report - U-Net Segmentation on BBBC010

## Exercise 2 - Model & Training

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

Best validation Dice reached at epoch **30**: **0.8591**. Notably the validation Dice was
still rising at epoch 30 (0.843 -> 0.852 -> 0.857 -> 0.859 over the final epochs) and the
validation loss was still falling, so the model had not fully converged within the 30-epoch
budget - training further would likely improve results (cf. the epoch-60 finding in HW 3).

---

## Exercise 3 - Test-Set Evaluation

Metrics computed on all 20 test wells; threshold = 0.5.

| Metric | Mean ± Std | Min | Max |
|---|---|---|---|
| Dice | 0.859 ± 0.024 | 0.816 | 0.899 |
| IoU | 0.753 ± 0.036 | 0.690 | 0.817 |

**Best samples** (highest IoU): wells C07 and D10 show clean, compact worm
clusters that closely match the ground-truth outline. The network confidently activates on
the foreground with very little false-positive background signal.

**Worst samples** (lowest IoU): wells D23 and C17 likely contain unusual worm
configurations (touching, overlapping, or out-of-focus worms) where the boundary between
foreground and background is ambiguous, leading to missed regions or spurious activations.

---

## Exercise 4 - Robustness to Gaussian Noise

Zero-mean Gaussian noise was added in pixel space ([0, 1]) at
σ ∈ {0, 0.001, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030}.

| σ | Mean IoU | Mean Dice |
|---|---|---|
| 0.000 | 0.753 | 0.859 |
| 0.001 | 0.758 | 0.862 |
| 0.002 | 0.759 | 0.863 |
| 0.005 | 0.764 | 0.866 |
| 0.010 | 0.696 | 0.816 |
| 0.015 | 0.405 | 0.564 |
| 0.020 | 0.160 | 0.271 |
| 0.030 | 0.075 | 0.139 |

**Where does the model break?**
Mean IoU drops below 0.5 at approximately σ ≈ **0.013** (interpolating between σ = 0.010,
IoU 0.696 and σ = 0.015, IoU 0.405). This corresponds to a pixel-level perturbation of
roughly ±3 grey levels in 8-bit space - a level of noise that is nearly invisible to the
human eye but sufficient to collapse the model's foreground signal.

The degradation is steep rather than gradual, which indicates the network has learned
high-frequency texture cues from the brightfield images rather than robust structural
features of the worm silhouette. EfficientNet-B0 was pretrained on natural RGB
photographs; its early-layer filters are sensitive to fine texture that is easily destroyed
by even mild additive noise.

**Deployment implications & mitigations**

Deploying this model in a real microscopy pipeline carries risk whenever imaging conditions
deviate from the training distribution (different camera gain, illumination fluctuations,
shot noise from low photon counts). The following mitigations would improve robustness:

1. **Noise augmentation during training** - adding Gaussian noise at the same σ range
   during training teaches the model to ignore it. This is the single most effective and
   lowest-cost fix.
2. **Denoising pre-processing** - a Gaussian or median filter applied before inference
   attenuates zero-mean noise with minimal loss of edge information.
3. **Domain-adapted backbone** - replacing the ImageNet backbone with one pre-trained on
   microscopy data (e.g. from CellPose or BioimageIO) yields features that are
   intrinsically more robust to imaging artefacts.
4. **Test-time augmentation (TTA)** - averaging predictions over multiple noise-perturbed
   copies of the input reduces variance from stochastic noise.
5. **Certified robustness / adversarial training** - training against worst-case
   perturbations (PGD) provides formal guarantees up to a given perturbation radius.

---

## Exercise 5 - Well-Mask Extension (4-channel input)

The HW 3 feedback identified the irregular grey **well background** as the biggest remaining
artifact and suggested *adding masks for the well*. This extension applies that idea to the
segmentation task (`03_WellMask_Experiment.ipynb`): the circular assay-well region is derived
from each brightfield image (threshold the non-black area -> largest component -> convex-hull
fill) and appended as a **4th input channel** (`RGB + well`). Everything else is identical to
the baseline, so the comparison is controlled. `smp` adapts the pretrained first convolution
to 4 channels (the RGB filters keep their ImageNet weights). A sanity check confirmed ~100 %
of the ground-truth worm foreground lies inside the derived well.

**Test-set comparison (20 wells, threshold 0.5):**

| Model | Dice | IoU |
|---|---|---|
| Baseline (RGB) | 0.859 ± 0.024 | 0.753 ± 0.036 |
| Well (RGB + well) | **0.869 ± 0.020** | **0.769 ± 0.032** |
| Δ | +0.011 | +0.016 |

On clean data the well channel gives a small but consistent improvement. The decisive gain is
in **noise robustness**, re-running the σ sweep for both models:

| σ | Baseline IoU | Well IoU | Δ |
|---|---|---|---|
| 0.010 | 0.696 | 0.765 | +0.069 |
| 0.015 | 0.405 | 0.726 | +0.321 |
| 0.020 | 0.160 | 0.636 | +0.476 |
| 0.030 | 0.075 | 0.448 | +0.373 |

Where the baseline collapses past σ ≈ 0.013, the well model degrades only gently. The reason is
that the well channel is a **noise-free geometric anchor**: it is a coarse morphological disc
that survives mild additive noise, so even when the RGB channels are corrupted the network
still knows where the well - and therefore the plausible foreground region - is. Making the
well boundary explicit both fixes the artifact highlighted in the HW 3 feedback and, as a
by-product, substantially hardens the model against imaging noise. (The well mask is derived
from the *noisy* image at each σ, i.e. a realistic inference-time setting, not a clean-image
leak.) Like the baseline, the 4-channel model was still improving at epoch 30
(best val Dice 0.8695), so both would benefit from a longer training schedule.
