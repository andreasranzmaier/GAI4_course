# Homework 2: OOD Detection = Ensemble vs Autoencoder (20 pts)
Ranzmaier

---

## Report: Deep Ensemble vs. Convolutional Autoencoder for OOD Detection

### Task Overview

Compared two different approaches to out-of-distribution (OOD) detection. The model is always trained on **UC Merced Land Use** aerial imagery (in-distribution), and at test time **FIDS30** fruit photos are used as OOD samples. Both methods use the same thresholding strategy: the threshold is set at the 95th percentile of the in-distribution score.

### Datasets

| Dataset | Role | Images | Classes |
|---|---|---|---|
| UC Merced Land Use | In-distribution (ID) | 2,100 aerial photos | 21 (airplane, forest, beach, ...) |
| FIDS30 | Out-of-distribution (OOD) | 971 fruit photos | 30 (apple, banana, mango, ...) |

UC Merced was split 70 / 15 / 15% stratified by class into train (1,470), val (315), and test (315) sets. FIDS30 was kept as a flat OOD pool and was never used during training.

---

### Method 1: Deep Ensemble

#### Setup

Ten EfficientNet-B0 models were fine-tuned independently on the UC Merced training set. Each model was initialised with ImageNet-pretrained weights and trained for 5 epochs using a different random seed, AdamW optimizer, lr = 1e-4, batch size 64. The models share the same architecture and data but different in weight initialisation and data shuffling order.

Final validation accuracies ranged from **89.8% to 95.9%** across the 10 models, showing that each individual model learned the classification task well.

#### OOD Score: Prediction Variance

The intuition behind this method is that a well-trained classifier is confident (low variance across ensemble members) on images it was trained on, but uncertain (high variance) on images from a different distribution. For each image, all 10 models produce a 21-dimensional softmax probability vector. The OOD score is computed as:

```
score(x) = mean over classes [ variance over models of p_k(y=c | x) ]
```

#### Results

| Split | Mean variance | Detection rate | False alarm rate |
|---|---|---|---|
| UC Merced test (ID) | 0.0027 | = | = |
| FIDS30 (OOD) | 0.0072 (2.7x higher) | = | = |
| Threshold = 0.0104 (95th pct of ID) | = | **13.1%** | 5.1% |

I found the ensemble struggled here. Despite FIDS30 having a 2.7x higher mean variance, the score distributions overlap heavily and only 13.1% of FIDS30 images were correctly flagged as OOD. It was only marginally above the false alarm rate and far from useful in practice.

---

### Method 2: Convolutional Autoencoder

#### Setup

A single convolutional autoencoder (186,371 parameters) was trained on UC Merced using only pixel reconstruction = no class labels are needed at any point.

| Stage | Layers | Channels | Spatial resolution |
|---|---|---|---|
| Encoder | 3 x Conv2d (stride 2) + ReLU | 3 -> 32 -> 64 -> 128 | 256 -> 32 |
| Decoder | 3 x ConvTranspose2d (stride 2) + ReLU | 128 -> 64 -> 32 -> 3 | 32 -> 256 |
| Output | Sigmoid | = | 256 x 256 |

Training used MSE loss, Adam optimizer (lr = 1e-3), batch size 32, for 15 epochs. Validation loss decreased steadily from 0.01238 to 0.00264, indicating clean convergence without overfitting.

#### OOD Score: Reconstruction Error

The autoencoder learns a compressed representation of the training distribution. In-distribution images can be reconstructed well because the latent space was optimised for aerial imagery. OOD images look nothing like the training data, so the decoder produces a blurry or incorrect reconstruction and the MSE is high.

```
score(x) = mean over pixels and channels of ( x - AE(x) )^2
```

#### Results

| Split | Mean MSE | Detection rate | False alarm rate |
|---|---|---|---|
| UC Merced test (ID) | 0.00258 | = | = |
| FIDS30 (OOD) | 0.01094 (4.2x higher) | = | = |
| Threshold = 0.00623 (95th pct of ID) | = | **71.7%** | 5.1% |

The autoencoder performed dramatically better, detecting 71.7% of FIDS30 images correctly at the same false alarm rate.

---

### Comparison

| Criterion | Deep Ensemble | Autoencoder |
|---|---|---|
| OOD score | Softmax variance across 10 models | Reconstruction MSE |
| Threshold | 0.0104 | 0.00623 |
| Detection rate | 13.1% | **71.7%** |
| False alarm rate | 5.1% | 5.1% |
| Mean score ratio (OOD / ID) | 2.7x | 4.2x |
| Requires class labels | Yes | No |
| Models trained | 10 | 1 |
| Total training epochs | 50 | 15 |
| Total parameters | ~40 M | ~186 k |
| Inference cost | 10 forward passes per image | 1 forward pass per image |

### Discussion

The autoencoder is the clear winner, with a detection rate of 71.7% versus only 13.1% for the ensemble, at the same false alarm rate. The ensemble's weakness stems from its pretrained backbone: EfficientNet-B0 still assigns confident, peaked softmax outputs to fruit images, so inter-model variance stays low and the OOD signal is weak. The autoencoder must actually reconstruct the image, and since fruit textures are completely absent from its training distribution, the MSE rises sharply on FIDS30.

The ensemble would be the better choice in settings that require both in-distribution classification and uncertainty estimates. For pure OOD detection with no labelling budget, the autoencoder is both more effective and far cheaper to train and run.
