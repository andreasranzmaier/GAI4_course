# Unit 3: Denoising Autoencoder & Anomaly Detection

PyTorch port of the Keras [autoencoder tutorial](https://keras.io/examples/vision/autoencoder/).
Train a convolutional denoising autoencoder on MNIST, then use reconstruction error to detect
Fashion-MNIST images as anomalies.

## Notebooks

* **`00_PrepData.ipynb`**: Downloads MNIST + Fashion-MNIST via torchvision, shows samples.
* **`01_DenoisingAutoencoder.ipynb`**: Trains a Conv autoencoder — first reconstruction, then denoising.
* **`02_AnomalyDetection.ipynb`**: Uses reconstruction error to flag Fashion-MNIST as anomalies.

## Usage

1. Run `uv sync` to install dependencies.
2. Run notebooks sequentially (00 → 01 → 02).
