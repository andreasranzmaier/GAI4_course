# Unit 1: Multi-Class Image Classification (FIDS30)

Train an **EfficientNet-B0** from scratch on the FIDS30 dataset (30 fruit classes).

## Notebooks

* **`00_PrepData.ipynb`**: Downloads FIDS30 and splits into Training/Validation/Test.
* **`01_TrainModel.ipynb`**: Basic training loop (15 epochs).
* **`02_Inference.ipynb`**: Confusion matrix and classification report.
* **`03_Debug_Model.ipynb`**: Top-5 logits, saliency maps, occlusion sensitivity.
* **`04_Inspect_GradCAM.ipynb`**: Grad-CAM and Grad-CAM++ visualizations.
* **`05_TrainModel_EarlyStop.ipynb`**: Training with early stopping and loss/accuracy curves.

## Usage

1. Run `uv sync` to install dependencies.
2. Run `00_PrepData.ipynb` first (downloads the dataset automatically).
3. Run any training notebook, then inference/debug notebooks.
