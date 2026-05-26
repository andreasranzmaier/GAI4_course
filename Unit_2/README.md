# Unit 2: Uncertainty Estimation, Bayesian Deep Learning

Train an **EfficientNet-B0** on FIDS30 (30 fruit classes) and compare two ways of
quantifying predictive uncertainty: **MC Dropout** and **Deep Ensembles**.

## Notebooks

* **`00_PrepData.ipynb`**: Downloads FIDS30 and splits into Training/Validation/Test.
* **`01_TrainModel.ipynb`**: Fine-tune one EfficientNet-B0 for 30-class classification.
* **`02_MC_Dropout.ipynb`**: Keep dropout on at inference, run T stochastic passes
  (T = 1, 2, 3, 5, 8, 13, 21), use the variance across passes as uncertainty.
* **`03_DeepEnsemble_vs_MCDropout.ipynb`**: Train 10 models with different seeds and
  compare a 10-model ensemble against 10 MC-Dropout passes of a single model and compare accuracy *and* uncertainty calibration.

## Usage

1. Run `uv sync` to install dependencies.
2. Run `00_PrepData.ipynb` first (downloads the dataset automatically).
3. Run `01_TrainModel.ipynb`, then `02_MC_Dropout.ipynb`.
4. `03_DeepEnsemble_vs_MCDropout.ipynb` trains the 10-model ensemble.
