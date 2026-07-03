# Unit 4: From Autoencoders to Variational Autoencoders

Train fully-connected autoencoders and VAEs on MNIST with low-dimensional latent
spaces (2-D and 3-D). Visualize the learned latent space and show why a plain AE
is not a generative model — and how a VAE fixes that.

## Notebooks

* **`00_FC_Autoencoder_2D.ipynb`**: Fully-connected AE with a 2-D bottleneck. Latent scatter, decoded manifold grid, random sampling.
* **`01_FC_Autoencoder_3D.ipynb`**: Same FC AE with a 3-D bottleneck. 3D scatter and pairwise 2D projections of the test set.
* **`02_VAE_2D.ipynb`**: Fully-connected VAE with a 2-D bottleneck. Side-by-side AE-vs-VAE latent space and decoder samples.
* **`03_VAE_3D.ipynb`**: Same VAE with a 3-D bottleneck, compared against the 3-D AE.

## Usage

1. Run `uv sync` to install dependencies.
2. Run the AE notebooks first (`00`, `01`) — they save weights and latents that the VAE notebooks load for the comparison.
3. Run the VAE notebooks (`02`, `03`) for the AE-vs-VAE side-by-side plots.
