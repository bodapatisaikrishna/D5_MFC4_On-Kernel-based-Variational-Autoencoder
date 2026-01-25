# D5_MFC4_On-Kernel-based-Variational-Autoencoder

This repository contains Jupyter Notebooks implementing Variational Autoencoders (VAE) and EVAE variants using PyTorch, trained on the MNIST dataset.

## Notebooks

*   **`MFC_VAE.ipynb`**: Implementation of a standard Variational Autoencoder (VAE).
*   **`MFC_EVAE.ipynb`**: Implementation of an EVAE model.

## Dataset

*   **MNIST**: The notebooks are configured to use the MNIST dataset for training and generation.

## Requirements

The code requires the following Python libraries:

*   `torch` (PyTorch)
*   `torchvision`
*   `matplotlib`
*   `numpy`
*   `jupyter`

## Usage

1.  Ensure you have the required dependencies installed.
2.  Start the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```
3.  Open `MFC_VAE.ipynb` or `MFC_EVAE.ipynb`.
4.  Run all cells to train the models.
    *   The notebooks will download the MNIST dataset automatically if not present.
    *   Training progress and loss metrics will be displayed.
    *   Generated images and reconstruction samples will be shown after each epoch.
    *   Metrics like FID and Sharpness will be calculated.

## Results

The notebooks visualize:
*   Training Loss
*   Reconstructed Images
*   Generated Samples from the latent space
