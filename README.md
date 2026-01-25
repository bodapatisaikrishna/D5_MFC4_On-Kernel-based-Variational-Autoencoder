# D5_MFC4_On-Kernel-based-Variational-Autoencoder



This repository contains Jupyter Notebooks implementing Variational Autoencoders (VAE) and EVAE variants using PyTorch, trained on the MNIST dataset.

## Team Members
*   **[Bodapati Sai krishna]** - [CB.SC.U4AIE24308]
*   **[Dokku Naga Shiva]** - [ CB.SC.U4AIE24315]
*   **[Enku Sai Mohith ]** - [CB.SC.U4AIE24316 ]
*   **[Inaganti Mahalakshmi ]** - [CB.SC.U4AIE24322]


## Reference Paper
*   [On Kernel-based Variational Autoencoder](https://arxiv.org/abs/2405.12783)

## Project Outline
The goal of this project is to explore and implement Variational Autoencoders (VAE) and their variants (EVAE).
*   **Objective**: To implement and evaluate the **Epanechnikov Variational Autoencoder (EVAE)**, identifying its improvements over standard VAEs in generating less noisy and sharper images by overcoming the limitations of the Gaussian latent space assumption.
*   **Methodology**:
    *   **Kernel-Based Posterior**: Bridging VAEs and Kernel Density Estimations (KDEs) to approximate the posterior.
    *   **Epanechnikov Kernel**: Implementing the Epanechnikov kernel for the KDE as it minimizes the derived upper bound of the KL divergence in the ELBO asymptotically.
    *   **Optimization**: Using "location-scale" reparametrization tricks to efficiently optimize the new ELBO.
    *   **Evaluation**: Comparing EVAE with standard VAE  on the MNIST dataset using metrics like Loss, FID score, and Sharpness.
*   **Current Status**: Implementation of VAE and EVAE on the MNIST dataset.


---

## Technical Documentation

### Notebooks

*   **`MFC_VAE.ipynb`**: Implementation of a standard Variational Autoencoder (VAE).
*   **`MFC_EVAE.ipynb`**: Implementation of an EVAE model.

### Dataset

*   **MNIST**: The notebooks are configured to use the MNIST dataset for training and generation.

### Requirements

The code requires the following Python libraries:

*   `torch` (PyTorch)
*   `torchvision`
*   `matplotlib`
*   `numpy`
*   `jupyter`

### Usage

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
    *   Metrics like FID and Sharpness may be calculated.

### Results

The notebooks visualize:
*   Training Loss
*   Reconstructed Images
*   Generated Samples from the latent space
