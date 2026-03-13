# D5_MFC4_On-Kernel-based-Variational-Autoencoder

This repository contains implementing Variational Autoencoders (VAE) and EVAE variants.


## Team Members
*   **[Bodapati Sai krishna]** - [CB.SC.U4AIE24308]
*   **[Dokku Naga Shiva]** - [ CB.SC.U4AIE24315]
*   **[Enku Sai Mohith ]** - [CB.SC.U4AIE24316 ]
*   **[Inaganti Mahalakshmi ]** - [CB.SC.U4AIE24322]

## Reference Paper
*   [On Kernel-based Variational Autoencoder](https://arxiv.org/abs/2405.12783)

## Project Outline
The goal of this project is to explore and implement Variational Autoencoders (VAE) and their variants (EVAE).
*   **Objective**: To implement and evaluate the **Epanechnikov Variational Autoencoder (EVAE)** and to identify its improvements over standard VAEs in generating less noisy and sharper images by leveraging kernel-based posterior approximations.
*   **Methodology**:
    *   **Kernel-Based Posterior**: Bridging VAEs and Kernel Density Estimations (KDEs) to approximate the posterior.
    *   **Epanechnikov Kernel**: Implementing the Epanechnikov kernel for the KDE as it minimizes the derived upper bound of the KL divergence in the ELBO asymptotically.
    *   **Optimization**: Using "location-scale" reparametrization tricks to efficiently optimize the new ELBO.
    *   **Evaluation**: Comparing EVAE with standard VAE on the MNIST dataset using metrics like Loss, FID score, and Sharpness.
*   **Current Status**: Implementation of VAE and EVAE on the MNIST dataset.

---

## Objective

- Implement the kernel-based EVAE from the referenced paper and integrate it into our PyTorch notebooks.
- Compare reconstruction and sample quality between standard VAE and EVAE on MNIST (qualitative and quantitative metrics).
- Demonstrate the behavior of kernel-based posterior approximations on simple toy problems to build intuition.

## Motivation / Why the project is interesting

- Standard VAEs commonly use simple Gaussian approximate posteriors (qφ(z|x)), which can lead to blurry samples and limited expressivity when the true posterior is multi-modal or heavy-tailed.
- Kernel-based approaches (as in the referenced paper) allow one to approximate the posterior density with a KDE built from learned "location-scale" transformed samples. This can increase the expressiveness of qφ and tighten the ELBO in practice.
- The Epanechnikov kernel is of particular interest because, under the assumptions in the paper, it minimizes an upper bound on the KL divergence in the ELBO asymptotically — suggesting better theoretical and practical behavior.
- Improved posterior approximations can yield sharper, more realistic samples and better downstream performance in tasks like anomaly detection or image restoration.

## Methodology (mathematical techniques and simple explanation)

High-level setup and notation:
- x: observed data
- z: latent variable
- pθ(x|z): decoder (likelihood)
- p(z): prior (usually standard Normal)
- qφ(z|x): encoder / approximate posterior

Standard ELBO:
ELBO(θ, φ; x) = E_{z ~ qφ(z|x)}[log pθ(x|z)] - KL(qφ(z|x) || p(z))

Kernel-based posterior (informal):
- Instead of a single parametric qφ (e.g., diagonal Gaussian), form q̂φ(z|x) as a kernel density estimate built from S samples obtained via a learned location-scale transform of base noise.
- Example KDE form:
  q̂φ(z|x) = (1/S) ∑_{s=1}^S K_h(z - μ_{φ,s}(x))
  where K_h is a kernel with bandwidth h and μ_{φ,s}(x) are learnable locations (or transformed noise samples).

Epanechnikov kernel (d-dimensional, compact support):
- K(u) ∝ (1 - ||u||^2) for ||u|| ≤ 1, and 0 otherwise.
- It has optimal mean-square error properties among kernels with compact support and, according to the paper, minimizes a derived upper bound on the KL divergence in the ELBO asymptotically.

Location-scale reparameterization:
- Use a base noise ε ~ p(ε) (e.g., standard Normal or Uniform), and a differentiable transform z = μφ(x) + σφ(x) ⊙ ε to obtain samples that enter the KDE.
- This enables backpropagation through the sample generation and through the KDE log-density terms via score-function or reparameterization approximations; the paper discusses efficient ways to optimize this objective.

Toy demonstration recipe (recommended, small and interpretable):
1. Toy dataset: 1D mixture of Gaussians (e.g., two well-separated modes).
2. Small encoder/decoder (MLP with 1–2 hidden layers) and latent dim = 1.
3. Train:
   - VAE baseline with Gaussian qφ.
   - EVAE variant with KDE q̂φ using Epanechnikov kernel and S = 16 samples per example for the KDE.
4. Visualize:
   - Learned q(z|x) for a fixed x (plot KDE vs Gaussian).
   - Reconstructed likelihood pθ(x|z) samples.
   - Compare marginal pθ(x) samples from generative process.
5. Expected observation:
   - EVAE should better capture multi-modal posterior shapes and produce sharper reconstructions or better sample diversity in such toy settings.

## Notebooks

*   **`MFC_VAE.ipynb`**: Implementation of a standard Variational Autoencoder (VAE).
*   **`MFC_EVAE.ipynb`**: Implementation of an EVAE model (kernel-based posterior, Epanechnikov kernel).

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
    *   Metrics like FID and Sharpness may be calculated.

## Results & discussion

- Implementation status: VAE and kernel-based EVAE variants are implemented in the provided notebooks and can be trained on MNIST.
- Qualitative observations (from initial runs and expected from the theory):
  - EVAE's kernel-based posterior often produces sharper, less noisy generated images compared to a diagonal-Gaussian VAE in visual comparisons, especially when the true posterior is non-Gaussian.
  - The Epanechnikov kernel yields a compact support KDE which helps concentrate probability mass and can reduce spurious blurring that arises from overly smooth Gaussian qφ.
- Quantitative evaluation:
  - Recommended metrics: ELBO components (reconstruction loss, KL), FID for sample quality, and a sharpness metric (e.g., variance of Laplacian).
  - Theoretical claims from the paper: the Epanechnikov kernel minimizes a certain upper bound on the KL term asymptotically; empirical evaluation should verify whether this translates to better held-out likelihoods or improved FID in practice.
- Caveats:
  - KDE-based posteriors require additional compute (multiple samples per data point for KDE construction) and careful tuning of kernel bandwidth, number of samples S, and architectural choices.
  - Results may vary with dataset complexity (MNIST is relatively simple; more challenging datasets will reveal strengths/limitations more clearly).



## Releases

This project includes the following releases:

### Latest Release: eval_2
- **Tag**: `eval_2` (Latest)
- **Date**: 4 minutes ago (2026-03-13)
- **Description**: Final release containing END SEMESTER REPORT and related codes and files.
- **Assets**: 
  - Final_EVAE_Doc.mlx (1.01 MB)
  - MFC_EVAE.ipynb (471 KB)
  - MFC_VAE.ipynb (466 KB)
  - PILAE_MNIST.mlx (9.69 KB)
  - PIL_VAE5.mlx (18.7 KB)
  - Source code (zip and tar.gz)

### Release: eval_1
- **Tag**: `eval_1`
- **Date**: January 31, 2026
- **Description**: Initial evaluation release with foundational code and notebooks.
- **Assets**: 5 assets included

### Release: Why Pseudo-Inverse Fails in EVAE
- **Tag**: `pinv_check_in_evae`
- **Date**: January 31, 2026
- **Description**: Analysis and documentation on why Pseudo Inverse cannot be used in standard EVAE. As an extension, work is being done on a new architecture to incorporate Pseudo Inverse as an alternative solution rather than making changes to EVAE. The new release will include related codes and documentation required.
- **Assets**: 
  - evae vs pinv.pdf (Documentation)

All releases are available in the [Releases section](https://github.com/bodapatisaikrishna/D5_MFC4_On-Kernel-based-Variational-Autoencoder/releases) of this repository.
