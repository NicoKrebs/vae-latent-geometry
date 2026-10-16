# Topology and Information Geometry of VAE Latent Spaces

Code for my Bachelor's thesis, which uses tools from differential and information geometry to study what the latent spaces of variational autoencoders (VAEs) actually look like.

**Core idea:** a trained decoder induces a Riemannian metric on the latent space, the Fisher information metric. From this metric we compute the Gaussian curvature *K* by automatic differentiation, and we check numerically whether the **Chern–Gauss–Bonnet theorem** holds:

$$\int_M K \, dA = 2\pi\,\chi(M)$$

This turns the topology of a learned latent space into a quantity you can measure.

## What's in here

- **Topologically constrained VAEs** with latent spaces on the torus T² and the sphere S², plus a flat Euclidean baseline (2D and 3D latent).
- **Synthetic datasets** whose true generative factors live on a plane, a torus, or a sphere. This makes it possible to test whether a model recovers the right topology.
- **A geometry engine** (`vae_topology/geometry.py`) that computes the Fisher metric, Christoffel symbols, the Riemann tensor and the Gaussian curvature via autodiff. It includes a Gauss–Bonnet verification suite that is validated against metrics with known closed-form answers (flat, spherical, hyperbolic, torus).
- **Diagnostics** such as curvature maps, latent scatter plots, validation curves and GIFs of how the latent space and its curvature evolve during training.

## Repository structure

```
vae_topology/              Python package
  datasets.py              synthetic plane / torus / sphere image datasets
  distributions.py         wrapped-normal (circle) and projected-normal (sphere) posteriors
  geometry.py              Fisher metric, curvature, Gauss–Bonnet checks
  models/                  Euclidean, torus and sphere VAEs
  diagnostics/             curvature and latent-space plots
  experiments/             training and evaluation scripts
Master_Experiments.ipynb   main notebook: runs and analyzes all experiments
Diagnose_Torus_Collapse.ipynb
notebooks/                 analysis notebooks used for the thesis figures
colab/                     original Colab notebooks + guide for running on a Colab GPU from VS Code
legacy/                    early prototypes, kept for reference
docs/DEVELOPMENT_NOTES.md  detailed technical notes: design, verification, run storage
```

## Getting started

```bash
git clone https://github.com/NicoKrebs/vae-latent-geometry.git
cd vae-latent-geometry
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

**Easiest way in:** open `Master_Experiments.ipynb` and run the cells from top to bottom. The default epoch counts are small, so a smoke run finishes in about a minute on a CPU. The thesis results used 1,000–10,000 epochs on a GPU.

**From the command line** (run these from the repository root):

```bash
python -m vae_topology.experiments.run_euclidean_2d        # flat VAE, 2D latent, all datasets
python -m vae_topology.experiments.run_all_topologies      # torus + sphere VAEs, all six combinations
python -m vae_topology.experiments.run_euclidean_3d        # flat VAE, 3D latent
```

Every script accepts `--epochs`, `--seed`, `--device` and `--only <key>`. Run a script with `--help` to see all options. Results are saved under `results/`.

## Background

- Bachelor's thesis in physics: *Topology and Information Geometry of Latent Spaces of Generative Models*. A paper based on the thesis is in preparation.
- Related work: Falorsi et al. (2018), *Explorations in Homeomorphic Variational Auto-Encoding*. Arvanitidis et al. (2018), *Latent Space Oddity: On the Curvature of Deep Generative Models*.

## Author

**Nico Krebs**, M.Sc. student in Theoretical Physics at LMU Munich
