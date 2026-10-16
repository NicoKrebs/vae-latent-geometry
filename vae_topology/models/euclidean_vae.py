"""The standard Euclidean-latent VAE: the baseline every topological model
in this package is compared against.

Merged from the original `architecture/standardvae.py` (latent_dim=2) and
`architecture/standardvae_3dim.py` (latent_dim=3, otherwise identical apart
from which diagnostics module its __main__ block calls) into one class with
a `latent_dim` parameter. Nothing about the model itself depends on the
dimension; only which diagnostics apply does (2d latents get the
theta1/theta2 curvature-map diagnostics in
`diagnostics.latent_plots.run_diagnostics`, 3d latents get the point-cloud +
singular-value diagnostics in `diagnostics.latent_plots.run_diagnostics_3d`).

Encoder/decoder use Tanh throughout (not ReLU) so the decoder stays smooth
enough for `geometry.gaussian_curvature_from_metric` (needs three
derivatives) to give a meaningful, nonzero answer -- see the note in
`geometry.py`.
"""

import torch
import torch.nn as nn


class SimpleVAE(nn.Module):
    def __init__(self, data_dim=2, latent_dim=2, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu     = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, data_dim),
        )

    def encode(self, data):
        h = self.encoder(data)
        return self.mu(h), self.logvar(h)

    def reparametrize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        return mu + sigma * torch.randn_like(sigma)

    def decode(self, theta):
        return self.decoder(theta)

    def forward(self, data):
        mu, logvar = self.encode(data)
        theta = self.reparametrize(mu, logvar)
        return self.decode(theta), mu, logvar


def elbo(x_recon, x, mu, logvar, beta=1.0):
    """Standard Gaussian-decoder ELBO with an (implicit, fixed) unit decoder
    variance -- the reconstruction term is a raw sum of squared errors, not
    a normalized Gaussian NLL (contrast with the torus/sphere models, which
    learn an explicit decoder noise scale sigma_x)."""
    recon = ((x_recon - x) ** 2).sum(dim=1).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return recon + beta * kl, recon.item(), kl.item()


def train(epochs=400, lr=1e-3, beta=1.0, dataset=None, latent_dim=2, seed=10,
         device=None, log_every=50, eval_hook=None, eval_every=0):
    """dataset: a zero-argument callable returning (imgs, angles), e.g. any
    `make_*_dataset` from `vae_topology.datasets`.

    Returns (model, data, angles, history); `history` is a list of per-epoch
    {epoch, loss, recon, kl} dicts, for parity with `models.torus_vae.train`
    / `models.sphere_vae.train` -- added so a training curve can be plotted,
    or a checkpoint taken via `eval_hook` (see `experiments.
    validation_curves.make_hook`), without re-running training later.
    `device` is accepted for parity with the GPU-adapted driver in
    `experiments/run_euclidean_3d.py` (`Colab_run_files/latent_3d_colab.ipynb`
    moved training onto a T4 GPU this way); it defaults to CPU, matching the
    original local-only script.
    """
    if dataset is None:
        from vae_topology.datasets import make_plane_dataset_old
        dataset = make_plane_dataset_old
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device(device or "cpu")

    imgs, angles = dataset()
    data = torch.as_tensor(imgs, dtype=torch.float32, device=device)
    model = SimpleVAE(data_dim=data.shape[1], latent_dim=latent_dim).to(device)
    optimize = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    for ep in range(epochs):
        optimize.zero_grad()
        x_recon, mu, logvar = model(data)
        loss, r, k = elbo(x_recon, data, mu, logvar, beta)
        loss.backward()
        optimize.step()
        history.append(dict(epoch=ep, loss=loss.item(), recon=r, kl=k))

        if eval_hook is not None and eval_every and (ep % eval_every == 0 or ep == epochs - 1):
            eval_hook(model, ep)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d}  loss={loss.item():.4f}  recon={r:.4f}  kl={k:.4f}")
    return model, data, angles, history


if __name__ == "__main__":
    from vae_topology.datasets import (
        make_plane_dataset, make_plane_dataset_old, make_torus_dataset,
        make_sphere_dataset_batched,
    )

    # -- 2d latent: theta1/theta2 curvature-map diagnostics ------------------
    from vae_topology.diagnostics.latent_plots import run_diagnostics

    model, data, angles, _ = train(epochs=1000, dataset=make_plane_dataset_old, latent_dim=2)
    run_diagnostics(model, data, angles, grid=100, fname="inv_ff_1.pdf")

    model, data, angles, _ = train(epochs=1000, dataset=make_plane_dataset, latent_dim=2)
    run_diagnostics(model, data, angles, grid=100, fname="inv_ff_bordercase_1.pdf")

    model, data, angles, _ = train(epochs=50, dataset=make_torus_dataset, latent_dim=2)
    run_diagnostics(model, data, angles, grid=100, fname="moebius.pdf")

    # -- 3d latent: point-cloud + singular-value diagnostics ------------------
    # NOTE: run_diagnostics_3d has no `grid` parameter -- passing one raised
    # TypeError in the original standardvae_3dim.py __main__ block. Caught
    # and fixed in Colab_run_files/latent_3d_colab.ipynb; fix applied here.
    from vae_topology.diagnostics.latent_plots import run_diagnostics_3d

    model, data, angles, _ = train(epochs=1000, dataset=make_plane_dataset_old,
                                   latent_dim=3, seed=42)
    run_diagnostics_3d(model, data, angles,
                       fname="latent_3d_flat_with_borders_on_flat.pdf")

    model, data, angles, _ = train(epochs=1000, dataset=make_plane_dataset,
                                   latent_dim=3, seed=42)
    run_diagnostics_3d(model, data, angles, fname="latent_3d_flat_on_flat.pdf")

    model, data, angles, _ = train(epochs=1000, dataset=make_torus_dataset,
                                   latent_dim=3, seed=42)
    run_diagnostics_3d(model, data, angles, fname="latent_3d_torus_on_flat.pdf")

    # NOTE: the sphere's polar angle is on [0, pi], not cyclic -- also fixed
    # in the Colab notebook (a cyclic colormap, e.g. "twilight", would fake a
    # seam at the poles). run_diagnostics's default cyclic=(True, True) is
    # wrong here.
    model, data, angles, _ = train(epochs=5000, dataset=make_sphere_dataset_batched,
                                   latent_dim=3, seed=42)
    run_diagnostics_3d(model, data, angles, cyclic=(False, True),
                       fname="latent_3d_sphere_on_flat.pdf")
