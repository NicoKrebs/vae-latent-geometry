# ARCHIVED -- kept for the historical record, not part of the live pipeline.
#
# Status: WORKING, superseded. Earliest working chart-based torus VAE
# (direction-vector encoding via atan2, wrapped-normal KL as a truncated
# Fourier series). Predates the sigma-floor-annealing fix in
# legacy/first_fvae.py, so it is susceptible to the same "torn encoder /
# chessboard latent" failure mode described there. Its `wrap`,
# `angles_from_unconstrained`, `kl_uniform`, `elbo` are imported by
# legacy/torvae_mod.py.
#
# Superseded by vae_topology/models/torus_vae.py; see the note at the top of
# legacy/first_fvae.py for how the lineage connects.

# In the following, we want to implement and train a VAE with a periodic, meaning a toroidal latent space.
# For the implementation, look at the documentation on git Hub

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from projects.visualization.vaediagnostic import run_diagnostics
from datasets import make_torus_dataset
from datasets import make_plane_dataset
from datasets import make_sphere_dataset
 
torch.manual_seed(42)





 
TWO_PI = 2.0 * math.pi
torch.manual_seed(42)
 
 
# --------------------------------------------------------- chart helpers ---
def wrap(theta):
    """Angles into [-pi, pi). Gradient is 1 a.e., so backprop-safe."""
    return (theta + math.pi) % TWO_PI - math.pi
 
 
def angles_from_unconstrained(h):
    """[CHANGE 2] (..., 2k) direction vectors -> (..., k) angles, seam-free."""
    return torch.atan2(h[..., 1::2], h[..., 0::2])
 
 
def torus_embed(theta):
    """The chart: (..., k) angles -> (..., 2k) = (cos t1, sin t1, cos t2, sin t2)."""
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([c, s], dim=-1).flatten(start_dim=-2)
 
 
# ------------------------------------------------------------------ model ---
class torusVAE(nn.Module):
    def __init__(self, data_dim=784, latent_dim=2, hidden=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # [CHANGE 1] 2 outputs PER ANGLE (a direction vector), not 1
        self.dir      = nn.Linear(hidden, 2 * latent_dim)
        self.logsigma = nn.Linear(hidden, latent_dim)      # unchanged: 1 width per angle
        # [CHANGE 3] decoder input is the CHART, dimension 2*latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, data_dim),
        )
 
    def encode(self, x):
        h = self.encoder(x)                                # compute ONCE
        mu = angles_from_unconstrained(self.dir(h))        # (B, k) mean angles
        ls = self.logsigma(h).clamp(-4.0, 2.0)             # sigma in ~[0.018, 7.4]
        return mu, ls
 
    def reparametrize(self, mu_angles, log_sigma):
        """[CHANGE 4] wrapped normal: theta = wrap(mu + sigma*eps)."""
        sigma = torch.exp(log_sigma)
        return wrap(mu_angles + sigma * torch.randn_like(sigma))
 
    def decode(self, theta):
        """[CHANGE 6] takes ANGLES; applies the chart internally."""
        return self.decoder(torus_embed(theta))
 
    def forward(self, x):
        mu, ls = self.encode(x)
        theta = self.reparametrize(mu, ls)
        return self.decode(theta), mu, ls, theta
 
 
# ------------------------------------------------------------------- loss ---
def kl_uniform(theta, mu_angles, log_sigma, n_terms=30):

    sigma = torch.exp(log_sigma)
    d = wrap(theta - mu_angles).unsqueeze(-1)
    n = torch.arange(1, n_terms + 1, device=theta.device, dtype=theta.dtype)
    decay = torch.exp(-0.5 * (sigma.unsqueeze(-1) * n) ** 2)
    series = 1.0 + 2.0 * (decay * torch.cos(n * d)).sum(-1)
    kl_series = torch.log(series.clamp_min(1e-12))
    kl_gauss = math.log(TWO_PI) - 0.5 * torch.log(TWO_PI * math.e * sigma ** 2)
    return torch.where(sigma < 0.5, kl_gauss, kl_series).sum(-1).mean()
 
 
def elbo(x_recon, x, mu, log_sigma, theta, beta=1.0):
    recon = ((x_recon - x) ** 2).sum(dim=1).mean()
    kl = kl_uniform(theta, mu, log_sigma)
    return recon + beta * kl, recon.item(), kl.item()
 
 



def train(
    epochs=400,
    lr=1e-3,
    beta=1.0,
    batch_size=256,
    warmup=100,
    grad_clip=5.0,
    data_fn=None,
    model_cls=None,
    device=None,
    seed=42,
    log_every=50,
    **model_kwargs,
):
    if data_fn is None:
        data_fn = make_torus_dataset
    if model_cls is None:
        model_cls = torusVAE

    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    imgs, angles = data_fn()

    data = torch.as_tensor(imgs, dtype=torch.float32, device=device)
    if data.dim() > 2:
        data = data.flatten(1)

    if angles is not None:
        angles = torch.as_tensor(angles, dtype=torch.float32)

    model = model_cls(data_dim=data.shape[1], **model_kwargs).to(device)
    optimize = torch.optim.Adam(model.parameters(), lr=lr)

    n = data.shape[0]
    batch_size = min(batch_size, n)
    history = []

    for ep in range(epochs):
        model.train()

        beta_ep = beta * min(1.0, (ep + 1) / warmup) if warmup else beta

        perm = torch.randperm(n, device=device)
        tot_loss = tot_recon = tot_kl = 0.0

        for i in range(0, n, batch_size):
            xb = data[perm[i:i + batch_size]]

            x_recon, mu, log_sigma, theta = model(xb)
            loss, r, k = elbo(x_recon, xb, mu, log_sigma, theta, beta_ep)

            optimize.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimize.step()

            w = xb.shape[0]
            tot_loss += loss.item() * w
            tot_recon += r * w
            tot_kl += k * w

        ep_loss, ep_recon, ep_kl = tot_loss / n, tot_recon / n, tot_kl / n
        history.append({"epoch": ep, "loss": ep_loss, "recon": ep_recon,
                        "kl": ep_kl, "beta": beta_ep})

        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d}  loss={ep_loss:.4f}  recon={ep_recon:.4f}  "
                  f"kl={ep_kl:.4f}  beta={beta_ep:.3f}")

    return model, data, angles, history
