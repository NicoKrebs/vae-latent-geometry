"""A VAE with an S^2 latent space.

Probabilistic model:
    prior      p(theta)   = Uniform(S^2)
    posterior  q(theta|x) = ProjectedNormal(mu(x), sigma(x)) via the
                             exponential map at mu(x) (see
                             `vae_topology.distributions.exp_map` /
                             `tangent_normal_log_prob` for the derivation)
    likelihood p(x|theta) = N(dec(theta), sigma_x^2 I),  sigma_x learned scalar

`encode` returns the posterior mean as a UNIT 3-VECTOR mu (not angles): this
avoids ever having to differentiate through the (theta, phi) chart's pole
singularity during training. Angles are only recovered on demand, via
`encode_angles`, for plotting/evaluation.
"""

import math

import torch
import torch.nn as nn

from vae_topology.distributions import (
    FOUR_PI, angles_from_vec, exp_map, kl_tangent_normal_to_uniform,
)


class SphereVAE(nn.Module):

    def __init__(self, data_dim=784, hidden=256):
        super().__init__()
        self.latent_dim = 2
        self.data_dim = data_dim

        self.min_log_sigma = math.log(1.0)
        self.max_log_sigma = math.log(3.0)

        self.backbone = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )

        self.head_dir = nn.Linear(hidden, 3)
        self.head_log_sigma = nn.Linear(hidden, 1)

        self.decoder_net = nn.Sequential(
            nn.Linear(3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, data_dim),
        )

        self.log_sigma_x = nn.Parameter(torch.zeros(()))

    def encode(self, x):
        h = self.backbone(x)
        v = self.head_dir(h)
        mu = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        log_sigma = self.head_log_sigma(h).clamp(self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma

    def encode_angles(self, x):
        """(theta, phi) form of encode(), for plotting only."""
        return angles_from_vec(self.encode(x)[0])

    def sample_posterior(self, mu, log_sigma):
        """Project isotropic Gaussian noise onto the tangent plane at mu
        (removing its mu-component), scale it, and push it onto the sphere
        via the exponential map. Returns (posterior sample, tangent vector);
        the tangent vector is what `tangent_normal_log_prob` needs."""
        eps = torch.randn_like(mu)
        eps = eps - (eps * mu).sum(-1, keepdim=True) * mu
        v = torch.exp(log_sigma) * eps
        return exp_map(mu, v), v

    def decode_vec(self, z):
        """Decode from a raw unit 3-vector (what sample_posterior/encode
        produce)."""
        return self.decoder_net(z)

    def decode(self, u):
        """Decode from (theta, phi) angles -- the interface `geometry.py`'s
        autodiff (jacfwd w.r.t. a 2-vector) needs."""
        from vae_topology.distributions import sphere_embed
        return self.decoder_net(sphere_embed(u))

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z, v = self.sample_posterior(mu, log_sigma)
        return self.decode_vec(z), mu, log_sigma, v


def negative_elbo(model, x, beta=1.0):
    x_recon, mu, log_sigma, v = model(x)

    ls_x = model.log_sigma_x.clamp(-6.0, 2.0)
    d = model.data_dim
    recon = (0.5 * ((x - x_recon) ** 2).sum(dim=-1) * torch.exp(-2.0 * ls_x)
             + d * ls_x + 0.5 * d * math.log(2.0 * math.pi)).mean()

    kl = kl_tangent_normal_to_uniform(v, log_sigma)

    return recon + beta * kl, recon.item(), kl.item()


def train(
    epochs=400,
    lr=1e-3,
    beta=1.0,
    batch_size=256,
    sigma_floor_start=1.0,
    sigma_floor_end=0.1,
    sigma_anneal_epochs=50,
    grad_clip=5.0,
    data_fn=None,
    model_cls=SphereVAE,
    device=None,
    seed=42,
    eval_hook=None,
    eval_every=0,
    log_every=0,
    **model_kwargs,
):
    """Train a SphereVAE. Returns (model, data, angles, history).

    `history` is a list of per-epoch dicts (loss, recon, kl, sigma_floor,
    sigma_x) -- added for parity with `models.torus_vae.train`, which
    already tracked this; the original sphere trainer discarded it, so a run
    could only be inspected live (via `eval_hook`) and not after the fact.
    Combined with `experiments.run_store.RunStore.save_checkpoint` (called
    from `eval_hook` at the same cadence as `eval_every`), this is what makes
    it possible to regenerate a training-curve or curvature plot for a past
    run without retraining.
    """
    if data_fn is None:
        from vae_topology.datasets import make_sphere_dataset_new
        data_fn = make_sphere_dataset_new
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    imgs, angles = data_fn()
    data = torch.as_tensor(imgs, dtype=torch.float32, device=device)
    if data.dim() > 2:
        data = data.flatten(1)

    model = model_cls(data_dim=data.shape[1], **model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = data.shape[0]
    batch_size = min(batch_size, n)
    anneal_epochs = max(1, sigma_anneal_epochs)
    ls_hi, ls_lo = math.log(sigma_floor_start), math.log(sigma_floor_end)

    history = []
    for ep in range(epochs):
        model.train()
        t = min(1.0, ep / anneal_epochs)
        model.min_log_sigma = ls_hi + t * (ls_lo - ls_hi)

        perm = torch.randperm(n, device=device)
        tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        for i in range(0, n, batch_size):
            xb = data[perm[i:i + batch_size]]
            loss, r, k = negative_elbo(model, xb, beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            w = xb.shape[0]
            tot["loss"] += loss.item() * w
            tot["recon"] += r * w
            tot["kl"] += k * w

        rec = {key: val / n for key, val in tot.items()}
        rec.update(epoch=ep, beta=beta,
                   sigma_floor=math.exp(model.min_log_sigma),
                   sigma_x=math.exp(model.log_sigma_x.item()))
        history.append(rec)

        if eval_hook is not None and eval_every and (ep % eval_every == 0
                                                     or ep == epochs - 1):
            eval_hook(model, ep)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d}  loss={rec['loss']:.3f}  "
                  f"recon={rec['recon']:.3f}  kl={rec['kl']:.3f}  "
                  f"sigma_floor={rec['sigma_floor']:.3f}  "
                  f"sigma_x={rec['sigma_x']:.3f}")

    return model, data, angles, history
