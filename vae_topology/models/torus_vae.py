"""A VAE with a 2-torus latent space, built so that the Fisher information
metric of the decoder distribution p(x|theta) is well defined and can be
differentiated (at least) twice more, for curvature computations.

Probabilistic model:
    prior      p(theta)     = Uniform(T^2)
    posterior  q(theta|x)   = WrappedNormal(mu(x), sigma(x)) per angle
    likelihood p(x|theta)   = N(dec(theta), sigma_x^2 I),  sigma_x learned scalar

Fisher metric of the likelihood in theta:
    G(theta) = J(theta)^T J(theta) / sigma_x^2,   J = d dec / d theta.
Since sigma_x is constant in theta, G is a constant conformal rescaling of
the Euclidean pullback J^T J: Christoffel symbols are identical, and the
Gaussian curvature rescales as K_Fisher = sigma_x^2 * K_pullback (see
`geometry.py`).

Smoothness: theta -> dec(theta) uses only sin/cos, Linear and Tanh (all
C-infinity). No clamp, wrap, or piecewise operation appears anywhere in the
decode path, so arbitrarily many derivatives are safe (see the smoothness
note in `geometry.py`: K needs three).

This is the final, "production" torus model. It is the third iteration of
this idea in the original project: `legacy/torvae_charts.py` (first working
chart-based version) was refined into `legacy/first_fvae.py` (adds the
annealed-sigma-floor fix for a real winding-number bug -- read that file's
header for the diagnosis), which was refined into this file (adds a properly
normalized Gaussian reconstruction likelihood via a learned sigma_x, and
switches the KL from a truncated Fourier series to a direct wrapped-normal
log-prob -- see `vae_topology/distributions.py`).
"""

import math

import torch
import torch.nn as nn

from vae_topology.distributions import (
    TWO_PI, circle_embed, kl_wrapped_to_uniform, wrap_angle,
)


class TorusVAE(nn.Module):
    def __init__(self, data_dim=784, latent_dim=2, hidden=256):
        super().__init__()
        if latent_dim != 2:
            raise ValueError("This model is specifically for a 2-torus.")
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.min_log_sigma = math.log(1.0)
        self.max_log_sigma = math.log(3.0)

        self.backbone = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )

        self.head_dir = nn.Linear(hidden, 2 * latent_dim)
        self.head_log_sigma = nn.Linear(hidden, latent_dim)

        self.decoder_net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, data_dim),
        )

        self.log_sigma_x = nn.Parameter(torch.zeros(()))

    def encode(self, x):
        h = self.backbone(x)
        v = self.head_dir(h).view(*x.shape[:-1], self.latent_dim, 2)
        mu = torch.atan2(v[..., 1], v[..., 0])
        log_sigma = self.head_log_sigma(h).clamp(self.min_log_sigma,
                                                 self.max_log_sigma)
        return mu, log_sigma

    def sample_posterior(self, mu, log_sigma):
        eps = torch.randn_like(mu)
        return wrap_angle(mu + torch.exp(log_sigma) * eps)

    def decode(self, theta):
        return self.decoder_net(circle_embed(theta))

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        theta = self.sample_posterior(mu, log_sigma)
        return self.decode(theta), mu, log_sigma, theta


def negative_elbo(model, x, beta=1.0):
    x_recon, mu, log_sigma, theta = model(x)

    ls_x = model.log_sigma_x.clamp(-6.0, 2.0)
    d = model.data_dim
    recon = (0.5 * ((x - x_recon) ** 2).sum(dim=-1) * torch.exp(-2.0 * ls_x)
             + d * ls_x + 0.5 * d * math.log(TWO_PI)).mean()

    kl = kl_wrapped_to_uniform(theta, mu, log_sigma)

    return recon + beta * kl, recon.item(), kl.item()


def train(
    epochs=400,
    lr=1e-3,
    beta=1.0,
    batch_size=256,
    sigma_floor_start=1.0,
    sigma_floor_end=0.02,
    sigma_anneal_epochs=500,
    grad_clip=5.0,
    data_fn=None,
    model_cls=TorusVAE,
    device=None,
    seed=42,
    log_every=50,
    eval_hook=None, eval_every=0,
    **model_kwargs,
):
    """Train a TorusVAE. Returns (model, data, angles, history).

    beta is constant. Instead, the *floor* of the posterior noise is annealed
    log-linearly from sigma_floor_start to sigma_floor_end over the first
    `sigma_anneal_epochs` epochs: winding numbers of the encoder are
    discrete and can only change through tears, and a tear is only
    penalized where posterior samples straddle it -- i.e. with weight
    proportional to sigma. Large early noise keeps the map deformable until
    it settles into a single consistent winding, then the floor is
    released. (This is the fix diagnosed at length in
    `legacy/first_fvae.py`'s header comment.)

    `sigma_anneal_epochs` is an ABSOLUTE epoch count, matching
    `models.sphere_vae.train`'s parameterization -- it used to be
    `sigma_anneal_frac * epochs` (half of whatever total epoch count was
    requested), which meant training for 200 vs. 1000 epochs annealed over
    100 vs. 500 epochs respectively: two different schedules, not a short
    run and a continuation of it, since the noise floor at any given early
    epoch differed between the two. That made whether the encoder healed or
    stayed torn sensitive to the total epoch BUDGET you asked for, not just
    how long you actually trained -- fixed by making the window absolute, so
    the early trajectory no longer depends on `epochs`.

    Note this changes the actual anneal duration for long runs relative to
    the old default: previously `epochs=1000` implied a 500-epoch anneal
    window; now it's 50 by default, same as `sphere_vae.train`. A shorter
    anneal gives a tear less time to heal, which can make one MORE likely on
    long runs, not less -- pass `sigma_anneal_epochs=500` explicitly to
    reproduce the old ~1000-epoch behavior if you need to compare against
    earlier results.
    """
    if data_fn is None:
        from vae_topology.datasets import make_torus_dataset
        data_fn = make_torus_dataset
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
        if eval_hook is not None and eval_every and (ep % eval_every == 0 or ep == epochs - 1):
            eval_hook(model, ep)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d}  loss={rec['loss']:.3f}  "
                  f"recon={rec['recon']:.3f}  kl={rec['kl']:.3f}  "
                  f"sigma_floor={rec['sigma_floor']:.3f}  "
                  f"sigma_x={rec['sigma_x']:.3f}")

    return model, data, angles, history
