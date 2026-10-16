# ARCHIVED -- kept for the historical record, not part of the live pipeline.
#
# Status: contains a real, easy-to-miss bug, preserved as-is. The first ~50
# lines define a local `wrap()` (and `log_prob_wrapped`, `kl_wrapped_to_uniform`,
# etc.), but the later `from projects.bin_scripts.torvae_charts import wrap, ...`
# rebinds the module-level name `wrap` to torvae_charts' version (a different
# angle convention: [-pi, pi) centred vs. this file's own [0, 2*pi)). Every
# use of `wrap` below that import point is silently the imported one -- the
# first ~50 lines are dead code. Left uncorrected here since this is a copy
# of what was actually run, not a fixed version.
#
# Otherwise: explores a *different* architectural idea from the rest of the
# torus prototypes -- a learned periodic feature grid (`TorusGrid`,
# bilinearly interpolated on the torus) as the decoder input instead of a
# sin/cos Fourier embedding. Not imported anywhere else in the codebase, and
# not carried forward into vae_topology/models/torus_vae.py; kept here in
# case the grid-representation idea is worth revisiting.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

TWO_PI = 2.0 * math.pi
LOG_SIGMA_MIN = -6.0
LOG_SIGMA_MAX = 1.0      
N_WINDINGS = 3         


def wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x, TWO_PI)


def wrap_centered(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x + math.pi, TWO_PI) - math.pi


def torus_delta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return wrap_centered(a - b)


def rsample_wrapped(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(mu)
    return wrap(mu + eps * log_sigma.exp())


def log_prob_wrapped(z, mu, log_sigma, n_windings: int = N_WINDINGS):
    sigma = log_sigma.exp()
    k = torch.arange(-n_windings, n_windings + 1, device=z.device, dtype=z.dtype)

    d = wrap_centered(z - mu).unsqueeze(-1) + TWO_PI * k          # (..., d, K)
    log_comp = (
        -0.5 * (d / sigma.unsqueeze(-1)).pow(2)
        - log_sigma.unsqueeze(-1)
        - 0.5 * math.log(TWO_PI)
    )
    return torch.logsumexp(log_comp, dim=-1).sum(-1)


def kl_wrapped_to_uniform(z, mu, log_sigma):

    d = mu.shape[-1]
    log_p = -d * math.log(TWO_PI)
    return log_prob_wrapped(z, mu, log_sigma) - log_p



import math
import torch
import torch.nn as nn

from projects.bin_scripts.torvae_charts import wrap, angles_from_unconstrained, kl_uniform, elbo

TWO_PI = 2.0 * math.pi


def smooth_periodic_init(res, channels, max_harmonic=2, scale=0.1):
    th = torch.arange(res, dtype=torch.float32) * TWO_PI / res
    g = torch.zeros(res, res, channels)
    for kx in range(-max_harmonic, max_harmonic + 1):
        for ky in range(-max_harmonic, max_harmonic + 1):
            if kx == 0 and ky == 0:
                continue
            amp = torch.randn(channels) / (1.0 + kx * kx + ky * ky)
            phase = torch.rand(channels) * TWO_PI
            g = g + amp * torch.cos(
                kx * th[:, None, None] + ky * th[None, :, None] + phase
            )
    return scale * g / g.std()


class TorusGrid(nn.Module):
    def __init__(self, res=32, channels=64, max_harmonic=2):
        super().__init__()
        self.res = res
        self.channels = channels
        self.grid = nn.Parameter(smooth_periodic_init(res, channels, max_harmonic))

    def forward(self, theta):
        lead = theta.shape[:-1]
        t = theta.reshape(-1, 2)

        u = torch.remainder(t, TWO_PI) / TWO_PI * self.res
        u0 = u.floor()
        frac = u - u0
        i0 = u0.long() % self.res
        i1 = (i0 + 1) % self.res

        g = self.grid
        c00 = g[i0[:, 0], i0[:, 1]]
        c10 = g[i1[:, 0], i0[:, 1]]
        c01 = g[i0[:, 0], i1[:, 1]]
        c11 = g[i1[:, 0], i1[:, 1]]

        fu = frac[:, 0:1]
        fv = frac[:, 1:2]
        out = (c00 * (1 - fu) + c10 * fu) * (1 - fv) + (c01 * (1 - fu) + c11 * fu) * fv
        return out.reshape(*lead, self.channels)


class gridTorusVAE(nn.Module):
    def __init__(self, data_dim=784, latent_dim=2, hidden=256, res=32, channels=64,
                 log_sigma_min=-4.0, log_sigma_max=2.0, log_sigma_init=-2.5,
                 max_harmonic=2):
        super().__init__()
        assert latent_dim == 2
        self.latent_dim = latent_dim
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.dir = nn.Linear(hidden, 2 * latent_dim)
        self.logsigma = nn.Linear(hidden, latent_dim)

        frac = (log_sigma_init - log_sigma_min) / (log_sigma_max - log_sigma_min)
        nn.init.zeros_(self.logsigma.weight)
        nn.init.constant_(self.logsigma.bias, math.log(frac / (1.0 - frac)))

        self.grid = TorusGrid(res, channels, max_harmonic)
        self.decoder = nn.Sequential(
            nn.Linear(channels, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, data_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        v = self.dir(h).reshape(*h.shape[:-1], self.latent_dim, 2)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        mu = torch.atan2(v[..., 1], v[..., 0])
        raw = self.logsigma(h)
        ls = self.log_sigma_min + (self.log_sigma_max - self.log_sigma_min) * torch.sigmoid(raw)
        return mu, ls

    def reparametrize(self, mu_angles, log_sigma):
        sigma = torch.exp(log_sigma)
        return wrap(mu_angles + sigma * torch.randn_like(sigma))

    def decode(self, theta):
        return self.decoder(self.grid(theta))

    def forward(self, x):
        mu, ls = self.encode(x)
        theta = self.reparametrize(mu, ls)
        return self.decode(theta), mu, ls, theta

    @torch.no_grad()
    def sample(self, n, device=None):
        device = device or next(self.parameters()).device
        theta = (torch.rand(n, self.latent_dim, device=device) - 0.5) * TWO_PI
        return self.decode(theta), theta

    @torch.no_grad()
    def encode_angles(self, x, batch_size=4096):
        was_training = self.training
        self.eval()
        out = [self.encode(x[i:i + batch_size])[0] for i in range(0, x.shape[0], batch_size)]
        if was_training:
            self.train()
        return torch.cat(out, dim=0)

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