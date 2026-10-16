# ARCHIVED -- kept for the historical record, not part of the live pipeline.
#
# Status: WORKING, superseded. This is the most developed of the pre-final
# torus-VAE prototypes: it diagnoses and fixes a real bug (a "torn" encoder
# whose winding number gets frozen wrong by beta-warmup -- see the note
# below) via annealing the posterior sigma FLOOR instead of beta, plus an
# optional equivariance regularizer. Both ideas carried forward into the
# final model. Not imported anywhere else in the codebase.
#
# Superseded by vae_topology/models/torus_vae.py (from
# architecture/new_tor_vae_charts.py), which keeps the sigma-floor-annealing
# fix, drops the equivariance regularizer (not needed once the fix works),
# switches the KL from the truncated Fourier-series form used here to a
# direct wrapped-normal log-prob (mathematically equivalent, simpler to
# read), and adds a learned decoder noise scale sigma_x for a properly
# normalized Gaussian reconstruction likelihood (this file uses a raw sum of
# squared errors instead).
#
# Torus VAE with chart-based latent space -- FIXED training procedure.
#
# Diagnosis of the "chessboard" plot:
#   The chart maps (torus_embed / angles_from_unconstrained / wrap / kl_uniform)
#   are topologically CORRECT. The chessboard is not a wrong-winding bug in one
#   smooth function; it is a TORN encoder: locally degree-1 patches glued with
#   ~pi offsets, separated by seams (the white bands your latent codes avoid).
#
#   Why it happens: the winding number of the encoder map (data torus -> latent
#   torus) is a discrete homotopy invariant. Gradient descent can only change it
#   through a tear, and tears are free in this parametrization: atan2 lets mu
#   jump wherever the direction vector v(x) passes near 0, unpenalized.
#
#   Why it never heals with the ORIGINAL schedule: beta-warmup (100 epochs of
#   beta ~ 0) lets log_sigma collapse to the clamp floor e^-4 immediately, so
#   sampled thetas never straddle a seam and torn configurations cost zero
#   reconstruction loss. By the time the KL turns on, the topology is frozen.
#
# Fixes implemented here:
#   1. [FIX A] Anneal the posterior sigma FLOOR (large -> small) instead of
#      warming up beta. With sigma ~ 1 rad early on, wrapped-normal samples
#      straddle any seam, so tears produce bad reconstructions and get glued
#      while the map is still soft. beta stays at its final value from epoch 0.
#   2. [FIX B] Optional equivariance regularizer (equiv_weight > 0): rolling
#      the image by s pixels along x is EXACTLY theta1 += 2*pi*s/res for this
#      renderer (rows for theta2). Penalizing
#          1 - cos( mu(roll(x)) - mu(x) - delta )
#      pins the winding number to (1,0)/(0,1) by construction. Off by default.
#   3. direction_norm(): seam diagnostic -- ||v(x)|| dips toward 0 on tears.

import math
import torch
import torch.nn as nn

from datasets import make_torus_dataset

TWO_PI = 2.0 * math.pi
torch.manual_seed(42)


# --------------------------------------------------------- chart helpers ---
def wrap(theta):
    """Angles into [-pi, pi). Gradient is 1 a.e., so backprop-safe."""
    return (theta + math.pi) % TWO_PI - math.pi


def angles_from_unconstrained(h):
    """(..., 2k) direction vectors -> (..., k) angles, seam-free."""
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
        # [FIX A] lower clamp bound for log_sigma; annealed by train() from
        # log(sigma_floor_start) down to log(sigma_floor_end).
        self.ls_min = -4.0
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.dir      = nn.Linear(hidden, 2 * latent_dim)  # direction vector per angle
        self.logsigma = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, data_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = angles_from_unconstrained(self.dir(h))
        ls = self.logsigma(h).clamp(self.ls_min, 2.0)   # [FIX A] annealed floor
        return mu, ls

    def reparametrize(self, mu_angles, log_sigma):
        """Wrapped normal: theta = wrap(mu + sigma*eps)."""
        sigma = torch.exp(log_sigma)
        return wrap(mu_angles + sigma * torch.randn_like(sigma))

    def decode(self, theta):
        """Takes ANGLES; applies the chart internally."""
        return self.decoder(torus_embed(theta))

    def forward(self, x):
        mu, ls = self.encode(x)
        theta = self.reparametrize(mu, ls)
        return self.decode(theta), mu, ls, theta


# ------------------------------------------------------------------- loss ---
def kl_uniform(theta, mu_angles, log_sigma, n_terms=30):
    """KL( WrappedNormal(mu, sigma) || Uniform(S^1) ), summed over angles.
    Closed Gaussian form for small sigma, single-sample Fourier-series
    estimate of E[log q] + log(2pi) for large sigma. Note d = wrap(sigma*eps),
    so the KL is (correctly) independent of mu."""
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


# ---------------------------------------------- [FIX B] equivariance loss ---
def equivariance_loss(model, xb, mu, res, generator=None):
    """Rolling the rendered image by s pixels along axis j shifts theta_j by
    exactly 2*pi*s/res (torus_render is pixel-periodic). Enforce
        mu(roll(x)) = mu(x) + delta   (mod 2pi)
    with the smooth wrapped penalty 1 - cos(.). This fixes winding = 1 for the
    shifted angle and winding = 0 of that shift for the other angle.
    NOTE: theta1 <-> image COLUMNS (last dim), theta2 <-> image ROWS.
    """
    B = xb.shape[0]
    img = xb.view(B, res, res)
    s = int(torch.randint(1, res, (1,), generator=generator).item())
    j = int(torch.randint(0, 2, (1,), generator=generator).item())  # which angle
    dims = 2 if j == 0 else 1          # theta1 -> columns, theta2 -> rows
    xr = torch.roll(img, shifts=s, dims=dims).reshape(B, -1)
    mur, _ = model.encode(xr)
    delta = TWO_PI * s / res
    target = mu.clone()
    target[:, j] = target[:, j] + delta
    return (1.0 - torch.cos(mur - target)).sum(dim=1).mean()


# ------------------------------------------------------------ diagnostics ---
@torch.no_grad()
def direction_norm(model, x):
    """||v(x)|| per angle, shape (B, latent_dim). Seams/tears in the encoder
    show up as dips toward 0 (atan2 is only 'discontinuous' through the
    origin of the direction vector). After a healthy fit this should be
    bounded well away from 0 everywhere on the data."""
    h = model.encoder(x)
    v = model.dir(h)
    v = v.view(*v.shape[:-1], -1, 2)
    return v.norm(dim=-1)


# ---------------------------------------------------------------- training ---
def train(
    epochs=400,
    lr=1e-3,
    beta=1.0,
    batch_size=256,
    # [FIX A] sigma-floor annealing replaces beta warmup:
    sigma_floor_start=1.0,     # ~57 deg of posterior noise at epoch 0
    sigma_floor_end=0.018,     # = exp(-4), the original lower clamp
    sigma_anneal_frac=0.5,     # floor reaches its final value at this fraction of training
    # [FIX B] optional winding guarantee (uses known pixel-shift group action):
    equiv_weight=0.0,          # try e.g. 5.0 to pin winding numbers exactly
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

    res = int(round(math.sqrt(data.shape[1])))  # for the roll-equivariance loss

    model = model_cls(data_dim=data.shape[1], **model_kwargs).to(device)
    optimize = torch.optim.Adam(model.parameters(), lr=lr)

    n = data.shape[0]
    batch_size = min(batch_size, n)
    history = []

    ls_start = math.log(sigma_floor_start)
    ls_end = math.log(sigma_floor_end)
    anneal_epochs = max(1, int(sigma_anneal_frac * epochs))

    for ep in range(epochs):
        model.train()

        # [FIX A] log-linear decay of the sigma floor; beta constant from epoch 0.
        t = min(1.0, ep / anneal_epochs)
        model.ls_min = ls_start + t * (ls_end - ls_start)

        perm = torch.randperm(n, device=device)
        tot_loss = tot_recon = tot_kl = 0.0

        for i in range(0, n, batch_size):
            xb = data[perm[i:i + batch_size]]

            x_recon, mu, log_sigma, theta = model(xb)
            loss, r, k = elbo(x_recon, xb, mu, log_sigma, theta, beta)

            if equiv_weight > 0.0 and res * res == xb.shape[1]:
                loss = loss + equiv_weight * equivariance_loss(model, xb, mu, res)

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
                        "kl": ep_kl, "beta": beta,
                        "sigma_floor": math.exp(model.ls_min)})

        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d}  loss={ep_loss:.4f}  recon={ep_recon:.4f}  "
                  f"kl={ep_kl:.4f}  sigma_floor={math.exp(model.ls_min):.3f}")

    return model, data, angles, history


if __name__ == "__main__":
    # Unsupervised fix only:
    model, data, angles, hist = train(epochs=400)
    # Or, with the winding-number guarantee for comparison:
    # model, data, angles, hist = train(epochs=400, equiv_weight=5.0)

    from projects.visualization.vaediagnostic import diagnose
    fig = diagnose(model, data, angles, title="chart model (fixed schedule)",
                   save="torus_latent_fixed.png")

    # Seam check: should be bounded well away from 0 everywhere.
    nv = direction_norm(model, data[:2000])
    print("direction-vector norm: min per angle =", nv.min(0).values.tolist())