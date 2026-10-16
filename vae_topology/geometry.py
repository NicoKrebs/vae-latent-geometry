"""Fisher information metric, curvature, and the Chern-Gauss-Bonnet checks.

This is the numerical core of the thesis: everything needed to go from a
trained decoder p(x|theta) to the Fisher information metric G(theta) on its
latent space, to the Gaussian curvature K of (latent space, G), to a verified
integral check against a topological invariant (chi via Gauss-Bonnet on a
closed surface, or the 2*pi*chi(D)=2*pi disk form with a boundary term).

Everything below the "generic metric-function core" section is written
against a callable `metric_fn : u (2,) -> g (2, 2)` rather than against a
decoder directly, so the SAME curvature/Gauss-Bonnet code works for:
  - the pullback metric J^T J of any smooth embedding/decoder
    (`pullback_metric_fn`),
  - the Fisher information metric of a Gaussian decoder
    (`fisher_metric_pointwise`),
  - an analytic test metric with a known closed-form answer
    (`conformal_metric_fn`, used by `gauss_bonnet_selftest`).
There is exactly one implementation of Christoffel symbols, one of the
Riemann tensor / Gaussian curvature, and one of the disk-based Gauss-Bonnet
integral -- consolidated from what were previously three independent copies
in `structure/runstore.py`, `visualization/curvature_plot.py`, and
`visualization/vaediagnostic.py` (the last of which had already started this
consolidation; this module completes it and merges in the other two).

Smoothness requirement, worth stating once: computing K needs THREE
derivatives of the decoder (metric = 1 derivative, Christoffel = 2,
Riemann/K = 3). Piecewise-linear activations (ReLU, LeakyReLU) give K == 0
identically -- a meaningless "verification" -- because their second
derivative is zero a.e. Every model in `models/` uses Tanh/SiLU throughout
for exactly this reason.

Convention check (matches the thesis): the Riemann tensor is
    R^i_{jkl} = d_k Gamma^i_{jl} - d_l Gamma^i_{jk}
              + Gamma^i_{ka} Gamma^a_{lj} - Gamma^i_{la} Gamma^a_{kj}
(main.tex, Sec. "Geometry of Information"), and in 2d the Gaussian curvature
is K = R_{1212} / det(g) (main.tex, "Theorema Egregium to Global
Gauss-Bonnet"). Every einsum below was checked index-by-index against this
convention.
"""

import math

import numpy as np
import torch
from torch.func import jacfwd, jacrev, vmap

TWO_PI = 2.0 * math.pi
FOUR_PI = 4.0 * math.pi


# ============================================== generic metric-function core

def christoffel_from_metric(metric_fn, u):
    """Gamma[k, i, j] = Gamma^k_ij of the Levi-Civita connection of g=metric_fn.

        Gamma^k_ij = (1/2) g^kl (d_i g_lj + d_j g_li - d_l g_ij)
    """
    g = metric_fn(u)
    g_inv = torch.linalg.inv(g)
    dg = jacfwd(metric_fn)(u)                # dg[i, j, k] = d_k g_ij
    G  = 0.5 * torch.einsum('kl,jli->kij', g_inv, dg)
    G += 0.5 * torch.einsum('kl,ilj->kij', g_inv, dg)
    G -= 0.5 * torch.einsum('kl,ijl->kij', g_inv, dg)
    return G


def gaussian_curvature_from_metric(metric_fn, u):
    """K = R_{1212} / det(g). In 2d this single component is the whole
    Riemann tensor. Needs the SECOND derivative of g, hence the THIRD
    derivative of whatever produced g (a decoder, an embedding, ...) -- see
    the module docstring on activation smoothness.
    """
    G  = christoffel_from_metric(metric_fn, u)
    dG = jacfwd(lambda uu: christoffel_from_metric(metric_fn, uu))(u)
    R  = (torch.einsum('rnsm->rsmn', dG) - torch.einsum('rmsn->rsmn', dG)
          + torch.einsum('rml,lns->rsmn', G, G) - torch.einsum('rnl,lms->rsmn', G, G))
    g  = metric_fn(u)
    Rl = torch.einsum('pl,lsmn->psmn', g, R)
    return Rl[0, 1, 0, 1] / torch.linalg.det(g)


def volume_element_from_metric(metric_fn, u):
    """sqrt(det g), the Riemannian area element at u."""
    return torch.sqrt(torch.linalg.det(metric_fn(u)).clamp_min(1e-300))


def pullback_metric_fn(embedding):
    """metric_fn for the pullback metric J^T J of a smooth map `embedding`
    (a decoder, or an analytic chart like the torus/sphere embeddings)."""
    def metric_fn(u):
        J = jacfwd(embedding)(u)                 # (D, 2)
        return J.T @ J                           # (2, 2)
    return metric_fn


def magnification(embedding, u):
    """sqrt(det g) of the pullback metric -- how much `embedding` locally
    stretches area. Blows up where the map tears (rank-deficient Jacobian)."""
    return volume_element_from_metric(pullback_metric_fn(embedding), u)


# ============================================ Fisher metric of a decoder ===
#
# For a diagonal-Gaussian decoder p(x|theta) = N(mean(theta), sigma(theta)^2 I)
# the Fisher information metric decomposes as
#     G_ij = sum_d (d_i mean_d)(d_j mean_d) / sigma_d^2
#          + 2 sum_d (d_i s_d)(d_j s_d),      s = log sigma
# (derivation: 0.5 tr(Sigma^-1 d_i Sigma Sigma^-1 d_j Sigma) with diagonal
# Sigma reduces to exactly this second term). For the HOMOSCEDASTIC case used
# throughout this codebase (a single learned scalar sigma_x, independent of
# theta) the second term vanishes identically and
#     G(theta) = J(theta)^T J(theta) / sigma_x^2,   J = d mean / d theta,
# which is EXACT, not merely proportional -- consequences worth stating
# explicitly since they are easy to get backwards:
#   - Christoffel symbols are unchanged by the constant factor,
#   - sqrt(det G) = sqrt(det J^T J) / sigma_x^2,
#   - K_Fisher = sigma_x^2 * K_pullback.
# The heteroscedastic term (`log_sigma_fn`) is implemented anyway so this
# stays correct if a decoder is later given theta-dependent variance.
# =============================================================================

def fisher_metric_pointwise(decode, u, log_sigma_x, data_dim, log_sigma_fn=None):
    """Fisher metric at a single latent point u (2,). Used inside vmap for
    the grid evaluators below, and directly by `gauss_bonnet_vae`."""
    J = jacfwd(decode)(u)
    if log_sigma_fn is None:
        return (J.T @ J) * torch.exp(-2.0 * log_sigma_x)
    ls = log_sigma_fn(u)
    dls = jacfwd(log_sigma_fn)(u)
    return (J.T @ J) * torch.exp(-2.0 * ls) \
        + 2.0 * data_dim * torch.outer(dls, dls)


def decoder_jacobian(fn, z, chunk=256):
    """J[n] = d fn(z[n]) / d z[n], shape (N, D, L). Batched over many latents
    at once (unlike `fisher_metric_pointwise`, which is one point at a time
    under vmap) -- used for the full metric-on-a-grid map, which only needs
    ONE derivative and can therefore afford a much larger batch/grid than the
    curvature maps below (which need three).

    Uses torch.func.vmap + jacrev, which needs a vmap-safe module (no
    batchnorm in train mode, no in-place ops on the input).
    """
    def single(zi):
        return fn(zi.unsqueeze(0)).reshape(-1)

    jac = vmap(jacrev(single))
    return torch.cat([jac(z[i:i + chunk]) for i in range(0, z.shape[0], chunk)],
                     dim=0)


def fisher_metric_batched(decode_fn, z, log_sigma_fn=None, sigma=1.0, chunk=256):
    """Batched Fisher-Rao metric of a diagonal-Gaussian decoder, for many
    latents `z` (N, L) at once. Returns G of shape (N, L, L)."""
    z = z.detach()
    J = decoder_jacobian(decode_fn, z, chunk)                  # (N, D, L)

    if log_sigma_fn is None:
        G = torch.einsum("ndi,ndj->nij", J, J) / (sigma ** 2)
    else:
        s = log_sigma_fn(z)
        if s.dim() == 1:                                       # (D,) -> (N, D)
            s = s.unsqueeze(0).expand(z.shape[0], -1)
        inv_var = torch.exp(-2.0 * s)                          # (N, D)
        G = torch.einsum("ndi,nd,ndj->nij", J, inv_var, J)
        S = decoder_jacobian(log_sigma_fn, z, chunk)           # (N, D, L)
        G = G + 2.0 * torch.einsum("ndi,ndj->nij", S, S)

    return 0.5 * (G + G.transpose(-1, -2))   # symmetrise away roundoff


def fisher_metric_grid(decode_fn, grid=64, log_sigma_fn=None, sigma=1.0,
                       chunk=256, device="cpu", lo=-np.pi, hi=np.pi,
                       periodic=True, dtype=torch.float32):
    """Full 2x2 Fisher metric on a (grid, grid) lattice over [lo, hi)^2 --
    for visualising/cross-checking the metric itself (anisotropy, condition
    number, ...), NOT curvature. See `fisher_curvature_map_torus` for that.

    Returns (G, t1, t2): G shape (grid, grid, 2, 2), t1/t2 numpy (grid,).
    Index convention matches meshgrid(indexing='ij'): G[i, j] is the metric
    at (t1[i], t2[j]).

    periodic=True omits the duplicated endpoint (correct for angles).
    """
    if periodic:
        t1 = torch.linspace(lo, hi, grid + 1, device=device, dtype=dtype)[:grid]
    else:
        t1 = torch.linspace(lo, hi, grid, device=device, dtype=dtype)
    t2 = t1.clone()

    Z = torch.stack(torch.meshgrid(t1, t2, indexing="ij"), dim=-1).reshape(-1, 2)
    G = fisher_metric_batched(decode_fn, Z, log_sigma_fn, sigma, chunk)
    G = G.reshape(grid, grid, 2, 2).detach().cpu()
    return G, t1.cpu().numpy(), t2.cpu().numpy()


def metric_invariants(G):
    """Coordinate-scalar quantities from G of shape (..., 2, 2).

    Returns a dict of numpy arrays with the leading shape of G:
      det, sqrt_det (the area element), trace,
      lambda_max / lambda_min (principal stretches squared),
      anisotropy = lambda_max / lambda_min  (1 = locally conformal),
      condition  = sqrt(anisotropy)         (ratio of principal stretches),
      g11, g12, g22 (raw components, coordinate dependent).
    """
    G = G.detach().cpu().numpy() if isinstance(G, torch.Tensor) else np.asarray(G)
    g11, g12 = G[..., 0, 0], G[..., 0, 1]
    g21, g22 = G[..., 1, 0], G[..., 1, 1]

    det = g11 * g22 - g12 * g21
    tr = g11 + g22
    disc = np.clip(tr ** 2 - 4.0 * det, 0.0, None)
    root = np.sqrt(disc)
    lam_max = 0.5 * (tr + root)
    lam_min = 0.5 * (tr - root)
    aniso = lam_max / np.clip(lam_min, 1e-30, None)

    return {
        "det": det,
        "sqrt_det": np.sqrt(np.clip(det, 0.0, None)),
        "trace": tr,
        "lambda_max": lam_max,
        "lambda_min": lam_min,
        "anisotropy": aniso,
        "condition": np.sqrt(aniso),
        "g11": g11, "g12": g12, "g22": g22,
    }


# ===================================== curvature maps over a whole latent ==

@torch.no_grad()
def fisher_curvature_map_torus(model, grid=64, chunk=32, device="cpu",
                               log_sigma_fn=None):
    """Evaluate K and sqrt(det G) of the Fisher metric on a (grid, grid)
    midpoint lattice over [-pi, pi)^2, i.e. over the whole latent torus.
    Returns (K, V, t1, t2)."""
    model = model.double().to(device).eval()
    decode = model.decode
    ls_x = model.log_sigma_x.detach().double()
    D = model.data_dim

    def metric_fn(u):
        return fisher_metric_pointwise(decode, u, ls_x, D, log_sigma_fn)

    t = -math.pi + (torch.arange(grid, dtype=torch.float64) + 0.5) * (TWO_PI / grid)
    T1, T2 = torch.meshgrid(t, t, indexing="xy")
    pts = torch.stack([T1.reshape(-1), T2.reshape(-1)], dim=1).to(device)

    Kf = vmap(lambda u: gaussian_curvature_from_metric(metric_fn, u))
    Vf = vmap(lambda u: volume_element_from_metric(metric_fn, u))

    # CHUNKED: vmapping a jacfwd-of-jacfwd (needed for K) over every grid
    # point at once allocates tens of GB.
    Ks, Vs = [], []
    for i in range(0, pts.shape[0], chunk):
        block = pts[i:i + chunk]
        Ks.append(Kf(block))
        Vs.append(Vf(block))

    K = torch.cat(Ks).reshape(grid, grid).cpu().numpy()
    V = torch.cat(Vs).reshape(grid, grid).cpu().numpy()
    return K, V, t.numpy(), t.numpy()


@torch.no_grad()
def fisher_curvature_map_sphere(model, n_theta=64, n_phi=128, eps=0.08,
                                chunk=32, device="cpu", log_sigma_fn=None):
    """Evaluate K and sqrt(det G) of the Fisher metric on a midpoint grid over
        theta in (eps, pi - eps),  phi in [-pi, pi).
    `eps` avoids the coordinate singularity of (theta, phi) at the poles
    (sqrt(det G) picks up a genuine sin(theta) factor there that has nothing
    to do with the model -- see `plot_abs_curvature_sphere`, which divides it
    back out for the plotted "area magnification" panel).
    Returns (K, V, theta, phi), K and V of shape (n_theta, n_phi).
    """
    model = model.double().to(device).eval()
    decode = model.decode
    ls_x = model.log_sigma_x.detach().double()
    D = model.data_dim

    def metric_fn(u):
        return fisher_metric_pointwise(decode, u, ls_x, D, log_sigma_fn)

    span = math.pi - 2.0 * eps
    dth = span / n_theta
    dph = TWO_PI / n_phi
    theta = eps + (torch.arange(n_theta, dtype=torch.float64) + 0.5) * dth
    phi = -math.pi + (torch.arange(n_phi, dtype=torch.float64) + 0.5) * dph

    TH, PH = torch.meshgrid(theta, phi, indexing="ij")
    pts = torch.stack([TH.reshape(-1), PH.reshape(-1)], dim=1).to(device)

    Kf = vmap(lambda u: gaussian_curvature_from_metric(metric_fn, u))
    Vf = vmap(lambda u: volume_element_from_metric(metric_fn, u))

    Ks, Vs = [], []
    for i in range(0, pts.shape[0], chunk):
        block = pts[i:i + chunk]
        Ks.append(Kf(block))
        Vs.append(Vf(block))

    K = torch.cat(Ks).reshape(n_theta, n_phi).cpu().numpy()
    V = torch.cat(Vs).reshape(n_theta, n_phi).cpu().numpy()
    return K, V, theta.numpy(), phi.numpy()


@torch.no_grad()
def curvature_map_euclidean(model, mu_codes, grid=40, pad=0.15, device="cpu", chunk=64,
                            lo=None, hi=None):
    """Evaluate |K| and sqrt(det g) of the PULLBACK metric (not Fisher -- no
    learned decoder noise to divide out) on a grid covering the region where
    the data actually lives (set from the encoded mu's; curvature far outside
    the data is meaningless extrapolation). For a standard Euclidean-latent
    VAE (`models/euclidean_vae.py`).
    Returns (K, sqrtg, extent).

    `mu_codes` is moved to `device` internally -- if it was encoded on a
    different device than `device` (e.g. a model trained and encoded on
    "cuda" while this defaults to "cpu" for the curvature pass), the
    lo/hi/linspace chain below would otherwise mix devices and raise a
    device-mismatch error. Callers used to work around this by moving both
    model and data to CPU themselves before calling in; not required now.

    Pass `lo`/`hi` (each a length-2 sequence) to use an EXPLICIT grid extent
    instead of deriving one from `mu_codes` -- e.g. to keep the same window
    across many checkpoints of the same run for an evolution video
    (`experiments.render_evolution`), where the flat latent's data-dependent
    range would otherwise shift/rescale the extent every frame.
    """
    model = model.double().to(device)
    dec = model.decode
    mu_codes = mu_codes.to(device)

    if lo is None or hi is None:
        lo = mu_codes.min(0).values.double()
        hi = mu_codes.max(0).values.double()
        span = (hi - lo)
        lo = lo - pad * span
        hi = hi + pad * span
    else:
        lo = torch.as_tensor(lo, dtype=torch.float64, device=device)
        hi = torch.as_tensor(hi, dtype=torch.float64, device=device)

    xs = torch.linspace(lo[0].item(), hi[0].item(), grid, dtype=torch.float64, device=device)
    ys = torch.linspace(lo[1].item(), hi[1].item(), grid, dtype=torch.float64, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    metric_fn = pullback_metric_fn(dec)
    Kf = vmap(lambda u: gaussian_curvature_from_metric(metric_fn, u))
    Mf = vmap(lambda u: volume_element_from_metric(metric_fn, u))
    # CHUNKED: see fisher_curvature_map_torus.
    Ks, Ms = [], []
    for i in range(0, pts.shape[0], chunk):
        block = pts[i:i + chunk]
        Ks.append(Kf(block))
        Ms.append(Mf(block))
    K = torch.cat(Ks).reshape(grid, grid)
    M = torch.cat(Ms).reshape(grid, grid)
    extent = [lo[0].item(), hi[0].item(), lo[1].item(), hi[1].item()]
    return K.cpu().numpy(), M.cpu().numpy(), extent


# ==================================================== Gauss-Bonnet on a disk
#
#   int_D K dA  +  oint_{dD} k_g ds  =  2 pi chi(D)  =  2 pi
#
# This is the version of Chern-Gauss-Bonnet used to sanity-check a metric
# ANYWHERE it is defined, not just on a closed manifold: pick a topological
# disk D (a Euclidean circle in latent coordinates -- smooth, no corners, so
# there are no exterior-angle terms), and check that the area integral of K
# plus the boundary integral of the geodesic curvature k_g equals 2*pi.
#
# Preconditions that are on the caller, not on the theorem:
#   * g smooth and positive definite on the CLOSED disk, boundary included. A
#     pullback metric degenerates where the decoder loses rank -- exactly the
#     extrapolation region a generous radius reaches into. Check
#     `bad_fraction` and the condition number (`metric_invariants`) before
#     believing anything.
#   * the region is a topological disk: no holes, no punctures where det g=0.
#   * the boundary is traversed counterclockwise (region on the left).
#     Reversing it flips the sign of k_g.
# =============================================================================

_TAU = TWO_PI


def _levi_civita(dtype, device):
    return torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=dtype, device=device)


def geodesic_curvature(metric_fn, u, udot, uddot):
    """k_g of a curve through u with velocity udot and acceleration uddot,
    w.r.t. the INWARD unit normal (the convention that makes Gauss-Bonnet
    close with a counterclockwise boundary).

        v      = sqrt(g(udot, udot))
        T      = udot / v
        a^k    = uddot^k + Gamma^k_ij udot^i udot^j        (covariant accel.)
        omega_i = -sqrt(det g) eps_ij T^j,   N^i = g^ij omega_j   (inward)
        k_g    = g(a, N) / v^2

    The 1/v^2 is because the curve is not arclength-parameterised; the
    correction term of d/dt(udot/v) is parallel to T and dies against N.

    Sanity check: g = I, circle of radius R  ->  k_g = 1/R, oint k_g ds = 2 pi.
    If you get -1/R your normal orientation is flipped.

    Returns (k_g, v).
    """
    g = metric_fn(u)
    Gam = christoffel_from_metric(metric_fn, u)

    v2 = torch.einsum('i,ij,j->', udot, g, udot)
    v = torch.sqrt(v2.clamp_min(1e-300))
    T = udot / v

    a = uddot + torch.einsum('kij,i,j->k', Gam, udot, udot)

    detg = torch.linalg.det(g).clamp_min(1e-300)
    eps = _levi_civita(g.dtype, g.device)
    omega = -torch.sqrt(detg) * (eps @ T)      # lower-index inward normal
    N = torch.linalg.solve(g, omega)           # raise the index

    kg = torch.einsum('i,ij,j->', a, g, N) / v2
    return kg, v


@torch.no_grad()
def boundary_term_circle(metric_fn, center, radius, n_bdry=512, chunk=64,
                         dtype=torch.float64, device="cpu"):
    """oint k_g ds over gamma(t) = c + R (cos t, sin t), counterclockwise.

    Midpoint rule on a periodic integrand is spectrally accurate, so this
    term is essentially exact for a few hundred nodes -- all of the
    quadrature error in the Gauss-Bonnet residual lives in the area term.

    Returns (integral, k_g array, speed array, boundary points).
    """
    c = torch.as_tensor(center, dtype=dtype, device=device).reshape(2)
    R = float(radius)
    dt = _TAU / n_bdry
    t = (torch.arange(n_bdry, dtype=dtype, device=device) + 0.5) * dt
    ct, st = torch.cos(t), torch.sin(t)

    gamma = c + R * torch.stack([ct, st], dim=1)
    gdot = R * torch.stack([-st, ct], dim=1)
    gddot = -R * torch.stack([ct, st], dim=1)

    f = vmap(lambda u, ud, udd: geodesic_curvature(metric_fn, u, ud, udd))
    kgs, vs = [], []
    for i in range(0, n_bdry, chunk):
        k_, v_ = f(gamma[i:i + chunk], gdot[i:i + chunk], gddot[i:i + chunk])
        kgs.append(k_)
        vs.append(v_)
    kg = torch.cat(kgs)
    v = torch.cat(vs)

    integral = float((kg * v).sum() * dt)
    return integral, kg.cpu().numpy(), v.cpu().numpy(), gamma.cpu().numpy()


@torch.no_grad()
def area_term_disk(metric_fn, center, radius, n_r=64, n_theta=128, chunk=64,
                   dtype=torch.float64, device="cpu"):
    """int_D K dA over a polar midpoint grid fitted to the disk.

        int K sqrt(det g) r dr dtheta

    Polar, not Cartesian: a Cartesian grid masked to r <= R chops cells at
    the boundary and the resulting O(h) error swamps everything (see
    `gauss_bonnet_from_curvature_map`). Midpoint in r also dodges the r = 0
    coordinate singularity for free.

    Returns (int K dA, Riemannian area, K grid, sqrt(det g) grid, bad fraction).
    """
    c = torch.as_tensor(center, dtype=dtype, device=device).reshape(2)
    R = float(radius)
    dr = R / n_r
    dth = _TAU / n_theta

    r = (torch.arange(n_r, dtype=dtype, device=device) + 0.5) * dr
    th = (torch.arange(n_theta, dtype=dtype, device=device) + 0.5) * dth
    RR, TT = torch.meshgrid(r, th, indexing="ij")
    pts = c + torch.stack([RR * torch.cos(TT), RR * torch.sin(TT)],
                          dim=-1).reshape(-1, 2)
    w = (RR * dr * dth).reshape(-1)          # r dr dtheta

    Kf = vmap(lambda u: gaussian_curvature_from_metric(metric_fn, u))
    Vf = vmap(lambda u: volume_element_from_metric(metric_fn, u))

    Ks, Vs = [], []
    for i in range(0, pts.shape[0], chunk):
        b = pts[i:i + chunk]
        Ks.append(Kf(b))
        Vs.append(Vf(b))
    K = torch.cat(Ks)
    V = torch.cat(Vs)

    good = torch.isfinite(K) & torch.isfinite(V)
    intK = float((K[good] * V[good] * w[good]).sum())
    area = float((V[good] * w[good]).sum())
    bad = float((~good).double().mean())

    return (intK, area,
            K.reshape(n_r, n_theta).cpu().numpy(),
            V.reshape(n_r, n_theta).cpu().numpy(),
            bad)


@torch.no_grad()
def gauss_bonnet_disk(metric_fn, center, radius, n_r=64, n_theta=128,
                      n_bdry=512, chunk=64, dtype=torch.float64,
                      device="cpu", verbose=True):
    """Verify int K dA + oint k_g ds = 2 pi on the disk of radius `radius`
    about `center`, in latent coordinates.

    metric_fn : u (2,) -> g (2, 2). Use pullback_metric_fn(model.decode), or
                lambda u: fisher_metric_pointwise(decode, u, ls_x, D, ...).

    Returns a dict with both terms, the residual, the Riemannian area, and
    the raw grids for plotting.
    """
    intK, area, Kg, Vg, badfrac = area_term_disk(
        metric_fn, center, radius, n_r, n_theta, chunk, dtype, device)
    bnd, kg, speed, bpts = boundary_term_circle(
        metric_fn, center, radius, n_bdry, chunk, dtype, device)

    total = intK + bnd
    res = total - _TAU

    if verbose:
        print(f"  centre {tuple(float(x) for x in np.asarray(center).ravel())}  radius {radius:g}")
        print(f"  int K dA      : {intK:+.6f}")
        print(f"  oint k_g ds   : {bnd:+.6f}")
        print(f"  sum           : {total:+.6f}    target 2pi = {_TAU:.6f}")
        print(f"  residual      : {res:+.3e}   ({abs(res) / _TAU:.2%} of 2pi)")
        print(f"  Riem. area    : {area:.6e}")
        print(f"  bdry speed    : min {speed.min():.3e}  max {speed.max():.3e}")
        if badfrac > 0:
            print(f"  [WARNING] non-finite K or sqrt(det g) on "
                  f"{100 * badfrac:.2f}% of the area grid -- metric degenerates "
                  f"inside the disk, Gauss-Bonnet does not apply as stated")

    return dict(area_term=intK, boundary_term=bnd, total=total, residual=res,
                area=area, K_polar=Kg, V_polar=Vg, kg=kg, speed=speed,
                boundary_points=bpts, bad_fraction=badfrac,
                center=np.asarray(center, dtype=float).ravel(),
                radius=float(radius))


def disk_from_codes(mu, pad=0.05, center="bbox"):
    """Smallest circle containing all latent codes, inflated by `pad`.

    center="bbox" uses the bounding-box midpoint (robust to density, which is
    what you want -- the centre only fixes the coordinate origin, not the
    answer); center="mean" uses the centroid.

    Keep `pad` SMALL. A generous radius does not make the test safer; it
    pushes the boundary into the region the decoder never saw, where J loses
    rank.
    """
    a = mu.detach().cpu().numpy() if isinstance(mu, torch.Tensor) else np.asarray(mu)
    m = a.reshape(-1, 2)
    if center == "mean":
        c = m.mean(0)
    else:
        c = 0.5 * (m.min(0) + m.max(0))
    R = np.linalg.norm(m - c, axis=1).max() * (1.0 + pad)
    return c, float(R)


@torch.no_grad()
def gauss_bonnet_refinement(metric_fn, center, radius, levels=(16, 32, 64, 128),
                            n_theta=128, n_bdry=512, chunk=64,
                            dtype=torch.float64, device="cpu"):
    """Convergence study in the radial resolution.

    This is the ONLY thing that distinguishes 'my code is right and the
    metric is what it is' from 'my quadrature is too coarse'. The residual
    should fall like O(dr^2) -- roughly a factor of 4 per doubling of n_r. A
    residual that plateaus means either the metric is genuinely singular
    somewhere in the disk, or K is peaked on a scale your grid never
    resolves (typical: learned metrics are near-flat inside data clusters
    and sharply curved between them).
    """
    rows = []
    print(f"  {'n_r':>6} {'int K dA':>14} {'oint k_g ds':>14} "
          f"{'residual':>12} {'ratio':>8}")
    prev = None
    for n_r in levels:
        r = gauss_bonnet_disk(metric_fn, center, radius, n_r=n_r,
                              n_theta=n_theta, n_bdry=n_bdry, chunk=chunk,
                              dtype=dtype, device=device, verbose=False)
        ratio = abs(prev / r["residual"]) if (prev is not None and r["residual"] != 0) else float("nan")
        print(f"  {n_r:>6d} {r['area_term']:>+14.6f} {r['boundary_term']:>+14.6f} "
              f"{r['residual']:>+12.3e} {ratio:>8.2f}")
        prev = r["residual"]
        rows.append(r)
    return rows


@torch.no_grad()
def gauss_bonnet_from_curvature_map(K, M, extent, center, radius,
                                    metric_fn=None, n_bdry=512, chunk=64,
                                    dtype=torch.float64, device="cpu",
                                    verbose=True):
    """Reuse an already-computed curvature-map output (e.g. from
    `curvature_map_euclidean`) for the area term.

    Cheap, since it costs no new decoder evaluations -- but the disk
    boundary cuts through Cartesian cells, and that ragged edge is an O(h)
    error that does NOT converge cleanly under refinement. Empirically it
    stalls around 1e-3 while `area_term_disk` keeps dropping by 4x per
    refinement. Use this as a consistency check on the polar result, not as
    the number you quote.

    Pass `metric_fn` to also get the boundary term and a full residual;
    otherwise only the area term is returned.
    """
    K = np.asarray(K)
    M = np.asarray(M)
    ny, nx = K.shape                      # curvature map: K[iy, ix]
    x0, x1, y0, y1 = extent
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    dx = (x1 - x0) / (nx - 1)
    dy = (y1 - y0) / (ny - 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    inside = ((X - center[0]) ** 2 + (Y - center[1]) ** 2) <= radius ** 2
    good = inside & np.isfinite(K) & np.isfinite(M)
    intK = float((K[good] * M[good]).sum() * dx * dy)

    covered = radius <= min(center[0] - x0, x1 - center[0],
                            center[1] - y0, y1 - center[1])
    out = dict(area_term=intK, n_cells=int(good.sum()),
               fully_covered=bool(covered))

    if verbose:
        print(f"  [cartesian mask, {int(good.sum())} cells]")
        print(f"  int K dA      : {intK:+.6f}")
        if not covered:
            print(f"  [WARNING] the disk is not contained in the grid extent; "
                  f"the area term is truncated")
        if inside.sum() and not np.isfinite(K[inside]).all():
            print(f"  [WARNING] non-finite K on "
                  f"{100 * (~np.isfinite(K[inside])).mean():.2f}% of cells in the disk")

    if metric_fn is not None:
        bnd, kg, speed, _ = boundary_term_circle(metric_fn, center, radius,
                                                 n_bdry, chunk, dtype, device)
        out.update(boundary_term=bnd, total=intK + bnd,
                   residual=intK + bnd - _TAU, kg=kg, speed=speed)
        if verbose:
            print(f"  oint k_g ds   : {bnd:+.6f}")
            print(f"  sum           : {out['total']:+.6f}    "
                  f"target 2pi = {_TAU:.6f}")
            print(f"  residual      : {out['residual']:+.3e}   "
                  f"({abs(out['residual']) / _TAU:.2%} of 2pi)")
    return out


@torch.no_grad()
def gauss_bonnet_vae(model, mu_codes=None, data=None, center=None, radius=None,
                     pad=0.05, n_r=64, n_theta=128, n_bdry=512, chunk=64,
                     device="cpu", metric="pullback", log_sigma_fn=None,
                     refine=True, fname=None, verbose=True):
    """Convenience entry point: run the Gauss-Bonnet disk check on a VAE's
    latent space, over the disk that contains the encoded codes.

    metric : "pullback" for J^T J, "fisher" for the decoder's Fisher metric
             (requires model.log_sigma_x).

    Pass `fname` to also render the three-panel figure via
    `diagnostics.curvature_plots.plot_gauss_bonnet`.
    """
    model = model.double().to(device).eval()
    decode = model.decode

    if mu_codes is None:
        if data is None:
            raise ValueError("pass either mu_codes or data")
        mu_codes, _ = model.encode(data.double().to(device))

    if metric == "fisher":
        ls_x = model.log_sigma_x.detach().double()
        D = model.data_dim
        metric_fn = lambda u: fisher_metric_pointwise(decode, u, ls_x, D, log_sigma_fn)
    else:
        metric_fn = pullback_metric_fn(decode)

    if center is None or radius is None:
        c, R = disk_from_codes(mu_codes, pad=pad)
        center = c if center is None else center
        radius = R if radius is None else radius

    if verbose:
        print(f"Gauss-Bonnet on the latent disk  [{metric} metric]")
    res = gauss_bonnet_disk(metric_fn, center, radius, n_r=n_r,
                            n_theta=n_theta, n_bdry=n_bdry, chunk=chunk,
                            device=device, verbose=verbose)

    if refine:
        print("  refinement in n_r (residual should fall ~4x per doubling):")
        gauss_bonnet_refinement(metric_fn, center, radius,
                                levels=(n_r // 4, n_r // 2, n_r),
                                n_theta=n_theta, n_bdry=n_bdry, chunk=chunk,
                                device=device)

    if fname:
        from vae_topology.diagnostics.curvature_plots import plot_gauss_bonnet
        plot_gauss_bonnet(res, mu_codes=mu_codes, fname=fname)
    return res


def conformal_metric_fn(lam):
    """g = exp(2 lam) * delta, for analytic test metrics. K = -exp(-2 lam) Lap lam."""
    def m(u):
        return torch.exp(2.0 * lam(u)) * torch.eye(2, dtype=u.dtype, device=u.device)
    return m


def gauss_bonnet_selftest(verbose=True):
    """Run the Gauss-Bonnet-on-a-disk pipeline on metrics whose answer is
    known in closed form.

    Run this BEFORE pointing it at a decoder. If these close to 2*pi, the
    code is right and any residual on a learned metric is a property of that
    metric (or of the resolution), not a bug.

        flat            K = 0                 oint = 2 pi
        sphere          K = +1, rho = 2 atan R  int = 2pi(1 - cos rho)
        Poincare disk   K = -1, rho = 2 atanh R int = -2pi(cosh rho - 1)
    """
    out = {}

    if verbose:
        print("FLAT  (K = 0)")
    flat = conformal_metric_fn(lambda u: torch.zeros((), dtype=u.dtype))
    out["flat"] = gauss_bonnet_disk(flat, (0.0, 0.0), 1.7, n_r=24, n_theta=32,
                                    n_bdry=128, verbose=verbose)

    if verbose:
        print("\nROUND SPHERE, stereographic chart  (K = +1),  R = 0.6")
    sph = conformal_metric_fn(lambda u: torch.log(2.0 / (1.0 + (u * u).sum())))
    out["sphere"] = gauss_bonnet_disk(sph, (0.0, 0.0), 0.6, n_r=64, n_theta=64,
                                      n_bdry=128, verbose=verbose)
    rho = 2 * math.atan(0.6)
    if verbose:
        print(f"    exact int K dA = {2 * math.pi * (1 - math.cos(rho)):+.6f}   "
              f"exact oint = {2 * math.pi * math.cos(rho):+.6f}")

    if verbose:
        print("\nPOINCARE DISK  (K = -1),  R = 0.7")
    hyp = conformal_metric_fn(lambda u: torch.log(2.0 / (1.0 - (u * u).sum())))
    out["hyperbolic"] = gauss_bonnet_disk(hyp, (0.0, 0.0), 0.7, n_r=128,
                                          n_theta=64, n_bdry=128, verbose=verbose)
    rho = math.log(1.7 / 0.3)
    if verbose:
        print(f"    exact int K dA = {-2 * math.pi * (math.cosh(rho) - 1):+.6f}   "
              f"exact oint = {2 * math.pi * math.cosh(rho):+.6f}")

    if verbose:
        print("\nPULLBACK path, torus embedding R^2 -> R^3,  R = 0.9")

    def torus(u):
        Rr, rr = 2.0, 1.0
        t1, t2 = u[0], u[1]
        return torch.stack([(Rr + rr * torch.cos(t2)) * torch.cos(t1),
                            (Rr + rr * torch.cos(t2)) * torch.sin(t1),
                            rr * torch.sin(t2)])

    out["torus"] = gauss_bonnet_disk(pullback_metric_fn(torus), (0.0, 0.0), 0.9,
                                     n_r=64, n_theta=64, n_bdry=128,
                                     verbose=verbose)
    return out
