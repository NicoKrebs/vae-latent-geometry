"""Posterior/prior distributions for topologically constrained latent spaces.

A Euclidean VAE uses an isotropic Gaussian posterior and a standard-normal
prior. Neither exists on a torus or a sphere -- there is no additive noise
on a circle, and no flat, unbounded Gaussian on a curved, compact manifold.
This module collects the two constructions used instead:

    circle:  a WRAPPED normal. Sample additive Gaussian noise in the
             universal cover R, then project down to the circle via
             theta -> theta mod 2*pi. Used per-angle for the torus (T^2 = S^1
             x S^1), one independent wrapped normal per angle.

    sphere:  a PROJECTED (tangent-normal) construction. Sample an isotropic
             Gaussian in the tangent plane at the mean direction mu, then map
             it onto S^2 with the Riemannian exponential map. This is the
             general recipe for "Gaussian on a manifold via a chart at the
             mean" and is what makes the reparametrization trick well
             defined without ever leaving the manifold.

Both priors are simply the uniform distribution on the manifold (Uniform(S^1)
per angle for the torus, Uniform(S^2) for the sphere), so every KL term below
is KL(q(theta|x) || Uniform) = E_q[log q(theta|x)] + log(volume).

Consolidated from `architecture/new_tor_vae_charts.py` (circle) and
`architecture/new_sph_vae_chart.py` (sphere); used by
`models/torus_vae.py` and `models/sphere_vae.py` respectively. The
alternative Fourier-series form of the wrapped-normal KL used by the earlier
prototypes (`legacy/torvae_charts.py`) is mathematically equivalent but not
reproduced here -- see the note in that file.
"""

import math

import torch

TWO_PI = 2.0 * math.pi
FOUR_PI = 4.0 * math.pi


# ============================================================== circle =====

def wrap_angle(a):
    """Fold an angle (or tensor of angles) into [-pi, pi)."""
    return torch.remainder(a + math.pi, TWO_PI) - math.pi


def circle_embed(theta):
    """(..., k) angles -> (..., 2k) = (cos, sin) per angle. This is the chart
    a decoder should actually consume: sin/cos is smooth and seam-free, while
    an angle by itself has a jump discontinuity at the wrap point that would
    force the decoder to be discontinuous there too."""
    return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)


def wrapped_normal_log_prob(x, mu, log_sigma, n_wraps=5):
    """log density of a wrapped normal on the circle, evaluated by directly
    summing the images of the Gaussian under x -> x + 2*pi*k (the standard
    definition of a wrapped distribution), truncated to |k| <= n_wraps.

    x, mu, log_sigma broadcast against each other; the sum is taken over the
    last axis internally, so this returns one log-density value per *angle*
    (call `.sum(-1)` yourself to combine several independent angles, e.g. the
    two angles of a torus -- see `models/torus_vae.py`).

    n_wraps=5 truncation is accurate to machine precision for any sigma where
    the posterior is remotely informative (a term at k=6 needs the sampled
    noise to be >= 12*pi/sigma standard deviations out); it only becomes
    inaccurate once sigma is close to the prior's full range, at which point
    the KL term is dominated by the prior anyway.
    """
    k = torch.arange(-n_wraps, n_wraps + 1, device=x.device, dtype=x.dtype)
    d = (x - mu).unsqueeze(-1) + TWO_PI * k
    sigma = torch.exp(log_sigma).unsqueeze(-1)
    log_terms = -0.5 * (d / sigma) ** 2 - log_sigma.unsqueeze(-1) \
                - 0.5 * math.log(TWO_PI)
    return torch.logsumexp(log_terms, dim=-1)


def kl_wrapped_to_uniform(theta, mu, log_sigma, n_wraps=5):
    """KL( WrappedNormal(mu, sigma) || Uniform(S^1) ) per angle, summed over
    the trailing angle axis and averaged over the batch.

    Uniform(S^1) has density 1/(2*pi), so
        KL = E_q[log q(theta)] - log(1/(2*pi)) = E_q[log q(theta)] + log(2*pi)
    with the expectation approximated by the single reparametrized sample
    `theta` already drawn from q (standard single-sample ELBO estimator).
    """
    log_q = wrapped_normal_log_prob(theta, mu, log_sigma, n_wraps).sum(-1)
    return (log_q + math.log(TWO_PI)).mean()


# ============================================================== sphere =====

def sphere_embed(u):
    """(..., 2) colatitude/azimuth angles -> (..., 3) unit vectors on S^2."""
    theta, phi = u[..., 0], u[..., 1]
    st = torch.sin(theta)
    return torch.stack([st * torch.cos(phi), st * torch.sin(phi), torch.cos(theta)], dim=-1)


def angles_from_vec(z):
    """(..., 3) unit vectors -> (..., 2) (theta, phi), inverse of sphere_embed."""
    theta = torch.acos(z[..., 2].clamp(-1.0, 1.0))
    phi = torch.atan2(z[..., 1], z[..., 0])
    return torch.stack([theta, phi], dim=-1)


def exp_map(mu, v):
    """Riemannian exponential map of S^2 at mu, applied to tangent vector v.

    Walks the geodesic through mu in direction v/|v| for arclength |v|:
    exp_mu(v) = cos(|v|) mu + sin(|v|) v/|v|. Requires v to already be
    tangent to the sphere at mu (v . mu = 0); `sample_posterior` in
    models/sphere_vae.py enforces this before calling here.
    """
    r = v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.cos(r) * mu + torch.sin(r) * (v / r)


def tangent_normal_log_prob(v, log_sigma):
    """log density, ON THE SPHERE, of the pushforward of an isotropic 2d
    Gaussian in the tangent plane through the exponential map.

    Derivation: write v in tangent-plane polar coordinates (r, phi). The flat
    tangent-plane area element is r dr dphi, but the sphere's geodesic-polar
    area element at radius r is sin(r) dr dphi (the circumference of a
    geodesic circle of radius r on the unit sphere is 2*pi*sin(r), not
    2*pi*r). Conservation of probability under the exp map,
        p_tangent(v) * r dr dphi = q(exp_mu(v)) * sin(r) dr dphi,
    gives q(exp_mu(v)) = p_tangent(v) * r / sin(r). This function returns
    log q, i.e. log p_tangent(v) + log(r) - log(sin(r)); the first term is
    the ordinary isotropic-Gaussian log-density on the tangent plane's own
    (r, phi) coordinates (equivalently: 1/(2*pi*sigma^2) exp(-r^2/2 sigma^2)).

    This is the standard "projected normal on a sphere via the exponential
    map" construction, e.g. Falorsi et al. 2018 (cited in the thesis).
    """
    r = v.norm(dim=-1).clamp_min(1e-8)
    sigma = torch.exp(log_sigma.squeeze(-1))
    log_gauss = -math.log(TWO_PI) - 2.0 * torch.log(sigma) - 0.5 * (r / sigma) ** 2
    return log_gauss + torch.log(r) - torch.log(torch.sin(r).clamp_min(1e-8))


def kl_tangent_normal_to_uniform(v, log_sigma):
    """KL( ProjectedNormal(mu, sigma) || Uniform(S^2) ), averaged over the
    batch. Uniform(S^2) has density 1/(4*pi) (surface area of the unit
    sphere), so KL = E_q[log q(z)] + log(4*pi)."""
    log_q = tangent_normal_log_prob(v, log_sigma)
    return (log_q + math.log(FOUR_PI)).mean()
