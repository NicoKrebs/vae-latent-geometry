# ARCHIVED -- kept for the historical record, not part of the live pipeline.
#
# Status: WORKING, not imported anywhere else in the codebase. A self-contained
# 3d torus plotting toolkit (latent scatter on the donut surface, decoder
# scalar fields painted on the torus, winding-number diagnostics via
# `winding_numbers`/`loop_indices`, posterior-coverage heatmaps). The winding-
# number and generative-loop machinery here is more thorough than anything in
# the live diagnostics and may be worth promoting into
# vae_topology/diagnostics/ if you need it again.
#
# The plotting actually wired into the live pipeline (vae_topology.diagnostics
# .latent_plots) is a separate, independently written implementation
# (`plot_torus_latent`), not derived from this file.

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

TWO_PI = 2.0 * math.pi


R_MAJOR = 2.0
R_MINOR = 0.85



def _np(a):
    """Accept torch tensors or anything array-like."""
    if hasattr(a, "detach"):
        a = a.detach().cpu()
    return np.asarray(a, dtype=np.float64)


def wrapc(x):
    """Angle difference folded into (-pi, pi]."""
    return np.remainder(x + np.pi, TWO_PI) - np.pi


def torus_embed(theta, R=R_MAJOR, r=R_MINOR, bulge=1.0):
    """
    T^2 -> R^3, the standard donut. theta: (..., 2) -> (..., 3).

    theta[...,0] runs around the major circle (the hole), theta[...,1] around
    the tube. `bulge` lifts points slightly off the surface so scatter markers
    are not z-fought by the surface mesh.
    """
    theta = _np(theta)
    t1, t2 = theta[..., 0], theta[..., 1]
    rr = r * bulge
    return np.stack([(R + rr * np.cos(t2)) * np.cos(t1),
                     (R + rr * np.cos(t2)) * np.sin(t1),
                     rr * np.sin(t2)], axis=-1)


def _surface(R=R_MAJOR, r=R_MINOR, n=120):
    u = np.linspace(0, TWO_PI, n)
    v = np.linspace(0, TWO_PI, n)
    U, V = np.meshgrid(u, v, indexing="ij")
    P = torus_embed(np.stack([U, V], -1), R, r)
    return P[..., 0], P[..., 1], P[..., 2], U, V


def _style_axes(ax, title=None):
    ax.set_box_aspect((1, 1, 0.42))
    lim = R_MAJOR + R_MINOR
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim * 0.42, lim * 0.42)
    ax.set_axis_off()
    ax.view_init(elev=38, azim=-55)
    if title:
        ax.set_title(title, fontsize=10, pad=0)


def _draw_shell(ax, alpha=0.10, color="0.55"):
    X, Y, Z, _, _ = _surface()
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha,
                    linewidth=0, antialiased=True, shade=True, zorder=0)



def plot_latent_scatter(ax, mu, color=None, cmap="twilight", size=4,
                        alpha=0.75, shell=True, cbar_label=None, title=None):
    """
    Scatter posterior means on the torus surface.

    `color` is typically a ground-truth angle, using a *cyclic* colormap so the
    seam is not a visual discontinuity. If the model recovered the generative
    factors up to identity, colour varies smoothly and monotonically once around
    the corresponding cycle; if it learned a doubling map, you will see the
    colour cycle twice.
    """
    mu = _np(mu)
    if shell:
        _draw_shell(ax)

    P = torus_embed(mu, bulge=1.03)
    kw = dict(s=size, alpha=alpha, linewidths=0, depthshade=False)
    if color is None:
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="#c0392b", **kw)
    else:
        c = _np(color).ravel()
        norm = Normalize(0, TWO_PI) if cmap == "twilight" else None
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=c, cmap=cmap, norm=norm, **kw)
        if cbar_label:
            cb = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
            cb.set_label(cbar_label, fontsize=8)
            cb.ax.tick_params(labelsize=7)
    _style_axes(ax, title)
    return sc


def plot_decoder_field(ax, field_fn, n=140, cmap="RdBu_r", title=None,
                       cbar_label=None):
    """
    Paint a scalar readout of the decoder over the whole latent torus.

    field_fn: (N,2) array of angles -> (N,) values.

    This is the visual periodicity check: a correct toroidal decoder produces a
    field with no seam anywhere, including where the surface closes on itself.
    A seam that is visible here is a real discontinuity, not a rendering artefact.
    """
    X, Y, Z, U, V = _surface(n=n)
    vals = np.asarray(field_fn(np.stack([U.ravel(), V.ravel()], -1))).reshape(U.shape)

    vmax = np.abs(vals).max()
    norm = Normalize(-vmax, vmax) if vals.min() < 0 else Normalize(vals.min(), vals.max())
    ax.plot_surface(X, Y, Z, facecolors=plt.get_cmap(cmap)(norm(vals)), linewidth=0,
                    antialiased=True, shade=False, rstride=1, cstride=1)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(vals)
    cb = plt.colorbar(mappable, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label(cbar_label or "decoder output", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _style_axes(ax, title)



def loop_indices(angles, axis=0, level=0.0, tol=0.12):
    """
    Indices of dataset points lying near a loop in *ground-truth* angle space:
    the other coordinate is held near `level`, and the result is sorted along
    `axis`. Approximates a generative loop using only the data you already have.
    """
    angles = _np(angles)
    other = 1 - axis
    keep = np.nonzero(np.abs(wrapc(angles[:, other] - level)) < tol)[0]
    return keep[np.argsort(angles[keep, axis])]


def winding_numbers(mu_loop):
    """
    Net winding of an encoded loop in each latent dimension.

    Sums the wrapped increments around a closed traversal; for a continuous map
    this is exactly 2*pi times an integer. Returns (raw, rounded). A large gap
    between the two means the encoded path is not continuous -- either the loop
    is undersampled or the encoder is not a well-behaved map.
    """
    mu = _np(mu_loop)
    steps = wrapc(np.diff(np.vstack([mu, mu[:1]]), axis=0))
    raw = steps.sum(0) / TWO_PI
    return raw, np.rint(raw).astype(int)


def plot_loops(ax, loops, cmap="tab10", shell=True, title=None, lw=1.6):
    """`loops` is a list of (label, mu_array) encoded generative loops."""
    if shell:
        _draw_shell(ax)
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, 10))
    for i, (label, mu) in enumerate(loops):
        P = torus_embed(mu, bulge=1.04)
        P = np.vstack([P, P[:1]])                       # close the path
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color=colors[i % 10],
                lw=lw, label=label, alpha=0.9)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    _style_axes(ax, title)

def plot_coverage(ax, mu, bins=None, cmap="magma", title=None):

    mu = _np(mu)
    if bins is None:
        bins = int(np.clip(np.sqrt(len(mu) / 20.0), 8, 64))
    H, _, _ = np.histogram2d(mu[:, 0], mu[:, 1], bins=bins,
                             range=[[0, TWO_PI], [0, TWO_PI]])
    H = H / H.mean()

    X, Y, Z, U, V = _surface(n=bins + 1)
    idx_u = np.clip((U[:-1, :-1] / TWO_PI * bins).astype(int), 0, bins - 1)
    idx_v = np.clip((V[:-1, :-1] / TWO_PI * bins).astype(int), 0, bins - 1)
    vals = H[idx_u, idx_v]

    norm = Normalize(0, max(2.0, vals.max()))
    ax.plot_surface(X, Y, Z, facecolors=plt.get_cmap(cmap)(norm(vals)),
                    linewidth=0, antialiased=True, shade=False,
                    rstride=1, cstride=1)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(vals)
    cb = plt.colorbar(mappable, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("density / uniform", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _style_axes(ax, title)



def decoder_field_fn(model, channel=0):
    """Wrap a ToroidalVAE decoder as a scalar field callable for plot_decoder_field."""
    import torch

    dev = next(model.parameters()).device

    def f(theta_np):
        z = torch.as_tensor(np.asarray(theta_np), dtype=torch.float32, device=dev)
        with torch.no_grad():
            out = model.decoder(z)
        return out[:, channel].cpu().numpy()

    return f


def plot_latent_summary(model, data, angles=None, save=None, channel=0,
                        max_points=6000, figsize=(15, 11), seed=0):
    """
    Four-panel overview: latent codes, decoder field, generative loops, coverage.
    Prints the estimated winding matrix, which is the diagnostic that actually
    settles whether a low distance-correlation is a valid automorphism or a bad fit.
    """
    mu_all = _np(model.encode_angles(data))
    ang = _np(angles) if angles is not None else None

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(mu_all), size=min(max_points, len(mu_all)), replace=False)

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    plot_latent_scatter(
        ax1, mu_all[sub],
        color=ang[sub, 0] if ang is not None else None,
        cbar_label="ground-truth theta_1" if ang is not None else None,
        title="latent codes mu(x) on the torus")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    plot_decoder_field(ax2, decoder_field_fn(model, channel),
                       title=f"decoder output, channel {channel}",
                       cbar_label=f"x_hat[{channel}]")

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    if ang is not None:
        loops, A = [], np.zeros((2, 2), dtype=int)
        for axis in (0, 1):
            idx = loop_indices(ang, axis=axis, level=0.0)
            if len(idx) < 16:
                continue
            mu_loop = _np(model.encode_angles(data[idx]))
            _, w = winding_numbers(mu_loop)
            A[:, axis] = w
            loops.append((f"theta_{axis+1} loop  ->  winding {tuple(w)}", mu_loop))
        plot_loops(ax3, loops, title="images of generative loops")
        print("estimated winding matrix A (columns = images of the two "
              "generative cycles):")
        print(A)
        print("det A =", int(round(np.linalg.det(A))),
              "  (|det| = 1 means a genuine automorphism; "
              "larger means a covering map)")
    else:
        ax3.set_axis_off()
        ax3.text2D(0.5, 0.5, "no ground-truth angles supplied",
                   ha="center", transform=ax3.transAxes, fontsize=9)

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    plot_coverage(ax4, mu_all, title="aggregate posterior coverage")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"saved {save}")
    return fig