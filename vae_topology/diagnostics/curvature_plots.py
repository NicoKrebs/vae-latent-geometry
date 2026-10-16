"""|K| and sqrt(det G) maps over a whole latent manifold, plus the
Gauss-Bonnet-disk summary figure.

The two `plot_abs_curvature*` functions are the figures behind the thesis's
central numerical claim: that the integral of the Fisher-metric Gaussian
curvature over the trained latent manifold reproduces the correct Euler
characteristic (0 for the torus, printed as "int K dA (should be ~0)"; 4*pi
= 2*pi*chi(S^2) for the sphere, printed against the `FOUR_PI` target).

Consolidated from `visualization/curvature_plot.py` (torus) and
`visualization/vaediagnostic.py` (sphere, Gauss-Bonnet); both now call the
shared curvature evaluators in `vae_topology.geometry` instead of keeping
their own copies.

Bug fixed here that was present in BOTH originals: every caller
(`experiments/run_all_topologies.py`, the Colab driver) passes `mu=` to
overlay the encoded latent codes on the curvature map, but the original
`plot_abs_curvature`/`plot_abs_curvature_sphere` computed `mu` and then
never actually plotted it -- a silent no-op with no error, so every
curvature figure ever produced by this code was missing that overlay. Now
implemented (small cyan scatter on both panels).
"""

import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

from vae_topology.geometry import (
    FOUR_PI, TWO_PI, fisher_curvature_map_sphere, fisher_curvature_map_torus,
)


def _np(a):
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def plot_abs_curvature(model, grid=100, chunk=32, device="cpu", mu=None,
                       log_sigma_fn=None, fname="fisher_curvature.png",
                       save_clean=False, panel_labels=("a)", "b)"), label_size=14,
                       log_path="curvature_log.txt", log_tag=None):
    """|K| and sqrt(det G) of the Fisher metric over the latent TORUS.

    `mu`, if given (encoded latent codes, (N, 2) angles), is scattered on top
    of both panels -- this shows WHERE on the manifold the data actually
    lives, since curvature far from any encoded code is extrapolation the
    decoder was never trained on. (Every caller in `experiments/` already
    passed `mu` expecting this overlay; earlier versions of this function
    computed it and then silently never plotted it -- fixed here.)

    Set `save_clean=True` to ALSO save a copy with no code overlay, at
    `fname` with "_clean" inserted before the extension -- e.g. for a paper
    figure, where the diagnostic scatter is noise you don't want, but you
    still want the overlay version around for checking where the data
    actually lives relative to the curvature. Has no effect if `mu=None`
    (nothing to add in the first place; `fname` alone is already clean).
    """
    K, V, t1, t2 = fisher_curvature_map_torus(model, grid, chunk, device, log_sigma_fn)

    dA = (TWO_PI / grid) ** 2
    total = float((K * V).sum() * dA)
    area = float(V.sum() * dA)

    extent = [-math.pi, math.pi, -math.pi, math.pi]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im0 = ax[0].imshow(np.log10(np.abs(K) + 1e-30), origin="lower",
                       extent=extent, cmap="inferno", aspect="equal")
    fig.colorbar(im0, ax=ax[0], fraction=0.046,
                 label=r"$\log_{10}|K|$")

    im1 = ax[1].imshow(np.log10(V + 1e-30), origin="lower",
                       extent=extent, cmap="viridis", aspect="equal")
    fig.colorbar(im1, ax=ax[1], fraction=0.046,
                 label=r"$\log_{10}\sqrt{\det G}$")

    for a, lab in zip(ax, panel_labels):
        a.set_xlabel(r"$\theta_1$")
        a.set_ylabel(r"$\theta_2$")
        a.text(-0.12, 1.02, lab, transform=a.transAxes,
               fontsize=label_size, fontweight="bold", va="bottom", ha="left")
    fig.tight_layout()

    if save_clean and mu is not None:
        stem, ext = os.path.splitext(fname)
        fig.savefig(f"{stem}_clean{ext}", dpi=150, bbox_inches="tight")

    if mu is not None:
        m = _np(mu)
        m_wrapped = (m + math.pi) % TWO_PI - math.pi
        for a in ax:
            a.scatter(m_wrapped[:, 0], m_wrapped[:, 1], s=2, c="cyan",
                     alpha=0.25, linewidths=0)

    fig.savefig(fname, dpi=150, bbox_inches="tight")

    abs_total = float((np.abs(K) * V).sum() * dA)
    rel = abs(total) / abs_total if abs_total > 0 else float("nan")

    lines = [
        f"|K|          : median {np.median(np.abs(K)):.3e}  max {np.abs(K).max():.3e}",
        f"sqrt(det G)  : median {np.median(V):.3e}  max {V.max():.3e}",
        f"Fisher area  : {area:.6e}",
        f"int K dA     : {total:.6e}   (should be ~0; relative {rel:.2e})",
    ]
    for line in lines:
        print(line)

    if log_path:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"[{stamp}] plot_abs_curvature  grid={grid}  figure={fname}"
        if log_tag:
            header += f"  tag={log_tag}"
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(header + "\n")
            fh.write("\n".join(lines) + "\n\n")

    return fig, K, V


def plot_abs_curvature_sphere(model, n_theta=64, n_phi=128, eps=0.08,
                              chunk=32, device="cpu", mu=None,
                              log_sigma_fn=None,
                              fname="fisher_curvature_sphere.png",
                              save_clean=False,
                              panel_labels=("a)", "b)"), label_size=14,
                              log_path="curvature_log.txt", log_tag=None):
    """|K| of the Fisher metric over the latent SPHERE, in an equal-area
    Mollweide projection. Returns (fig, K, V).

    `mu`, if given (encoded latent codes, either (N, 3) unit vectors or
    (N, 2) (theta, phi) angles), is scattered on both panels -- see the note
    in `plot_abs_curvature` about why this matters and why it was previously
    a no-op.

    Set `save_clean=True` to ALSO save a code-overlay-free copy at `fname`
    with "_clean" inserted before the extension -- see `plot_abs_curvature`.
    """
    K, V, theta, phi = fisher_curvature_map_sphere(
        model, n_theta, n_phi, eps, chunk, device, log_sigma_fn)

    dth = (math.pi - 2.0 * eps) / n_theta
    dph = TWO_PI / n_phi
    dA = dth * dph
    total = float((K * V).sum() * dA)
    area = float(V.sum() * dA)
    abs_total = float((np.abs(K) * V).sum() * dA)
    pos_frac = float((K > 0).mean())

    # sqrt(det G) contains the coordinate factor sin(theta), which vanishes
    # at the poles for reasons that have nothing to do with the model.
    # Dividing it out gives the area magnification RELATIVE TO THE ROUND
    # SPHERE, the coordinate-free quantity worth looking at.
    V_rel = V / np.sin(theta)[:, None]

    LON, LAT = np.meshgrid(phi, np.pi / 2 - theta)

    fig = plt.figure(figsize=(13, 5.5))
    ax0 = fig.add_subplot(1, 2, 1, projection="mollweide")
    m0 = ax0.pcolormesh(LON, LAT, np.log10(np.abs(K) + 1e-30),
                        cmap="inferno", shading="auto")
    fig.colorbar(m0, ax=ax0, fraction=0.03, pad=0.04,
                 label=r"$\log_{10}|K|$")

    ax1 = fig.add_subplot(1, 2, 2, projection="mollweide")
    m1 = ax1.pcolormesh(LON, LAT, np.log10(V_rel + 1e-30),
                        cmap="viridis", shading="auto")
    fig.colorbar(m1, ax=ax1, fraction=0.03, pad=0.04,
                 label=r"$\log_{10}\left(\sqrt{\det G}/\sin\theta\right)$")

    for a, lab in zip((ax0, ax1), panel_labels):
        a.grid(True, linewidth=0.3, color="0.7")
        a.set_xticklabels([])
        a.text(0.0, 1.05, lab, transform=a.transAxes,
               fontsize=label_size, fontweight="bold", va="bottom", ha="left")
    fig.tight_layout()

    if save_clean and mu is not None:
        stem, ext = os.path.splitext(fname)
        fig.savefig(f"{stem}_clean{ext}", dpi=150, bbox_inches="tight")

    if mu is not None:
        m = _np(mu)
        if m.shape[-1] == 3:                      # unit vectors -> angles
            m = m / np.linalg.norm(m, axis=-1, keepdims=True)
            th = np.arccos(np.clip(m[:, 2], -1.0, 1.0))
            ph = np.arctan2(m[:, 1], m[:, 0])
        else:                                      # already (theta, phi)
            th, ph = m[:, 0], m[:, 1]
        mu_lon = (ph + math.pi) % TWO_PI - math.pi
        mu_lat = math.pi / 2 - th
        for a in (ax0, ax1):
            a.scatter(mu_lon, mu_lat, s=2, c="cyan", alpha=0.25, linewidths=0)

    fig.savefig(fname, dpi=150, bbox_inches="tight")

    lines = [
        f"|K|          : median {np.median(np.abs(K)):.3e}  max {np.abs(K).max():.3e}",
        f"sqrt(det G)  : median {np.median(V):.3e}  max {V.max():.3e}",
        f"K > 0 on     : {100 * pos_frac:.1f}% of the grid",
        f"Fisher area  : {area:.6e}",
        f"int K dA     : {total:.6f}   target 4pi = {FOUR_PI:.6f}   "
        f"deviation {abs(total - FOUR_PI) / FOUR_PI:.2%}",
        f"int |K| dA   : {abs_total:.6f}   (>= |int K dA|; equal iff K has "
        f"one sign)",
        f"[eps = {eps}: excluded caps carry ~2pi*eps^2 = "
        f"{2 * math.pi * eps ** 2:.4f} of curvature for a round metric]",
    ]
    for line in lines:
        print(line)

    if log_path:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = (f"[{stamp}] plot_abs_curvature_sphere  "
                  f"n_theta={n_theta}  n_phi={n_phi}  figure={fname}")
        if log_tag:
            header += f"  tag={log_tag}"
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(header + "\n")
            fh.write("\n".join(lines) + "\n\n")

    return fig, K, V


def plot_gauss_bonnet(res, mu_codes=None, fname="gauss_bonnet.pdf",
                      panel_labels=("a)", "b)", "c)"), label_size=14):
    """Three panels for a `geometry.gauss_bonnet_disk` result: log|K| on the
    polar disk, the boundary integrand k_g * speed against t, and the
    cumulative radial mass of int K dA.

    The third panel is the useful diagnostic. If the cumulative curve is a
    staircase -- flat then jumping -- the radial grid is barely resolving the
    curvature peaks and the residual is quadrature error, not geometry.
    """
    TAU = TWO_PI
    K = res["K_polar"]
    V = res["V_polar"]
    n_r, n_theta = K.shape
    R = res["radius"]
    dr = R / n_r
    dth = TAU / n_theta
    r = (np.arange(n_r) + 0.5) * dr
    th = (np.arange(n_theta) + 0.5) * dth

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    axp = plt.subplot(1, 3, 1, projection="polar")
    TH, RR = np.meshgrid(th, r)
    logK = np.ma.masked_invalid(np.log10(np.abs(K) + 1e-30))
    cm = plt.get_cmap("inferno").copy()
    cm.set_bad("0.6")
    pm = axp.pcolormesh(TH, RR, logK, cmap=cm, shading="auto")
    axp.set_yticklabels([])
    fig.colorbar(pm, ax=axp, fraction=0.046, pad=0.08, label=r"$\log_{10}|K|$")
    axp.text(0.0, 1.05, panel_labels[0], transform=axp.transAxes,
             fontsize=label_size, fontweight="bold", va="bottom", ha="left")
    ax[0].remove()

    t = (np.arange(res["kg"].size) + 0.5) * (TAU / res["kg"].size)
    ax[1].plot(t, res["kg"] * res["speed"], lw=1.2)
    ax[1].axhline(0.0, color="0.7", lw=0.6)
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel(r"$k_g\,\|\dot\gamma\|_g$")
    ax[1].set_title(rf"$\oint k_g\,ds = {res['boundary_term']:+.4f}$", fontsize=10)
    ax[1].text(-0.12, 1.02, panel_labels[1], transform=ax[1].transAxes,
               fontsize=label_size, fontweight="bold", va="bottom", ha="left")

    shell = np.nansum(np.where(np.isfinite(K * V), K * V, 0.0), axis=1) * r * dr * dth
    ax[2].plot(r, np.cumsum(shell), lw=1.2)
    ax[2].axhline(res["area_term"], color="0.7", ls="--", lw=0.8)
    ax[2].set_xlabel(r"$r$")
    ax[2].set_ylabel(r"$\int_{|z-c|<r} K\,dA$")
    ax[2].set_title(rf"$\int K\,dA = {res['area_term']:+.4f}$, "
                    rf"sum $= {res['total']:.4f}$ vs $2\pi = {TAU:.4f}$",
                    fontsize=10)
    ax[2].text(-0.12, 1.02, panel_labels[2], transform=ax[2].transAxes,
               fontsize=label_size, fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    return fig
