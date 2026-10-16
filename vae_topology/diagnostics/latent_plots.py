"""Encoded-latent-code plots and the per-model `diagnose*`/`run_diagnostics*`
entry points.

Three latent geometries, three plotting conventions, because a shared "plot
the codes" function would either fake or hide the actual topology:
  - Euclidean R^2  (`run_diagnostics`, for `models.euclidean_vae`):
    scatter directly in (theta1, theta2), plus a pullback-curvature heatmap.
  - Euclidean R^3  (`run_diagnostics_3d`, for `models.euclidean_vae` with
    latent_dim=3): a 3d point cloud, plus singular values to check whether
    the codes actually use all three dimensions or collapsed onto a sheet.
  - Torus T^2      (`diagnose` -> `plot_torus_latent`, for `models.torus_vae`):
    scatter both on the embedded donut surface and in the flat (theta1,
    theta2) fundamental domain, coloured by the TRUE generative angle with a
    cyclic colormap (so a correctly learned map shows no fake seam).
  - Sphere S^2     (`diagnose_sphere` -> `plot_sphere_latent`, for
    `models.sphere_vae`): scatter on the embedded sphere and on a Mollweide
    projection (area-preserving, so uniform coverage of S^2 looks uniform in
    the panel -- S^2 has no fundamental domain to fall back on the way a
    torus does). The polar angle theta uses a sequential colormap (it is NOT
    cyclic; a cyclic colormap would paint theta=0 and theta=pi the same and
    fake a seam that is not there), the azimuth phi uses a cyclic one.

Consolidated from `visualization/vaediagnostic.py` and
`visualization/vaediagnostics_3d.py`; curvature/Gauss-Bonnet computation now
comes from `vae_topology.geometry` rather than a private copy.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from vae_topology.geometry import (
    curvature_map_euclidean, disk_from_codes, gauss_bonnet_disk,
    gauss_bonnet_from_curvature_map, pullback_metric_fn,
)
from vae_topology.diagnostics.curvature_plots import plot_gauss_bonnet


def _np(a):
    """Tensor / array / list -> detached numpy array."""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


# ============================================================ Euclidean R^2

def plot_reconstructions(model, data, n=8, res=28, ax=None):
    """Top row: originals. Bottom row: reconstructions (decoded from mu, no
    posterior noise)."""
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(data[:n])
        rec = model.decode(mu)
    err = ((rec - data[:n]) ** 2).sum(1).sqrt()
    if ax is None:
        fig, ax = plt.subplots(2, n, figsize=(1.3 * n, 3))
    for i in range(n):
        ax[0, i].imshow(data[i].reshape(res, res).cpu(), cmap="magma", vmin=0, vmax=1)
        ax[1, i].imshow(rec[i].reshape(res, res).cpu(), cmap="magma", vmin=0, vmax=1)
        ax[0, i].axis("off"); ax[1, i].axis("off")
        ax[1, i].set_title(f"{err[i]:.2f}", fontsize=7)
    ax[0, 0].set_ylabel("input"); ax[1, 0].set_ylabel("recon")
    return err


def run_diagnostics(model, data, angles=None, grid=40, chunk=64,
                    fname="vae_diagnostics.pdf", cyclic=(True, True),
                    show_codes=False, s=4, panel_labels=("a)", "b)"),
                    label_size=14, gauss_bonnet=False,
                    gb_pad=0.05, gb_n_r=64, gb_n_theta=128, gb_n_bdry=512):
    """Diagnostics for a standard Euclidean-latent (`models.euclidean_vae`)
    VAE with a 2d latent. Writes two figures:

        <stem>_latent<ext>      latent codes coloured by each true angle
        <stem>_curvature<ext>   log10|K| and log10 sqrt(det g), pullback metric

    `cyclic` says which ground-truth coordinate wraps: (True, True) for the
    torus/plane datasets, (False, True) for the sphere, whose polar angle is
    not cyclic. Colouring a non-cyclic coordinate with hsv paints both ends
    the same and fakes a seam.

    `gauss_bonnet=True` additionally runs the disk check (`geometry.
    gauss_bonnet_disk`) on the circle enclosing the codes and writes
    <stem>_gaussbonnet<ext>. The Cartesian grid computed here is reused for a
    cheap cross-check of the area term; the quoted numbers come from the
    polar quadrature, which actually converges (see `geometry.
    gauss_bonnet_from_curvature_map`).
    """
    # .cpu() here, not just .double(): if training happened on cuda (the
    # Colab GPU path), leaving model/data there made the curvature pass
    # below mix cuda and cpu tensors and crash. Found via
    # Colab_run_files/flat_vae_run_analyse (2).ipynb, whose `run()` worked
    # around exactly this by moving both back to cpu itself before calling
    # in; fixed at the source here instead so every caller gets it for free.
    model = model.double().cpu()
    data = data.double().cpu()
    with torch.no_grad():
        mu, _ = model.encode(data)

    K, M, extent = curvature_map_euclidean(model, mu, grid=grid, chunk=chunk)

    m = _np(mu)
    true = _np(angles) if angles is not None else None
    stem, ext = os.path.splitext(fname)
    ext = ext or ".pdf"

    # ---------------------------------------------------- figure 1: latents
    fig_lat, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for j, ax in enumerate(axes):
        if true is not None and true.shape[1] > j:
            if cyclic[j]:
                c, cmap, vmin, vmax = np.mod(true[:, j], 2 * np.pi), "twilight", 0.0, 2 * np.pi
            else:
                c, cmap, vmin, vmax = true[:, j], "viridis", None, None
        else:
            c, cmap, vmin, vmax = None, "viridis", None, None

        sc = ax.scatter(m[:, 0], m[:, 1], s=s, c=c, cmap=cmap,
                        vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        ax.text(-0.12, 1.02, panel_labels[j], transform=ax.transAxes,
                fontsize=label_size, fontweight="bold", va="bottom", ha="left")
        if c is not None:
            fig_lat.colorbar(sc, ax=ax, fraction=0.046, pad=0.02,
                             label=[r"true $\gamma_1$", r"true $\gamma_2$"][j])
    fig_lat.tight_layout()
    fig_lat.savefig(f"{stem}_latent{ext}", dpi=150, bbox_inches="tight")

    # ------------------------------------------------- figure 2: curvature
    # masked_invalid: det(g) underflows where the decoder is flat, inv(g)
    # blows up, and NaN spreads through the curvature stack. An all-NaN
    # array passed to imshow renders as blank axes with no indication why.
    logK = np.ma.masked_invalid(np.log10(np.abs(K) + 1e-12))
    logM = np.ma.masked_invalid(np.log10(M + 1e-12))

    fig_curv, axc = plt.subplots(1, 2, figsize=(12, 5))
    for ax, arr, cmap, lab in ((axc[0], logK, "inferno", r"$\log_{10}|K|$"),
                               (axc[1], logM, "viridis",
                                r"$\log_{10}\sqrt{\det G}$")):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("0.6")
        im = ax.imshow(arr, origin="lower", extent=extent, cmap=cm,
                       aspect="equal")
        if show_codes:
            ax.scatter(m[:, 0], m[:, 1], s=1, c="cyan", alpha=0.15, linewidths=0)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_ylabel(r"$\theta_2$")
        fig_curv.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=lab)

    for ax, lab in zip(axc, panel_labels):
        ax.text(-0.12, 1.02, lab, transform=ax.transAxes,
                fontsize=label_size, fontweight="bold", va="bottom", ha="left")

    fig_curv.tight_layout()
    fig_curv.savefig(f"{stem}_curvature{ext}", dpi=150, bbox_inches="tight")

    finite = np.isfinite(K)
    if finite.any():
        print(f"  |K|  : median {np.median(np.abs(K[finite])):.3e}   "
              f"max {np.abs(K[finite]).max():.3e}")
    print(f"  sqrt(det g): median {np.median(M):.3e}   max {M.max():.3e}")
    if not finite.all():
        print(f"  [non-finite on {100 * (~finite).mean():.1f}% of the grid, "
              f"drawn grey]")

    # --------------------------------------------- figure 3: Gauss-Bonnet
    gb = None
    if gauss_bonnet:
        metric_fn = pullback_metric_fn(model.decode)
        c, R = disk_from_codes(mu, pad=gb_pad)
        print("\n  Gauss-Bonnet on the disk enclosing the codes:")
        gb = gauss_bonnet_disk(metric_fn, c, R, n_r=gb_n_r, n_theta=gb_n_theta,
                               n_bdry=gb_n_bdry, chunk=chunk)
        print("  cross-check against the Cartesian curvature grid above "
              "(expect ~1e-3 disagreement from the ragged boundary):")
        gauss_bonnet_from_curvature_map(K, M, extent, c, R)
        plot_gauss_bonnet(gb, mu_codes=mu, fname=f"{stem}_gaussbonnet{ext}")

    return fig_lat, fig_curv, K, M, gb


# ============================================================ Euclidean R^3

def _depth_sort(pts, elev, azim):
    """Draw far points first, so the near side of the cloud is on top."""
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return np.argsort(pts @ d)


@torch.no_grad()
def run_diagnostics_3d(model, data, angles=None, n_points=4000, seed=0, s=5,
                       cyclic=(True, True), elev=22.0, azim=-60.0, res=28,
                       panel_labels=("a)", "b)"), label_size=14,
                       fname="vae_diagnostics_3d.png"):
    """Diagnostic for a standard (Euclidean) VAE with a 3-dimensional latent.

    The latent cloud in R^3, coloured by each ground-truth angle.

    `cyclic` says which ground-truth coordinates are angles that wrap: use
    (True, True) for the torus/plane datasets and (False, True) for the
    sphere, whose polar angle theta in [0, pi] is NOT cyclic. Colouring a
    non-cyclic coordinate with hsv paints both ends the same and fakes a seam.

    The singular values are still computed and printed: if the third is not
    small, the codes genuinely occupy three dimensions (which a torus needs);
    if it collapses, the model has flattened the data onto a sheet and
    cannot be representing a closed surface.
    """
    model = model.double().eval()
    data = data.double()

    mu, _ = model.encode(data)
    lat = _np(mu)
    if lat.shape[1] != 3:
        raise ValueError(f"expected a 3d latent, got {lat.shape[1]}")

    n = min(n_points, lat.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(lat.shape[0], size=n, replace=False)
    sub = lat[idx]
    true = _np(angles)[idx] if angles is not None else None

    order = _depth_sort(sub, elev, azim)
    sub = sub[order]
    if true is not None:
        true = true[order]

    fig = plt.figure(figsize=(12, 5.5))

    for j in range(2):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        if true is not None and true.shape[1] > j:
            if cyclic[j]:
                c = np.mod(true[:, j], 2 * np.pi)
                cmap, vmin, vmax = "twilight", 0.0, 2 * np.pi
            else:
                c = true[:, j]
                cmap, vmin, vmax = "viridis", None, None
            lbl = rf"true $\gamma_{j + 1}$"
        else:
            c, cmap, vmin, vmax, lbl = None, "viridis", None, None, "(no labels)"

        p = ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=c, cmap=cmap,
                       vmin=vmin, vmax=vmax, s=s, alpha=0.85, linewidths=0,
                       depthshade=False)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.set_zlabel("$z_3$")
        ax.set_box_aspect((1, 1, 1))
        ax.text2D(0.02, 0.95, panel_labels[j], transform=ax.transAxes,
                  fontsize=label_size, fontweight="bold", va="top", ha="left")
        if c is not None:
            fig.colorbar(p, ax=ax, shrink=0.55, pad=0.08, label=lbl)

    fig.tight_layout()
    fig.savefig(fname, dpi=130, bbox_inches="tight")

    centred = lat - lat.mean(0, keepdims=True)
    sv = np.linalg.svd(centred, compute_uv=False) / np.sqrt(len(lat))
    print(f"latent std per axis : {lat.std(0)}")
    print(f"singular values     : {sv}")
    print(f"s3/s1               : {sv[2] / sv[0]:.4f}   "
          f"(near 0 => codes lie on a 2d sheet)")
    print(f"radius from centroid: mean {np.linalg.norm(centred, axis=1).mean():.4f}"
          f"  min {np.linalg.norm(centred, axis=1).min():.4f}")
    return lat, sv


# =================================================================== torus

def torus_xyz(theta, R=2.0, r=1.0):
    t1, t2 = theta[:, 0], theta[:, 1]
    x = (R + r * np.cos(t2)) * np.cos(t1)
    y = (R + r * np.cos(t2)) * np.sin(t1)
    z = r * np.sin(t2)
    return x, y, z


def _wrap_pm_pi(a):
    """Angles into [-pi, pi), matching the curvature grid's domain."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def plot_torus_latent(latent_angles, true_angles, R=2.0, r=1.0, n_points=4000,
                      seed=0, s=6, cyclic=True, show_2d=True, save=None,
                      panel_labels=("a)", "b)"), label_size=14):
    lat = _np(latent_angles)
    true = _np(true_angles)

    n = min(n_points, lat.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(lat.shape[0], size=n, replace=False)
    lat, true = lat[idx], true[idx]

    x, y, z = torus_xyz(lat, R, r)
    order = np.argsort(z)
    x, y, z, lat, true = x[order], y[order], z[order], lat[order], true[order]

    # domain coordinates on [-pi, pi), so (0, 0) sits at the centre of the
    # panel exactly as it does in plot_abs_curvature
    lat_x, lat_y = _wrap_pm_pi(lat[:, 0]), _wrap_pm_pi(lat[:, 1])
    ticks = [-3, -2, -1, 0, 1, 2, 3]

    cmap = "twilight" if cyclic else "viridis"
    vmin, vmax = (0.0, 2 * np.pi) if cyclic else (None, None)
    cbar_labels = [r"true $\gamma_1$", r"true $\gamma_2$"]

    fig3d = plt.figure(figsize=(12, 5.5))
    fig2d = plt.figure(figsize=(12, 5.5)) if show_2d else None

    for j in range(2):
        c = np.mod(true[:, j], 2 * np.pi) if cyclic else true[:, j]

        ax = fig3d.add_subplot(1, 2, j + 1, projection="3d")
        p = ax.scatter(x, y, z, c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                       s=s, alpha=0.85, linewidths=0, depthshade=False)
        ax.set_box_aspect((1, 1, 0.45))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.text2D(0.02, 0.93, panel_labels[j], transform=ax.transAxes,
                  fontsize=label_size, fontweight="bold", va="top", ha="left")
        fig3d.colorbar(p, ax=ax, shrink=0.6, pad=0.02, label=cbar_labels[j])

        if show_2d:
            ax2 = fig2d.add_subplot(1, 2, j + 1)
            sc = ax2.scatter(lat_x, lat_y, c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                             s=s, alpha=0.85, linewidths=0)
            ax2.set_xlim(-np.pi, np.pi)
            ax2.set_ylim(-np.pi, np.pi)
            ax2.set_xticks(ticks)
            ax2.set_yticks(ticks)
            ax2.set_aspect("equal")
            ax2.set_xlabel(r"$\theta_1$")
            ax2.set_ylabel(r"$\theta_2$")
            ax2.text(-0.12, 1.02, panel_labels[j], transform=ax2.transAxes,
                     fontsize=label_size, fontweight="bold", va="bottom", ha="left")
            fig2d.colorbar(sc, ax=ax2, shrink=0.8, pad=0.02,
                           label=cbar_labels[j])

    for f in (fig3d, fig2d):
        if f is not None:
            f.tight_layout()

    if save:
        stem, ext = os.path.splitext(save)
        ext = ext or ".pdf"
        fig3d.savefig(f"{stem}_torus{ext}", dpi=150, bbox_inches="tight")
        if fig2d is not None:
            fig2d.savefig(f"{stem}_domain{ext}", dpi=150, bbox_inches="tight")

    return fig3d, fig2d


def diagnose(model, data, true_angles, save, batch_size=4096, **kwargs):
    """Entry point used by `experiments.run_all_topologies` for
    `models.torus_vae.TorusVAE`."""
    model.eval()
    if hasattr(model, "encode_angles"):
        mu = model.encode_angles(data)
    else:
        mu = torch.cat([model.encode(data[i:i + batch_size])[0]
                        for i in range(0, data.shape[0], batch_size)], dim=0)
    return plot_torus_latent(mu, true_angles, save=save, **kwargs)


# =================================================================== sphere

def to_angles(arr):
    """Accept either (N, 3) unit vectors or (N, 2) angles; return (N, 2)
    angles (theta, phi) with theta in [0, pi], phi in [-pi, pi]."""
    a = _np(arr)
    if a.shape[1] == 3:
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        theta = np.arccos(np.clip(a[:, 2], -1.0, 1.0))
        phi = np.arctan2(a[:, 1], a[:, 0])
        return np.stack([theta, phi], axis=1)
    return a


def sphere_xyz(angles, radius=1.0):
    theta, phi = angles[:, 0], angles[:, 1]
    st = np.sin(theta)
    return (radius * st * np.cos(phi),
            radius * st * np.sin(phi),
            radius * np.cos(theta))


def plot_sphere_latent(latent, true_angles, n_points=4000, seed=0, s=6,
                       elev=25.0, azim=-60.0, show_2d=True, save=None,
                       panel_labels=("a)", "b)"), label_size=14):
    """Sphere analogue of plot_torus_latent.

    Two differences from the torus case, both forced by the geometry:
      * the two true angles are NOT both cyclic. theta in [0, pi] is a polar
        angle (viridis); phi in [0, 2pi) is cyclic (hsv). Using a cyclic
        colormap for theta would fake a seam that is not there.
      * the 2d panel is a Mollweide projection, not a fundamental domain.
        S^2 has no fundamental domain; Mollweide is area-preserving, so
        uniform coverage of the sphere looks uniform in the panel.

    Returns (fig3d, fig2d); fig2d is None when show_2d is False.
    """
    lat = to_angles(latent)
    true = _np(true_angles)

    n = min(n_points, lat.shape[0])
    rng = np.random.default_rng(seed)
    idx = rng.choice(lat.shape[0], size=n, replace=False)
    lat, true = lat[idx], true[idx]

    x, y, z = sphere_xyz(lat)

    # depth-sort along the camera direction so the near hemisphere is drawn last
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    order = np.argsort(np.stack([x, y, z], axis=1) @ d)
    x, y, z, lat, true = x[order], y[order], z[order], lat[order], true[order]

    # colour: polar angle is not cyclic, azimuth is
    specs = [("viridis", 0.0, np.pi, r"true polar"),
             ("twilight", 0.0, 2 * np.pi, r"true azimuth")]

    fig3d = plt.figure(figsize=(12, 5.5))
    fig2d = plt.figure(figsize=(12, 5.5)) if show_2d else None

    for j, (cmap, vmin, vmax, label) in enumerate(specs):
        c = true[:, j] if j == 0 else np.mod(true[:, j], 2 * np.pi)

        ax = fig3d.add_subplot(1, 2, j + 1, projection="3d")
        u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
        ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v),
                          np.cos(v), color="0.85", linewidth=0.3)
        p = ax.scatter(x, y, z, c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                       s=s, alpha=0.9, linewidths=0, depthshade=False)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.text2D(0.02, 0.95, panel_labels[j], transform=ax.transAxes,
                  fontsize=label_size, fontweight="bold", va="top", ha="left")
        fig3d.colorbar(p, ax=ax, shrink=0.6, pad=0.02, label=label)

        if show_2d:
            ax2 = fig2d.add_subplot(1, 2, j + 1, projection="mollweide")
            lon = (lat[:, 1] + np.pi) % (2 * np.pi) - np.pi
            latitude = np.pi / 2 - lat[:, 0]
            sc = ax2.scatter(lon, latitude, c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                             s=s, alpha=0.9, linewidths=0)
            ax2.grid(True, linewidth=0.3, color="0.8")
            ax2.set_xticklabels([])
            ax2.text(0.0, 1.05, panel_labels[j], transform=ax2.transAxes,
                     fontsize=label_size, fontweight="bold", va="bottom", ha="left")
            fig2d.colorbar(sc, ax=ax2, fraction=0.03, pad=0.04, label=label)

    for f in (fig3d, fig2d):
        if f is not None:
            f.tight_layout()

    if save:
        stem, ext = os.path.splitext(save)
        ext = ext or ".pdf"
        fig3d.savefig(f"{stem}_sphere{ext}", dpi=150, bbox_inches="tight")
        if fig2d is not None:
            fig2d.savefig(f"{stem}_mollweide{ext}", dpi=150, bbox_inches="tight")

    return fig3d, fig2d


@torch.no_grad()
def diagnose_sphere(model, data, true_angles, save, batch_size=4096, **kwargs):
    """Entry point used by `experiments.run_all_topologies` for
    `models.sphere_vae.SphereVAE`."""
    model.eval()
    mu = torch.cat([model.encode(data[i:i + batch_size])[0]
                    for i in range(0, data.shape[0], batch_size)], dim=0)
    return plot_sphere_latent(mu, true_angles, save=save, **kwargs)
