"""Render a GIF of how the latent codes and the Fisher-metric curvature map
evolve over training, from a run's saved checkpoints.

    python -m vae_topology.experiments.render_evolution results/topologies__.../torus_torus
    python -m vae_topology.experiments.render_evolution <run_dir> --grid 30 --fps 10

Two-step, mirroring the rest of this package's "compute once, replot freely"
split (see `run_store.py`'s docstring):

    compute_evolution(run_dir, ...)        reloads EVERY checkpoint in
                                            `run_dir/checkpoints/`, encodes
                                            the training data and computes
                                            the Fisher curvature map at each
                                            one, and saves the whole stack to
                                            `arrays/evolution.npz`. This is
                                            the expensive step (one curvature
                                            evaluation per checkpoint).

    render_evolution_video(run_dir, ...)   reads that npz back and renders
                                            one PNG frame per checkpoint
                                            (latent scatter | curvature map,
                                            side by side, on a FIXED colour
                                            scale computed across the whole
                                            run -- without this, matplotlib's
                                            default per-frame autoscaling
                                            makes every frame's colour scheme
                                            jump independently, and you can
                                            no longer tell whether curvature
                                            is actually growing or shrinking
                                            over training), then assembles
                                            them into a GIF. Cheap: no model
                                            evaluation, just plotting.

Needs checkpoints saved densely enough to look smooth: pass
`--checkpoint-every` (a small number, e.g. 1-5) to `run_all_topologies.py`
(or `run_euclidean_2d.py`/`run_euclidean_3d.py`) when training the model you
want to animate -- the default (~50 checkpoints spread over the whole run)
still works, just coarser. This works on ANY run that has checkpoints,
whether or not it was trained specifically for a video -- `compute_evolution`
only needs `checkpoints/` and `meta.json`, nothing that marks a run as
"for animation". Retraining is only needed if the existing checkpoints are
too sparse for a smooth result, not because the run wasn't labelled for it.

Covers all three latent geometries: torus, sphere (`models.torus_vae`/
`models.sphere_vae`, dispatched via `meta["latent_manifold"]`), and the flat
Euclidean 2d latent (`models.euclidean_vae.SimpleVAE`). The flat case needs
one extra step the compact manifolds don't: T^2/S^2 have a fixed coordinate
domain ([-pi, pi) etc.) that is the same for every checkpoint, but a flat
R^2 latent's actual extent is data-dependent and drifts over training, so
plotting each frame with its own auto-fit extent would make the axes
themselves rescale/pan frame to frame -- indistinguishable from the latent
space actually changing. `compute_evolution` fixes this by computing ONE
extent from the union of all checkpoints' codes up front, then reusing it
for every frame's curvature grid and scatter plot (same principle as the
fixed colour scale, applied to space instead of colour).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from vae_topology import datasets as datasets_mod
from vae_topology.models.torus_vae import TorusVAE
from vae_topology.models.sphere_vae import SphereVAE
from vae_topology.models.euclidean_vae import SimpleVAE
from vae_topology.experiments.run_store import load_checkpoint
from vae_topology.geometry import (
    curvature_map_euclidean, fisher_curvature_map_sphere, fisher_curvature_map_torus,
)

MODEL_CLASSES = {"TorusVAE": TorusVAE, "SphereVAE": SphereVAE, "SimpleVAE": SimpleVAE}
TWO_PI = 2.0 * np.pi


def _checkpoint_epochs(run_dir):
    cdir = os.path.join(run_dir, "checkpoints")
    files = sorted(f for f in os.listdir(cdir) if f.startswith("epoch_") and f.endswith(".pt"))
    if not files:
        raise FileNotFoundError(
            f"no checkpoints in {cdir} -- retrain with a smaller --checkpoint-every")
    return [int(f[len("epoch_"):-3]) for f in files]


def _load_model_at(run_dir, model_cls, meta, epoch):
    ckpt = load_checkpoint(run_dir, epoch=epoch)
    kwargs = dict(data_dim=meta["data_dim"])
    if model_cls is SimpleVAE:
        kwargs["latent_dim"] = meta.get("latent_dim", 2)
    model = model_cls(**kwargs)
    model.load_state_dict(ckpt["state_dict"])
    return model.double().eval()


def compute_evolution(run_dir, grid=40, n_theta=32, n_phi=64, chunk=32,
                      device="cpu", every=1, log_every=1, extent_pad=0.15):
    """Reload every `every`-th checkpoint in `run_dir`, compute (latent
    codes, curvature map) at each, and save the stack to
    `arrays/evolution.npz`. Returns that path.
    """
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)

    manifold = meta["latent_manifold"]
    model_cls = MODEL_CLASSES[meta["model_class"]]
    data_fn = getattr(datasets_mod, meta["dataset"])
    imgs, angles = data_fn()
    if imgs.shape[1] != meta["data_dim"]:
        raise ValueError(
            f"dataset parameters differ from training (data_dim {imgs.shape[1]} "
            f"vs recorded {meta['data_dim']}) -- adjust res/n to match")
    data = torch.as_tensor(imgs, dtype=torch.float32)

    epochs = _checkpoint_epochs(run_dir)[::every]
    print(f"{manifold} run, {len(epochs)} checkpoints "
          f"(epoch {epochs[0]} .. {epochs[-1]})")

    if manifold == "euclidean":
        return _compute_evolution_euclidean(
            run_dir, meta, model_cls, data_fn, data, angles, epochs,
            grid, chunk, device, extent_pad, log_every)

    Ks, Vs, Mus = [], [], []
    grid_coords = None
    for i, ep in enumerate(epochs):
        model = _load_model_at(run_dir, model_cls, meta, ep)

        with torch.no_grad():
            if manifold == "torus":
                mu = model.encode(data.double())[0].numpy()
            else:
                mu = model.encode_angles(data.double()).numpy()
        Mus.append(mu)

        if manifold == "torus":
            K, V, t1, t2 = fisher_curvature_map_torus(model, grid=grid, chunk=chunk, device=device)
            grid_coords = (t1, t2)
        else:
            K, V, th, ph = fisher_curvature_map_sphere(
                model, n_theta=n_theta, n_phi=n_phi, chunk=chunk, device=device)
            grid_coords = (th, ph)
        Ks.append(K)
        Vs.append(V)

        if log_every and (i % log_every == 0 or i == len(epochs) - 1):
            print(f"  [{i + 1:>3}/{len(epochs)}] epoch {ep:6d}  "
                  f"|K| median {np.median(np.abs(K)):.3e}  max {np.abs(K).max():.3e}")

    out = dict(
        epochs=np.array(epochs), K=np.stack(Ks), V=np.stack(Vs), mu=np.stack(Mus),
        angles=np.asarray(angles), manifold=manifold,
    )
    if manifold == "torus":
        out["t1"], out["t2"] = grid_coords
    else:
        out["theta"], out["phi"] = grid_coords

    out_dir = os.path.join(run_dir, "arrays")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "evolution.npz")
    np.savez_compressed(out_path, **out)
    print("saved", out_path)
    return out_path


def _compute_evolution_euclidean(run_dir, meta, model_cls, data_fn, data, angles,
                                 epochs, grid, chunk, device, extent_pad, log_every):
    """Flat-latent version of `compute_evolution`'s main loop.

    Two passes over the checkpoints: first load every model and encode (
    cheap -- one forward pass each) to find the union of all codes' ranges,
    THEN compute the curvature grid for each checkpoint using that ONE fixed
    extent (`geometry.curvature_map_euclidean`'s `lo`/`hi` override) -- see
    the module docstring for why the extent has to be fixed across frames
    here but not for the torus/sphere case.
    """
    models, Mus = [], []
    for ep in epochs:
        model = _load_model_at(run_dir, model_cls, meta, ep)
        with torch.no_grad():
            mu = model.encode(data.double())[0]
        models.append(model)
        Mus.append(mu)

    mu_all = torch.cat(Mus, dim=0)
    lo = mu_all.min(0).values
    hi = mu_all.max(0).values
    span = hi - lo
    lo = (lo - extent_pad * span).tolist()
    hi = (hi + extent_pad * span).tolist()

    Ks, Vs = [], []
    for i, (model, mu) in enumerate(zip(models, Mus)):
        K, V, extent = curvature_map_euclidean(model, mu, grid=grid, device=device,
                                               chunk=chunk, lo=lo, hi=hi)
        Ks.append(K)
        Vs.append(V)
        ep = epochs[i]
        if log_every and (i % log_every == 0 or i == len(epochs) - 1):
            print(f"  [{i + 1:>3}/{len(epochs)}] epoch {ep:6d}  "
                  f"|K| median {np.median(np.abs(K)):.3e}  max {np.abs(K).max():.3e}")

    out = dict(
        epochs=np.array(epochs), K=np.stack(Ks), V=np.stack(Vs),
        mu=np.stack([m.numpy() for m in Mus]), angles=np.asarray(angles),
        manifold="euclidean", extent=np.array(extent, dtype=np.float64),
        cyclic0=bool(meta.get("cyclic", [True, True])[0]),
    )
    out_dir = os.path.join(run_dir, "arrays")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "evolution.npz")
    np.savez_compressed(out_path, **out)
    print("saved", out_path)
    return out_path


def _render_torus_frame(ep, mu, angles, K, extent, vmin, vmax, figsize, dpi):
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    m = (mu + np.pi) % TWO_PI - np.pi
    c = np.mod(angles[:, 0], TWO_PI)
    ax[0].scatter(m[:, 0], m[:, 1], s=3, c=c, cmap="twilight", vmin=0, vmax=TWO_PI, linewidths=0)
    ax[0].set_xlim(-np.pi, np.pi); ax[0].set_ylim(-np.pi, np.pi)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel(r"$\theta_1$"); ax[0].set_ylabel(r"$\theta_2$")
    ax[0].set_title("latent codes")

    im = ax[1].imshow(np.log10(np.abs(K) + 1e-30), origin="lower", extent=extent,
                      cmap="inferno", vmin=vmin, vmax=vmax, aspect="equal")
    ax[1].set_xlabel(r"$\theta_1$"); ax[1].set_ylabel(r"$\theta_2$")
    ax[1].set_title(r"$\log_{10}|K|$")
    fig.colorbar(im, ax=ax[1], fraction=0.046)

    fig.suptitle(f"epoch {ep}")
    fig.tight_layout()
    return fig


def _render_sphere_frame(ep, mu, angles, K, theta, phi, vmin, vmax, figsize, dpi):
    fig = plt.figure(figsize=figsize, dpi=dpi)

    lon = (mu[:, 1] + np.pi) % TWO_PI - np.pi
    lat = np.pi / 2 - mu[:, 0]
    c = np.mod(angles[:, 1], TWO_PI)
    ax0 = fig.add_subplot(1, 2, 1, projection="mollweide")
    ax0.scatter(lon, lat, s=3, c=c, cmap="twilight", vmin=0, vmax=TWO_PI, linewidths=0)
    ax0.grid(True, linewidth=0.3, color="0.7")
    ax0.set_xticklabels([])
    ax0.set_title("latent codes")

    LON, LAT = np.meshgrid(phi, np.pi / 2 - theta)
    ax1 = fig.add_subplot(1, 2, 2, projection="mollweide")
    m = ax1.pcolormesh(LON, LAT, np.log10(np.abs(K) + 1e-30), cmap="inferno",
                       vmin=vmin, vmax=vmax, shading="auto")
    ax1.grid(True, linewidth=0.3, color="0.7")
    ax1.set_xticklabels([])
    ax1.set_title(r"$\log_{10}|K|$")
    fig.colorbar(m, ax=ax1, fraction=0.03, pad=0.04)

    fig.suptitle(f"epoch {ep}")
    fig.tight_layout()
    return fig


def _render_euclidean_frame(ep, mu, angles, K, extent, vmin, vmax, figsize, dpi, cyclic0):
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    if cyclic0:
        c, cmap, cmin, cmax = np.mod(angles[:, 0], TWO_PI), "twilight", 0.0, TWO_PI
    else:
        c, cmap, cmin, cmax = angles[:, 0], "viridis", None, None
    ax[0].scatter(mu[:, 0], mu[:, 1], s=3, c=c, cmap=cmap, vmin=cmin, vmax=cmax, linewidths=0)
    ax[0].set_xlim(extent[0], extent[1]); ax[0].set_ylim(extent[2], extent[3])
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("$z_1$"); ax[0].set_ylabel("$z_2$")
    ax[0].set_title("latent codes")

    im = ax[1].imshow(np.log10(np.abs(K) + 1e-30), origin="lower", extent=extent,
                      cmap="inferno", vmin=vmin, vmax=vmax, aspect="equal")
    ax[1].set_xlabel("$z_1$"); ax[1].set_ylabel("$z_2$")
    ax[1].set_title(r"$\log_{10}|K|$")
    fig.colorbar(im, ax=ax[1], fraction=0.046)

    fig.suptitle(f"epoch {ep}")
    fig.tight_layout()
    return fig


def render_evolution_video(run_dir, out="evolution.gif", fps=8, dpi=100,
                           figsize=(10, 4.5), color_percentiles=(1, 99)):
    """Read `arrays/evolution.npz` (from `compute_evolution`) and render an
    animated GIF: latent codes and |K| side by side, one frame per saved
    checkpoint, epoch progressing left to right in time.

    The colour scale for |K| is fixed across ALL frames (from
    `color_percentiles` of the whole stack, robust to a handful of early,
    wildly large curvature values right after initialization) so that a
    darker/brighter frame later in the GIF means curvature actually changed,
    not that matplotlib re-scaled the colormap.
    """
    npz_path = os.path.join(run_dir, "arrays", "evolution.npz")
    data = np.load(npz_path, allow_pickle=False)
    epochs, K, mu, angles = data["epochs"], data["K"], data["mu"], data["angles"]
    manifold = str(data["manifold"])

    logK = np.log10(np.abs(K) + 1e-30)
    vmin, vmax = np.percentile(logK, color_percentiles[0]), np.percentile(logK, color_percentiles[1])

    frame_dir = os.path.join(run_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []

    matplotlib.use("Agg")
    for i, ep in enumerate(epochs):
        if manifold == "torus":
            t1 = data["t1"]
            extent = [t1.min(), t1.max(), t1.min(), t1.max()]
            fig = _render_torus_frame(ep, mu[i], angles, K[i], extent, vmin, vmax, figsize, dpi)
        elif manifold == "sphere":
            fig = _render_sphere_frame(ep, mu[i], angles, K[i], data["theta"], data["phi"],
                                       vmin, vmax, figsize, dpi)
        else:
            fig = _render_euclidean_frame(ep, mu[i], angles, K[i], data["extent"],
                                          vmin, vmax, figsize, dpi, bool(data["cyclic0"]))
        fp = os.path.join(frame_dir, f"frame_{i:05d}.png")
        fig.savefig(fp)
        plt.close(fig)
        frame_paths.append(fp)

    frames = [Image.open(p) for p in frame_paths]
    out_path = os.path.join(run_dir, out)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    for f in frames:
        f.close()
    print(f"wrote {len(frame_paths)} frames to {frame_dir}")
    print("wrote", out_path)
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="a run folder containing checkpoints/ and meta.json")
    p.add_argument("--grid", type=int, default=40,
                   help="curvature grid (grid x grid); torus and flat/euclidean runs")
    p.add_argument("--n-theta", type=int, default=32)
    p.add_argument("--n-phi", type=int, default=64)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--every", type=int, default=1, help="use every Nth checkpoint")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", default="evolution.gif")
    p.add_argument("--skip-compute", action="store_true",
                   help="reuse an existing arrays/evolution.npz instead of recomputing")
    args = p.parse_args()

    if not args.skip_compute:
        compute_evolution(args.run_dir, grid=args.grid, n_theta=args.n_theta,
                          n_phi=args.n_phi, chunk=args.chunk, device=args.device,
                          every=args.every)
    render_evolution_video(args.run_dir, out=args.out, fps=args.fps)


if __name__ == "__main__":
    main()
