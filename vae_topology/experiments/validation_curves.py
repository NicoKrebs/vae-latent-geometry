"""Held-out reconstruction error curves and worst-reconstructed-point plots.

Answers two questions a training-loss curve alone cannot: is the model
overfitting (train vs. held-out error), and WHERE in the latent/data space
does it fail worst (as opposed to just "how much", on average).

Generalized from the validation/diagnostic cell in
`Colab_run_files/topo_vae_run_analyse.ipynb`, which itself generalized the
original, sphere-only `projects/sandbox.py` to work for both the torus and
sphere models (`kind="torus"`/`"sphere"`, dispatching on how each model's
`encode` return value needs to be turned into 2d angles for plotting). The
Colab version threaded state through a notebook-global `train_x_of` dict
keyed by experiment name; here the same data is passed explicitly as a
function argument instead, since this module is imported rather than run
cell-by-cell.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from vae_topology.distributions import angles_from_vec

VAL_N, VAL_SEED, N_WORST = 2000, 1234, 500


def _np(a):
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


@torch.no_grad()
def _recon(model, x):
    """SphereVAE.encode returns a unit 3-vector and decode() wants 2 angles;
    decode_vec() is the matching pair. TorusVAE.encode returns angles, which
    decode() takes directly."""
    mu = model.encode(x)[0]
    return model.decode_vec(mu) if hasattr(model, "decode_vec") else model.decode(mu)


@torch.no_grad()
def per_sample_err(model, x, batch=4096):
    """Noise-free reconstruction error per sample: encode to mu, decode, sum
    the squared error over pixels. No posterior sampling."""
    model.eval()
    p = next(model.parameters())
    x = x.to(device=p.device, dtype=p.dtype)
    e = torch.cat([((x[i:i + batch] - _recon(model, x[i:i + batch])) ** 2).sum(1)
                   for i in range(0, x.shape[0], batch)])
    model.train()
    return _np(e)


def make_val(data_fn, device=None, n=VAL_N, seed=VAL_SEED):
    """Same generator, same settings, DIFFERENT seed -> points the model
    never trained on, drawn from the same manifold."""
    imgs, ang = data_fn(n=n, seed=seed)
    x = torch.as_tensor(imgs, dtype=torch.float32, device=device)
    if x.dim() > 2:
        x = x.flatten(1)
    return x, np.asarray(ang)


def make_hook(curve, train_x, val_x, store=None, checkpoint_extra=None):
    """Build an `eval_hook(model, epoch)` for `models.torus_vae.train` /
    `models.sphere_vae.train`.

    Always appends {epoch, train err, val err, val median err} to `curve`
    (mutated in place). If `store` is a `RunStore`, ALSO snapshots the model
    at this epoch via `store.save_checkpoint` -- this is what lets you
    regenerate a plot (a different latent-space view, a curvature map at an
    earlier/later epoch, ...) after the fact, without retraining. Pass
    `eval_every` to `train()` to control how often this fires; every epoch is
    usually far more often than needed and wastes disk space on checkpoints.
    """
    def hook(model, ep):
        e_tr = per_sample_err(model, train_x)
        e_va = per_sample_err(model, val_x)
        curve.append(dict(epoch=ep, train=e_tr.mean(), val=e_va.mean(),
                          val_median=np.median(e_va),
                          val_p95=np.percentile(e_va, 95)))
        if store is not None:
            store.save_checkpoint(model, ep, extra=checkpoint_extra)
    return hook


def plot_curve(curve, save, panel_labels=("a)", "b)"), label_size=14,
               show=False):
    """Train vs. held-out reconstruction error over training, plus the
    val/train and mean/median ratios (a widening train/val gap is
    overfitting; mean >> median means a few points dominate the average
    error -- see `plot_worst_points`)."""
    c = pd.DataFrame(curve)
    stem, ext = os.path.splitext(save)
    ext = ext or ".pdf"

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(c.epoch, c.train, color="0.3", label="train")
    ax[0].plot(c.epoch, c.val, color="crimson", label="validation mean")
    ax[0].plot(c.epoch, c.val_median, "--", color="crimson", alpha=0.6,
               label="validation median")
    if "val_p95" in c:
        ax[0].plot(c.epoch, c.val_p95, ":", color="crimson", alpha=0.5,
                  label="validation 95th pct.")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("reconstruction error")
    ax[0].legend(fontsize=10)

    ax[1].plot(c.epoch, c.val / c.train, color="0.3", label="val / train")
    ax[1].plot(c.epoch, c.val / c.val_median, color="steelblue",
               label="mean / median (validation)")
    ax[1].axhline(1.0, color="0.7", lw=1)
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("ratio")
    ax[1].legend(fontsize=10)

    for a, lab in zip(ax, panel_labels):
        a.text(-0.12, 1.02, lab, transform=a.transAxes, fontsize=label_size,
               fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    fig.savefig(f"{stem}_curve{ext}", dpi=150, bbox_inches="tight")
    c.to_csv(f"{stem}_curve.csv", index=False)
    if not show:
        plt.close(fig)

    last = c.iloc[-1]
    print(f"  epoch {int(last.epoch)}: train {last.train:.4f}  "
          f"held-out {last.val:.4f}  gap {100 * (last.val / last.train - 1):.1f}%  "
          f"mean/median {last.val / last.val_median:.2f}")
    return fig, c


def plot_worst_points(model, kind, train_x, train_ang, val_x, val_ang,
                      n_worst=N_WORST, n_points=4000, seed=0, s=4,
                      cyclic=True, save=None, panel_labels=("a)", "b)"),
                      label_size=14, show=False):
    """Two panels: a) latent codes coloured by the true first generative
    angle, with the worst-reconstructed held-out points marked in black;
    b) the same worst points in TRUE generative-coordinate space, to see
    whether failures cluster somewhere specific (e.g. near a pole, or a
    tear) rather than being spread uniformly (which would indicate a
    capacity/training issue instead of a geometric one).

    kind: "torus" -- codes are already 2 angles, axes fixed to [-pi, pi).
          "sphere" -- codes are unit 3-vectors, converted via
          `distributions.angles_from_vec` to (theta, phi) for plotting, and
          plotted with x in [0, pi] (polar) instead of [-pi, pi].
          anything else (e.g. "euclidean") -- codes are raw R^2 coordinates,
          axes auto-scaled to wherever they land, generic z1/z2 labels. (An
          earlier version of this function only branched on kind=="torus"
          and treated every other value as "sphere", which would have
          silently mislabeled a flat/Euclidean latent's plot with polar/
          azimuth axis limits that do not apply to it.)
    """
    e_va = per_sample_err(model, val_x)
    bad = np.argsort(e_va)[-n_worst:]

    with torch.no_grad():
        p = next(model.parameters())
        mu_tr_raw = model.encode(train_x.to(device=p.device, dtype=p.dtype))[0]
        mu_bad_raw = model.encode(val_x[bad].to(device=p.device, dtype=p.dtype))[0]
        if kind == "sphere":
            mu_tr_raw = angles_from_vec(mu_tr_raw)
            mu_bad_raw = angles_from_vec(mu_bad_raw)
    mu_tr, mu_bad = _np(mu_tr_raw), _np(mu_bad_raw)

    true = _np(train_ang)
    val_ang = np.asarray(val_ang)

    n = min(n_points, mu_tr.shape[0])
    idx = np.random.default_rng(seed).choice(mu_tr.shape[0], size=n, replace=False)
    mu_tr, true = mu_tr[idx], true[idx]

    if cyclic:
        c_bg, cmap, vmin, vmax = np.mod(true[:, 0], 2 * np.pi), "twilight", 0.0, 2 * np.pi
    else:
        c_bg, cmap, vmin, vmax = true[:, 0], "viridis", None, None

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    sc = ax[0].scatter(mu_tr[:, 0], mu_tr[:, 1], s=s, c=c_bg, cmap=cmap,
                       vmin=vmin, vmax=vmax, linewidths=0, alpha=0.75)
    ax[0].scatter(mu_bad[:, 0], mu_bad[:, 1], s=3 * s, c="black", linewidths=0,
                  label=f"worst {n_worst} validation")
    if kind == "torus":
        ax[0].set_xlim(-np.pi, np.pi); ax[0].set_ylim(-np.pi, np.pi)
        ax[0].set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax[0].set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax[0].set_xlabel(r"$\theta_1$"); ax[0].set_ylabel(r"$\theta_2$")
    elif kind == "sphere":
        ax[0].set_xlim(0, np.pi); ax[0].set_ylim(-np.pi, np.pi)
        ax[0].set_xlabel(r"$\theta$ (polar)"); ax[0].set_ylabel(r"$\phi$ (azimuth)")
    else:
        ax[0].set_xlabel(r"$z_1$"); ax[0].set_ylabel(r"$z_2$")
    ax[0].set_aspect("equal")
    ax[0].legend(fontsize=10, markerscale=3)
    ax[0].text(-0.12, 1.02, panel_labels[0], transform=ax[0].transAxes,
               fontsize=label_size, fontweight="bold", va="bottom", ha="left")
    fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.02, label=r"true $\gamma_1$")

    ax[1].scatter(val_ang[:, 0], val_ang[:, 1], s=2, c="0.85", linewidths=0)
    ax[1].scatter(val_ang[bad, 0], val_ang[bad, 1], s=3 * s, c="black", linewidths=0)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel(r"true $\gamma_1$")
    ax[1].set_ylabel(r"true $\gamma_2$")
    ax[1].text(-0.12, 1.02, panel_labels[1], transform=ax[1].transAxes,
               fontsize=label_size, fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    if save:
        stem, ext = os.path.splitext(save)
        ext = ext or ".pdf"
        fig.savefig(f"{stem}_worst{ext}", dpi=150, bbox_inches="tight")
    if not show:
        plt.close(fig)

    print(f"  worst {n_worst}: err {e_va[bad].min():.3f}..{e_va[bad].max():.3f}   "
          f"median all {np.median(e_va):.3f}   "
          f"they carry {100 * e_va[bad].sum() / e_va.sum():.1f}% of total error")
    return fig, bad, e_va
