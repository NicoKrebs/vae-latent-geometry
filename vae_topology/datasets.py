"""Synthetic image datasets with a known, controllable latent topology.

Each `make_*_dataset` function returns `(imgs, angles)`:
    imgs   : (n, res*res) float32 in [0, 1], flattened row-major.
    angles : (n, k) float32, the TRUE generative parameters used to render
             each image (angles in radians). These are only used for
             evaluation (colouring latent plots, computing winding numbers,
             ...) -- never fed to a model as input.

Every renderer places a Gaussian "bump" at a location determined by the
generative angle(s); the datasets differ in which manifold that location is
sampled from and how the bump is drawn on the pixel grid, not in the general
recipe.

Consolidated from the original `projects/datasets.py` and
`projects/bin_scripts/sphere_data.py`. Two functions from the original
`datasets.py` were dropped here because nothing in the codebase called them:
`sphere_render_old` (stereographic-projection renderer) and its dataset
`make_sphere_dataset` -- both superseded by `sphere_render`/
`make_sphere_dataset_new` below before this cleanup, but never actually
removed. If you need them back, they are still in the original
`projects/datasets.py`.
"""

import os
from functools import lru_cache

import numpy as np

TWO_PI = 2.0 * np.pi


# ============================================================== plane =====
#
# Two DELIBERATELY different flat-plane datasets are kept, because they probe
# different questions when paired with a periodic (torus/sphere) latent
# space -- see the "border_flat_*" vs. "flat_*" experiments in
# experiments/run_all_topologies.py:
#
#   make_plane_dataset_old  samples the bump centre with a margin that keeps
#       it away from the pixel-grid edge, so the RENDERED IMAGE never wraps
#       or gets clipped. This is a genuinely flat, non-periodic baseline.
#
#   make_plane_dataset      samples the generative angle over the full
#       periodic range [0, 2*pi) and maps it linearly onto pixel coordinates
#       with NO wraparound in the renderer. Near the edges the bump is cut
#       off, so images for angles close to 0 and close to 2*pi (adjacent on
#       the circle) look nothing alike in pixel space -- a hard seam. This is
#       the "does a periodic latent space help when the data itself has a
#       topology mismatch" condition.
#
# The `_old` suffix is a holdover from development history, not a claim that
# the function is deprecated -- both are actively used. Left unrenamed so the
# name matches every call site and notebook that already references it.
# ============================================================================

def plane_render(cx, cy, res=28, sigma=3.0):
    """Bump at pixel coords (cx, cy), NOT wrapped: clipped at the edges."""
    idx = np.arange(res)
    dx = idx[None, :] - (cx * res) / (2 * np.pi)
    dy = idx[:, None] - (cy * res) / (2 * np.pi)
    return np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2)).astype(np.float32)


def make_plane_dataset(n=10000, res=28, sigma=3.0, seed=0, lo=0.0, hi=TWO_PI):
    """"Bordered flat plane": angle sampled over the full period, but the
    renderer does not wrap, so the image has a seam at the border."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(lo, hi, size=(n, 2)).astype(np.float32)
    imgs = np.stack([plane_render(t1, t2, res, sigma).ravel() for t1, t2 in angles])
    return imgs.astype(np.float32), angles


def plane_render_old(cx, cy, res=28, sigma=3.0):
    """Bump at raw pixel coords (cx, cy); callers keep cx, cy away from the
    edge (see `make_plane_dataset_old`) so the bump is never clipped."""
    idx = np.arange(res)
    dx = idx[None, :] - cx
    dy = idx[:, None] - cy
    return np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2)).astype(np.float32)


def make_plane_dataset_old(n=10000, res=28, sigma=3.0, margin=-3.0, seed=0,
                           as_angles=True, angle_max=(TWO_PI, TWO_PI)):
    """"Periodic flat plane": bump centre kept >= 3 sigma from every edge, so
    the image never touches the border. Genuinely flat, non-periodic data;
    `as_angles` just rescales the centre into an angle-shaped label for
    plotting/consistency with the other datasets, it does not make the data
    periodic."""
    lo = -margin * sigma
    hi = res - 1 + margin * sigma
    if hi <= lo:
        lo = hi = (res - 1) / 2.0
    rng = np.random.default_rng(seed)
    centres = rng.uniform(lo, hi, size=(n, 2)).astype(np.float32)
    imgs = np.stack([plane_render_old(a, b, res, sigma).ravel() for a, b in centres])

    labels = centres
    if as_angles:
        span = hi - lo
        if span <= 0:
            labels = np.zeros_like(centres)
        else:
            am = np.asarray(angle_max, dtype=np.float32)
            labels = ((centres - lo) / span * am).astype(np.float32)
    return imgs.astype(np.float32), labels


# ============================================================== torus =====

def torus_render(theta1, theta2, res=28, sigma=3.0):
    """Bump on a (res, res) pixel grid with periodic (wraparound) distance,
    i.e. the image is genuinely a discretised torus: pixel column/row 0 is
    adjacent to column/row res-1."""
    cx = (res * theta1 / (2 * np.pi)) % res
    cy = (res * theta2 / (2 * np.pi)) % res
    idx = np.arange(res)
    dx = np.abs(idx[None, :] - cx); dx = np.minimum(dx, res - dx)
    dy = np.abs(idx[:, None] - cy); dy = np.minimum(dy, res - dy)
    return np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2)).astype(np.float32)


def make_torus_dataset(n=10000, res=28, sigma=3.0, seed=0, lo=0.0, hi=TWO_PI):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(lo, hi, size=(n, 2)).astype(np.float32)
    imgs = np.stack([torus_render(t1, t2, res, sigma).ravel() for t1, t2 in angles])
    return imgs.astype(np.float32), angles


# ============================================================= sphere =====
#
# Two sphere renderers/datasets are kept, for different roles:
#
#   sphere_render / make_sphere_dataset_new   the SphereVAE's own default
#       training set (models/sphere_vae.py); a simple per-sample loop, bump
#       width `sigma` given directly in radians of geodesic distance.
#
#   sphere_render_batched / make_sphere_dataset_batched   used by
#       experiments/run_all_topologies.py for the "sphere -> S2" sweep; a
#       vectorised (batched) renderer for speed at larger n, with `sigma`
#       given in the same "pixel width" convention as torus_render/
#       plane_render (sigma=3.0 default), converted internally to radians via
#       sigma * pi / res. The two are NOT interchangeable defaults -- sigma=3.0
#       here means something different (in radians) than sigma=3.0 passed to
#       `sphere_render`.
#
# (Originally `bin_scripts/sphere_data.py::sphere_render` /
# `make_sphere_dataset_try3`.)
# ============================================================================

@lru_cache(maxsize=None)
def _sphere_grid(H, W):
    """Unit vectors of an equirectangular (lon/lat) pixel grid, (H, W, 3)."""
    lon = -np.pi + (np.arange(W) + 0.5) * (2 * np.pi / W)
    lat = np.pi / 2 - (np.arange(H) + 0.5) * (np.pi / H)
    LON, LAT = np.meshgrid(lon, lat)
    return np.stack([np.cos(LAT) * np.cos(LON),
                     np.cos(LAT) * np.sin(LON),
                     np.sin(LAT)], axis=-1)


def sphere_render(theta, phi, res=28, sigma=0.35):
    """Square equirectangular grid: (res, res) image. `sigma` is a geodesic
    angle in radians."""
    n = np.array([np.sin(theta) * np.cos(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(theta)])
    d = np.arccos(np.clip(_sphere_grid(res, res) @ n, -1.0, 1.0))
    return np.exp(-d ** 2 / (2 * sigma ** 2)).astype(np.float32)


def make_sphere_dataset_new(n=10000, res=28, sigma=0.35, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.arccos(1 - 2 * rng.uniform(size=n))    # uniform on S^2
    phi = rng.uniform(0, 2 * np.pi, size=n)
    angles = np.stack([theta, phi], axis=1).astype(np.float32)
    imgs = np.stack([sphere_render(t, p, res, sigma).ravel() for t, p in angles])
    return imgs.astype(np.float32), angles


@lru_cache(maxsize=None)
def _sphere_grid_colat(res, res_phi):
    """Unit vectors of an equirectangular pixel grid, shape (res, res_phi, 3).

    Rows are colatitude theta in (0, pi), columns azimuth phi in (0, 2pi).
    Pixel centres are used, so no row sits exactly on a pole. Distinct cache
    from `_sphere_grid` above: different (colatitude, azimuth) parametrization
    and independent row/column resolution.
    """
    theta = (np.arange(res) + 0.5) * np.pi / res
    phi = (np.arange(res_phi) + 0.5) * TWO_PI / res_phi
    st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
    sp, cp = np.sin(phi)[None, :], np.cos(phi)[None, :]
    return np.stack(
        [st * cp, st * sp, np.broadcast_to(ct, (res, res_phi))], axis=-1
    )


def _unit(theta, phi):
    """(theta, phi) -> unit vector(s) in R^3. Broadcasts over arrays."""
    theta, phi = np.asarray(theta), np.asarray(phi)
    st, ct = np.sin(theta), np.cos(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=-1)


def sphere_render_px(theta, phi, res=28, sigma=3.0, res_phi=None):
    """Single-sample render matching `sphere_render_batch`'s conventions
    (sigma in pixel units, converted to radians as sigma * pi / res).
    Originally also named `sphere_render`, in `bin_scripts/sphere_data.py`;
    renamed here only to avoid clashing with the radian-sigma `sphere_render`
    above now that both live in one module. Not called by any dataset
    constructor (`sphere_render_batch` builds the grid directly) -- kept
    because `diagnostics`/older notebooks import it by name."""
    res_phi = res if res_phi is None else res_phi
    grid = _sphere_grid_colat(res, res_phi)
    c = _unit(theta, phi)
    cosd = np.clip(grid @ c, -1.0, 1.0)
    d = np.arccos(cosd)
    sigma_rad = sigma * np.pi / res
    return np.exp(-d ** 2 / (2 * sigma_rad ** 2)).astype(np.float32)


def sphere_render_batch(angles, res=28, sigma=3.0, res_phi=None):
    """Vectorised renderer: (n, 2) angles -> (n, res * res_phi) flat images."""
    res_phi = res if res_phi is None else res_phi
    angles = np.atleast_2d(angles)
    dirs = _sphere_grid_colat(res, res_phi).reshape(-1, 3)       # (P, 3)
    c = _unit(angles[:, 0], angles[:, 1])                        # (n, 3)
    d = np.arccos(np.clip(c @ dirs.T, -1.0, 1.0))                # (n, P)
    sigma_rad = sigma * np.pi / res
    return np.exp(-d ** 2 / (2 * sigma_rad ** 2)).astype(np.float32)


def _sample_angles_sphere(n, rng, theta_max=np.pi, lo=0.0, hi=TWO_PI):
    u = rng.uniform(np.cos(theta_max), 1.0, n)
    theta = np.arccos(u)
    phi = rng.uniform(lo, hi, n)
    return np.stack([theta, phi], axis=1)


def make_sphere_dataset_batched(n=10000, res=28, sigma=3.0, seed=0,
                                theta_max=np.pi, lo=0.0, hi=TWO_PI, res_phi=None):
    """Return (imgs, angles) with imgs (n, res*res_phi) float32, angles (n, 2).

    Originally `make_sphere_dataset_try3`.
    """
    rng = np.random.default_rng(seed)
    angles = _sample_angles_sphere(n, rng, theta_max, lo, hi)
    imgs = sphere_render_batch(angles, res, sigma, res_phi)
    return imgs.astype(np.float32), angles


# ========================================================== galaxy mnist ===
#
# GalaxyMNIST (Walmsley et al.): ~10,000 REAL DECaLS galaxy photographs,
# labelled by Galaxy Zoo volunteers into 4 morphology classes. Every other
# dataset above is synthetic, with a KNOWN generative angle used only for
# evaluation; here there is no generative angle -- the second element of
# make_galaxy_mnist_dataset's return is a discrete class label (0..3, see
# GALAXY_MNIST_CLASSES), carried through the same "angles" slot purely so
# every existing diagnostic (run_diagnostics, run_diagnostics_3d,
# plot_sphere_latent, ...) can still colour its scatter by it. That slot
# being a true continuous angle for every other dataset, but a discrete
# label here, is exactly why the galaxy-mnist notebook ALSO builds its own
# qualitative, legend-based class-colour plots alongside those -- a
# continuous colormap over 4 integers is not "appropriate colours for
# classes".
#
# The `galaxy_mnist` package's own download links (a pair of Dropbox-hosted
# .hdf5.gz files) are dead as of this writing (404). This instead uses the
# resources served by the same author's later `galaxy-datasets` package
# (github.com/mwalmsley/pytorch-galaxy-datasets, `shared/galaxy_mnist.py`):
# two Dropbox parquet catalogues (train/test, canonical 80/20 split, exactly
# balanced across the 4 classes) plus one Dropbox jpeg tarball shared by
# every split. Taking only the 3 URLs (verified reachable) rather than that
# whole package as a dependency avoids pulling in pytorch-lightning and the
# HuggingFace `datasets` library just to read 3 files off Dropbox.
# ==============================================================================

GALAXY_MNIST_CLASSES = ["smooth_round", "smooth_cigar", "edge_on_disk", "unbarred_spiral"]

_GALAXY_MNIST_RESOURCES = dict(
    train_catalog=(
        "https://dl.dropboxusercontent.com/s/6xlym0ney5q9aec14vvy1/"
        "galaxy_mnist_train_catalog.parquet?rlkey=fc5knqscwk2h6r4z5d3156vzr&dl=0",
        "galaxy_mnist_train_catalog.parquet",
        "cd22b21d165802f4bc1adf997424aec2",
    ),
    test_catalog=(
        "https://dl.dropboxusercontent.com/s/9assxy247i1nq8wjy7o0a/"
        "galaxy_mnist_test_catalog.parquet?rlkey=y1ns24xw0tcyntodi61rv14fu&dl=0",
        "galaxy_mnist_test_catalog.parquet",
        "c5cca8d8afb6fb0d59baeb310a35d594",
    ),
    images=(
        "https://dl.dropboxusercontent.com/s/lj89307kjx5plme9f66js/"
        "galaxy_mnist_images.tar.gz?rlkey=gvlwx2rl3hqpo3gplb9zlqrz0",
        "galaxy_mnist_images.tar.gz",
        "1c6cb0447f2f7ed676c3363ee194ced9",
    ),
)


def _galaxy_mnist_root(root=None):
    return root or os.path.join(os.path.expanduser("~"), ".cache", "galaxy_mnist")


@lru_cache(maxsize=4)
def download_galaxy_mnist(root=None):
    """Fetch+extract the 2 catalogues and the (shared) image tarball into
    `root`, skipping anything already present. Returns the resolved root.

    Guards each resource with its own existence check BEFORE calling
    torchvision's download helpers, rather than relying on those helpers'
    own md5 check alone: `download_and_extract_archive` re-extracts the
    ~10,000-file tarball unconditionally every time it is called, even when
    the archive itself is already present and valid, which made every
    repeat call to `make_galaxy_mnist_dataset` (one per train/val split,
    once per cell re-run in a notebook) cost upwards of a minute for no
    reason. `lru_cache` on this function additionally makes repeat calls in
    the same process free.
    """
    from torchvision.datasets.utils import download_and_extract_archive, download_url

    root = _galaxy_mnist_root(root)
    os.makedirs(root, exist_ok=True)
    for key in ("train_catalog", "test_catalog"):
        url, filename, md5 = _GALAXY_MNIST_RESOURCES[key]
        if not os.path.exists(os.path.join(root, filename)):
            download_url(url, root=root, filename=filename, md5=md5)
    image_dir = os.path.join(root, "images")
    if not (os.path.isdir(image_dir) and len(os.listdir(image_dir)) >= 9999):
        url, filename, md5 = _GALAXY_MNIST_RESOURCES["images"]
        download_and_extract_archive(url, download_root=root, filename=filename, md5=md5)
    return root


@lru_cache(maxsize=4)
def _galaxy_mnist_catalog(root, split):
    import pandas as pd
    name = "train_catalog" if split == "train" else "test_catalog"
    _, filename, _ = _GALAXY_MNIST_RESOURCES[name]
    catalog = pd.read_parquet(os.path.join(root, filename))
    return catalog[["filename", "label"]].reset_index(drop=True)


def make_galaxy_mnist_dataset(n=None, seed=0, split="train", res=28,
                              grayscale=True, root=None):
    """Real galaxy images -- see the module note above.

    `split` selects GalaxyMNIST's own canonical train/test division (a real
    80/20 split of real images, not a reseeded redraw -- unlike every
    synthetic dataset above, a real dataset needs an ACTUAL held-out split
    to evaluate validation error without leakage; use split="train" for
    training, split="test" for validation). `n`/`seed` subsample WITHIN that
    split, for signature parity with every make_*_dataset above, so this
    still works as a drop-in `dataset`/`data_fn` for `models.*.train` and
    `experiments.validation_curves.make_val`.

    `res`/`grayscale` default to 28/True (matching the data_dim=784
    convention every other dataset and every diagnostic default -- e.g.
    `diagnostics.latent_plots.plot_reconstructions`'s `res=28` -- already
    assumes) so the existing dense Tanh encoder/decoder and diagnostics work
    completely unmodified. The source images are 224x224 RGB regardless;
    this downsamples with PIL.

    Returns (imgs, labels): imgs (n, res*res) [or (n, res*res*3) if
    grayscale=False] float32 in [0, 1]; labels (n, 1) float32 holding the
    class index 0..3 (see GALAXY_MNIST_CLASSES) -- NOT a generative angle.
    """
    from PIL import Image

    root = download_galaxy_mnist(root)
    catalog = _galaxy_mnist_catalog(root, split)
    image_dir = os.path.join(root, "images")

    idx = np.arange(len(catalog))
    if n is not None and n < len(idx):
        idx = np.random.default_rng(seed).choice(idx, size=n, replace=False)
    sub = catalog.iloc[idx]

    mode = "L" if grayscale else "RGB"
    shape = (len(sub), res, res) if grayscale else (len(sub), res, res, 3)
    arrs = np.empty(shape, dtype=np.float32)
    for i, fname in enumerate(sub["filename"]):
        with Image.open(os.path.join(image_dir, fname)) as im:
            im = im.convert(mode).resize((res, res), Image.BILINEAR)
            arrs[i] = np.asarray(im, dtype=np.float32) / 255.0

    imgs = arrs.reshape(len(sub), -1)
    labels = sub["label"].to_numpy(dtype=np.float32).reshape(-1, 1)
    return imgs, labels
