"""Train and evaluate all six (dataset topology x latent topology)
combinations used in the thesis, with full history/checkpoint persistence.

    python -m vae_topology.experiments.run_all_topologies
    python -m vae_topology.experiments.run_all_topologies --only torus_torus
    python -m vae_topology.experiments.run_all_topologies --epochs 20 --show

Structure mirrors `Colab_run_files/topo_vae_run_analyse.ipynb` cell-for-cell
(config -> helpers -> a `run_torus`/`run_sphere` driver function -> one call
per experiment), since that notebook is what actually produced the thesis's
numbers on a Colab GPU and is the closest thing to a specification of "how
this is supposed to be run" -- see that notebook's markdown cells for the
GPU-specific reasoning (CHUNK=1024 to avoid kernel-launch-overhead-bound
vmap on a GPU; float64 curvature pass being CPU-competitive even on a T4;
`plt.close('all')` between runs so six sequential trainings do not leak
matplotlib figures or GPU memory).

What this version adds on top of the Colab script: every run goes through a
`RunStore` (see `run_store.py`), which persists the trained model, its full
per-epoch training history, periodic checkpoints (so you can regenerate a
plot at an earlier epoch without retraining), and every figure/array -- the
Colab script only ever wrote `arrays_<key>.npz` (K, V, mu) and a curve CSV,
with no way to reconstruct anything else later. It also merges in the
validation-curve / worst-point diagnostics from the same notebook (absent
from the original `structure/sandbox_log.py` this script otherwise
descends from), generalized in `experiments/validation_curves.py`.

Every experiment run in ONE execution of this script (all six, or a
`--only` subset) nests under a single timestamped batch folder --
`results/topologies__<timestamp>/<key>/` -- rather than each experiment
getting its own independent timestamp; see `run_store.new_batch_dir`.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime

import numpy as np
import torch

# Headless unless --show. Must precede the pyplot import: six blocking
# plt.show() calls in a multi-thousand-epoch sweep leaves the machine idle
# waiting for you to close a window.
import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vae_topology.datasets import (          # noqa: E402
    make_plane_dataset, make_plane_dataset_old, make_torus_dataset,
    make_sphere_dataset_batched,
)
from vae_topology.models.torus_vae import TorusVAE, train as train_torus       # noqa: E402
from vae_topology.models.sphere_vae import SphereVAE, train as train_sphere    # noqa: E402
from vae_topology.diagnostics.latent_plots import diagnose, diagnose_sphere    # noqa: E402
from vae_topology.diagnostics.curvature_plots import (                        # noqa: E402
    plot_abs_curvature, plot_abs_curvature_sphere,
)
from vae_topology.geometry import fisher_metric_grid, metric_invariants       # noqa: E402
from vae_topology.experiments.run_store import RunStore, new_batch_dir        # noqa: E402
from vae_topology.experiments.validation_curves import (                      # noqa: E402
    make_hook, make_val, plot_curve, plot_worst_points,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(HERE, "..", "..", "results")

# ---------------------------------------------------------------------------
# the six experiments, as data rather than six copies of the same block.
# "key" matches the figure/checkpoint naming used throughout: latent_<key>.*,
# curvature_<key>.*, curve_<key>.*, worst_<key>.*
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    dict(key="border_flat_torus",  latent="torus",  data_fn=make_plane_dataset,
         label="bordered flat plane -> T2"),
    dict(key="flat_torus",         latent="torus",  data_fn=make_plane_dataset_old,
         label="periodic flat plane -> T2"),
    dict(key="torus_torus",        latent="torus",  data_fn=make_torus_dataset,
         label="torus -> T2"),
    dict(key="border_flat_sphere", latent="sphere", data_fn=make_plane_dataset,
         label="bordered flat plane -> S2"),
    dict(key="flat_sphere",        latent="sphere", data_fn=make_plane_dataset_old,
         label="periodic flat plane -> S2"),
    dict(key="sphere_sphere",      latent="sphere", data_fn=make_sphere_dataset_batched,
         label="sphere -> S2"),
]


def _batch_dir(args):
    """The folder every experiment from THIS execution nests inside (see
    `run_store.new_batch_dir`). Callers that loop over EXPERIMENTS (`main`,
    a notebook cell) should set `args.batch_dir` once before the loop; a
    one-off direct call to `run_torus`/`run_sphere` without it still works,
    getting its own single-experiment batch folder rather than erroring."""
    bd = getattr(args, "batch_dir", None)
    if bd is None:
        bd = new_batch_dir(RESULTS_ROOT, label="topologies")
    return bd


def _free(device):
    import gc
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _history_arrays(history):
    if history is None:
        return {}
    if isinstance(history, dict):
        return {k: np.asarray(v) for k, v in history.items()}
    if isinstance(history, (list, tuple)) and history and isinstance(history[0], dict):
        return {k: np.asarray([h[k] for h in history]) for k in history[0]}
    return {"loss": np.asarray(history)}


def _close_all():
    if plt.isinteractive():
        return
    plt.close("all")


@torch.no_grad()
def _encode_mu(model, data, batch=4096):
    model.eval()
    return torch.cat([model.encode(data[i:i + batch])[0]
                      for i in range(0, data.shape[0], batch)], dim=0).cpu()


def run_torus(cfg, args):
    key, data_fn, label = cfg["key"], cfg["data_fn"], cfg["label"]
    store = RunStore(root=_batch_dir(args), name=key, timestamp=False)
    print(f"\n=== {key}  ({label}) -> {store.path}")
    store.meta(experiment=key, description=label, latent_manifold="torus",
               dataset=data_fn.__name__, model_class=TorusVAE.__name__,
               epochs=args.epochs, seed=args.seed, grid=args.grid,
               started=datetime.now().isoformat(timespec="seconds"))

    val_x, val_ang = make_val(data_fn, device=args.device)
    train_x_raw, _ = data_fn()
    train_x = torch.as_tensor(train_x_raw, dtype=torch.float32, device=args.device)

    # checkpoint/eval cadence: --checkpoint-every overrides the default of
    # ~50 evenly-spaced snapshots across the run. A smaller value gives more
    # (denser) checkpoints -- e.g. for rendering a training-evolution video
    # with `experiments.render_evolution` -- at the cost of more disk space
    # and a bit more time spent computing the validation curve.
    checkpoint_every = getattr(args, "checkpoint_every", 0) or max(1, args.epochs // 50)

    curve = []
    hook = make_hook(curve, train_x, val_x, store=store)
    model, data, angles, history = train_torus(
        epochs=args.epochs, data_fn=data_fn, model_cls=TorusVAE, device=args.device,
        seed=args.seed, eval_hook=hook, eval_every=checkpoint_every)

    store.save_model(model)
    store.meta(data_dim=int(data.shape[1]))
    store.save_arrays("history", **_history_arrays(history))

    latent_fig = store.figure_path(f"latent_{key}")
    diagnose(model, data, angles, save=str(latent_fig))
    _close_all()

    plot_curve(curve, save=store.figure_path(f"curve_{key}"))
    plot_worst_points(model, "torus", train_x, angles, val_x, val_ang,
                      save=store.figure_path(f"worst_{key}"))
    _close_all()

    model = model.double()
    mu = _encode_mu(model, data.double())
    curv_fig = store.figure_path(f"curvature_{key}")
    fig, K, V = plot_abs_curvature(model, grid=args.grid, chunk=args.chunk,
                                   device=args.device, mu=mu, fname=str(curv_fig),
                                   save_clean=True,
                                   log_path=os.path.join(store.path, "log.txt"),
                                   log_tag=key)
    _close_all()

    arrays = dict(K=K, V=V, mu=mu)
    if args.metric:
        try:
            G, t1, t2 = fisher_metric_grid(model.decode, grid=args.grid,
                                           device=args.device, dtype=torch.float64)
            inv = metric_invariants(G)
            arrays.update(G=G, t1=t1, t2=t2, **inv)
            v = np.asarray(V)
            rel = np.abs(inv["sqrt_det"] - v) / np.clip(np.abs(v), 1e-30, None)
            msg = f"sqrt(det G) vs V : max relative deviation {rel.max():.2e}"
            print("   " + msg)
            store.log([msg], header="metric cross-check")
        except Exception as exc:                      # noqa: BLE001
            print(f"   [metric skipped: {type(exc).__name__}: {exc}]")
            store.log([f"fisher_metric_grid failed: {exc!r}"], header="WARNING")
    store.save_arrays("curvature_torus", **arrays)

    store.meta(finished=datetime.now().isoformat(timespec="seconds"), status="ok")
    _free(args.device)
    print(f"    {key} done -> {store.path}")
    return store.path


def run_sphere(cfg, args):
    key, data_fn, label = cfg["key"], cfg["data_fn"], cfg["label"]
    store = RunStore(root=_batch_dir(args), name=key, timestamp=False)
    print(f"\n=== {key}  ({label}) -> {store.path}")
    store.meta(experiment=key, description=label, latent_manifold="sphere",
               dataset=data_fn.__name__, model_class=SphereVAE.__name__,
               epochs=args.epochs, seed=args.seed,
               n_theta=args.n_theta, n_phi=args.n_phi,
               started=datetime.now().isoformat(timespec="seconds"))

    val_x, val_ang = make_val(data_fn, device=args.device)
    train_x_raw, _ = data_fn()
    train_x = torch.as_tensor(train_x_raw, dtype=torch.float32, device=args.device)

    # see the note in run_torus on checkpoint_every.
    checkpoint_every = getattr(args, "checkpoint_every", 0) or max(1, args.epochs // 50)

    curve = []
    hook = make_hook(curve, train_x, val_x, store=store)
    model, data, angles, history = train_sphere(
        epochs=args.epochs, data_fn=data_fn, model_cls=SphereVAE, device=args.device,
        seed=args.seed, eval_hook=hook, eval_every=checkpoint_every)

    store.save_model(model)
    store.meta(data_dim=int(data.shape[1]))
    store.save_arrays("history", **_history_arrays(history))

    latent_fig = store.figure_path(f"latent_{key}")
    diagnose_sphere(model, data, angles, save=str(latent_fig))
    _close_all()

    plot_curve(curve, save=store.figure_path(f"curve_{key}"))
    plot_worst_points(model, "sphere", train_x, angles, val_x, val_ang,
                      cyclic=False, save=store.figure_path(f"worst_{key}"))
    _close_all()

    model = model.double()
    mu = _encode_mu(model, data.double())
    curv_fig = store.figure_path(f"curvature_{key}")
    fig, K, V = plot_abs_curvature_sphere(
        model, n_theta=args.n_theta, n_phi=args.n_phi, chunk=args.chunk,
        device=args.device, mu=mu, fname=str(curv_fig), save_clean=True,
        log_path=os.path.join(store.path, "log.txt"), log_tag=key)
    _close_all()

    store.save_arrays("curvature_sphere", K=K, V=V, mu=mu)
    store.meta(finished=datetime.now().isoformat(timespec="seconds"), status="ok")
    _free(args.device)
    print(f"    {key} done -> {store.path}")
    return store.path


BACKENDS = {"torus": run_torus, "sphere": run_sphere}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--grid", type=int, default=100, help="torus curvature grid (grid x grid)")
    p.add_argument("--n-theta", type=int, default=64, help="sphere curvature grid, theta")
    p.add_argument("--n-phi", type=int, default=128, help="sphere curvature grid, phi")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chunk", type=int, default=None,
                   help="vmap chunk size for the curvature pass; default 1024 on "
                        "cuda (kernel-launch-overhead-bound below that), 32 on cpu")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="epochs between checkpoints; 0 = auto (~50 total). Set "
                        "smaller for a denser trajectory to feed "
                        "experiments.render_evolution")
    p.add_argument("--only", nargs="+", metavar="KEY", help="run only these experiments")
    p.add_argument("--show", action="store_true", help="open figure windows")
    p.add_argument("--no-metric", dest="metric", action="store_false",
                   help="skip the full 2x2 Fisher metric (torus only)")
    p.add_argument("--stop-on-error", action="store_true")
    args = p.parse_args()

    if args.chunk is None:
        args.chunk = 1024 if args.device == "cuda" else 32

    # one batch folder for this whole invocation -- see new_batch_dir's
    # docstring. Every experiment below nests inside it by name.
    args.batch_dir = new_batch_dir(RESULTS_ROOT, label="topologies")
    print(f"batch folder: {args.batch_dir}")

    todo = EXPERIMENTS
    if args.only:
        known = {e["key"] for e in EXPERIMENTS}
        unknown = set(args.only) - known
        if unknown:
            p.error(f"unknown keys {sorted(unknown)}; choose from {sorted(known)}")
        todo = [e for e in EXPERIMENTS if e["key"] in args.only]

    print(f"device={args.device}  chunk={args.chunk}  epochs={args.epochs}")

    results = {}
    for cfg in todo:
        try:
            results[cfg["key"]] = ("ok", BACKENDS[cfg["latent"]](cfg, args))
        except Exception:                             # noqa: BLE001
            traceback.print_exc()
            results[cfg["key"]] = ("FAILED", "")
            if args.stop_on_error:
                break
        finally:
            if args.show:
                plt.show()
            plt.close("all")

    print("\n" + "=" * 70)
    for key, (status, path) in results.items():
        print(f"{status:>7}  {key:<20} {path}")
    return 1 if any(s != "ok" for s, _ in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
