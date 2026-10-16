"""Train the standard Euclidean-latent VAE (`models.euclidean_vae`) with a
2d latent on all four datasets, and look at the pullback curvature this
induces -- the point of comparison the torus/sphere models in
`run_all_topologies.py` are meant to improve on.

    python -m vae_topology.experiments.run_euclidean_2d
    python -m vae_topology.experiments.run_euclidean_2d --only sphere_2d
    python -m vae_topology.experiments.run_euclidean_2d --epochs 200 --show

Only three of these four datasets were actually run with a 2d Euclidean
latent in the original project (`architecture/standardvae.py`'s __main__:
bordered plane, periodic plane, torus -- never the sphere). `sphere_2d` is
added here for completeness and as a deliberate NEGATIVE baseline: a flat
R^2 latent cannot represent S^2 without tearing, so this run is EXPECTED to
show a seam in the latent scatter and a curvature map that does not close
via Gauss-Bonnet the way `run_all_topologies.py`'s `sphere_sphere` run does
-- that gap is the thesis's actual result, not a bug in this script.

Persists through `RunStore` exactly like `run_all_topologies.py` /
`run_euclidean_3d.py`: full history, periodic checkpoints, every figure and
array, so any plot can be regenerated later without retraining (see
`run_store.py`'s docstring).

Also produces the held-out validation curve and worst-reconstructed-point
plot (`experiments.validation_curves`), matching `run_all_topologies.py` --
confirmed as the actually-intended workflow for this model too by
`Colab_run_files/flat_vae_run_analyse (2).ipynb`, whose own `train()`
tracked a validation curve inline and which called a near-identical
`plot_worst_points` after every run. That notebook's version was
sphere/torus-unaware (it predates `experiments.validation_curves`'s
generalization) and had a few things this file intentionally does NOT carry
over: a Gauss-Bonnet-only fast path (`run_gb`/`gb()` -- redundant with the
`--gauss-bonnet` flag already here, which does the same computation as part
of the bundled diagnostics), a separate error-histogram report (`report()`
-- not needed for this project's actual goal of curvature/topology
analysis), and one clearly exploratory scratch cell (mistyped dataset label
"mall"). It also revealed a real bug, fixed here rather than copied: its
`run()` had to manually move the model and data back to CPU before calling
`run_diagnostics`, because the CPU-only curvature pass crashed otherwise on
a GPU-trained model -- fixed at the source in `diagnostics/latent_plots.py`
and `geometry.py` instead of requiring every caller to remember it.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import torch

import matplotlib
if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vae_topology.datasets import (          # noqa: E402
    make_plane_dataset, make_plane_dataset_old, make_torus_dataset,
    make_sphere_dataset_batched,
)
from vae_topology.models.euclidean_vae import SimpleVAE, train  # noqa: E402
from vae_topology.diagnostics.latent_plots import run_diagnostics  # noqa: E402
from vae_topology.experiments.run_store import RunStore, new_batch_dir  # noqa: E402
from vae_topology.experiments.validation_curves import (      # noqa: E402
    make_hook, make_val, plot_curve, plot_worst_points,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(HERE, "..", "..", "results")

EXPERIMENTS = [
    dict(key="flat_2d_no_border", dataset=make_plane_dataset_old,
         cyclic=(True, True), label="flat plane, no border contact"),
    dict(key="flat_2d_border",    dataset=make_plane_dataset,
         cyclic=(True, True), label="flat plane, border contact"),
    dict(key="torus_2d",          dataset=make_torus_dataset,
         cyclic=(True, True), label="torus"),
    dict(key="sphere_2d",         dataset=make_sphere_dataset_batched,
         cyclic=(False, True), label="sphere (negative baseline)"),
]


def _history_arrays(history):
    if not history:
        return {}
    return {k: np.asarray([h[k] for h in history]) for k in history[0]}


def _batch_dir(args):
    """See run_store.new_batch_dir. Callers looping over EXPERIMENTS should
    set args.batch_dir once before the loop (main() does); a one-off direct
    call without it gets its own single-experiment batch folder instead."""
    bd = getattr(args, "batch_dir", None)
    if bd is None:
        bd = new_batch_dir(RESULTS_ROOT, label="euclidean2d")
    return bd


def run_one(cfg, args):
    key, label = cfg["key"], cfg["label"]
    store = RunStore(root=_batch_dir(args), name=key, timestamp=False)
    print(f"\n=== {key}  ({label}) -> {store.path}")
    store.meta(experiment=key, description=label, latent_dim=2,
               latent_manifold="euclidean", model_class=SimpleVAE.__name__,
               dataset=cfg["dataset"].__name__, cyclic=list(cfg["cyclic"]),
               epochs=args.epochs, seed=args.seed, grid=args.grid,
               started=datetime.now().isoformat(timespec="seconds"))

    # checkpoint/eval cadence: --checkpoint-every overrides the default of
    # ~50 evenly-spaced snapshots across the run, matching
    # run_all_topologies.py's convention (see the note there).
    checkpoint_every = args.checkpoint_every or max(1, args.epochs // 50)

    val_x, val_ang = make_val(cfg["dataset"], device=args.device)
    train_x_raw, _ = cfg["dataset"]()
    train_x = torch.as_tensor(train_x_raw, dtype=torch.float32, device=args.device)

    curve = []
    hook = make_hook(curve, train_x, val_x, store=store)

    t0 = time.time()
    model, data, angles, history = train(
        epochs=args.epochs, dataset=cfg["dataset"], latent_dim=2, seed=args.seed,
        device=args.device, log_every=args.log_every,
        eval_hook=hook, eval_every=checkpoint_every)
    t_train = time.time() - t0

    store.save_model(model)
    store.meta(data_dim=int(data.shape[1]))
    store.save_arrays("history", **_history_arrays(history))

    plot_curve(curve, save=store.figure_path(f"curve_{key}"))
    plot_worst_points(model, "euclidean", train_x, angles, val_x, val_ang,
                      cyclic=cfg["cyclic"][0], save=store.figure_path(f"worst_{key}"))
    plt.close("all")

    fig_lat, fig_curv, K, M, gb = run_diagnostics(
        model, data, angles, grid=args.grid, cyclic=cfg["cyclic"],
        fname=store.figure_path(f"vae_diagnostics_{key}"),
        gauss_bonnet=args.gauss_bonnet)
    store.save_arrays("curvature", K=K, M=M)
    if gb is not None:
        store.save_arrays("gauss_bonnet", **{k: v for k, v in gb.items()
                                             if not isinstance(v, (int, float, bool))})
        store.log([f"{k} = {v}" for k, v in gb.items()
                  if isinstance(v, (int, float, bool))],
                 header="gauss_bonnet_disk summary")

    plt.close("all")
    store.meta(finished=datetime.now().isoformat(timespec="seconds"), status="ok",
               train_seconds=round(t_train, 1))
    print(f"    {key} done  (train {t_train:.0f}s, total {time.time() - t0:.0f}s) -> {store.path}")

    del model, data
    gc.collect()
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return store.path


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--grid", type=int, default=100)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="epochs between checkpoints; 0 = auto (~50 total)")
    p.add_argument("--gauss-bonnet", action="store_true",
                   help="also run the Gauss-Bonnet-disk check on the encoded codes")
    p.add_argument("--only", nargs="+", metavar="KEY", help="run only these experiments")
    p.add_argument("--show", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    args = p.parse_args()

    args.batch_dir = new_batch_dir(RESULTS_ROOT, label="euclidean2d")
    print(f"batch folder: {args.batch_dir}")

    todo = EXPERIMENTS
    if args.only:
        known = {e["key"] for e in EXPERIMENTS}
        unknown = set(args.only) - known
        if unknown:
            p.error(f"unknown keys {sorted(unknown)}; choose from {sorted(known)}")
        todo = [e for e in EXPERIMENTS if e["key"] in args.only]

    print(f"device={args.device}  epochs={args.epochs}")

    results = {}
    for cfg in todo:
        try:
            results[cfg["key"]] = ("ok", run_one(cfg, args))
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
