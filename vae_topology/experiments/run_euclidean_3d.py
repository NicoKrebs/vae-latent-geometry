"""Train the standard Euclidean-latent VAE (`models.euclidean_vae`) with a
3d latent on all four datasets, and check whether the codes actually use all
three dimensions (a torus needs a genuinely non-flat R^3 embedding to close
up on itself) or collapse onto a 2d sheet.

    python -m vae_topology.experiments.run_euclidean_3d
    python -m vae_topology.experiments.run_euclidean_3d --only sphere_3d
    python -m vae_topology.experiments.run_euclidean_3d --epochs 500 --show

Structure mirrors `Colab_run_files/latent_3d_colab.ipynb`'s `run()` driver
(one timed, memory-cleaned-up call per dataset -- originally four separate
cells in `architecture/standardvae_3dim.py`'s __main__ block, generalized
into a loop here as in the Colab notebook), including the two fixes that
notebook's markdown documents:
  - `run_diagnostics_3d` takes no `grid` argument (the original script's
    calls passed one and raised TypeError).
  - the sphere run needs `cyclic=(False, True)`: its polar angle is on
    [0, pi], and the default cyclic colormap would fake a seam at the poles.

What this version adds on top of the Colab script: a `RunStore` per
experiment, persisting the model, its full per-epoch loss history, and
periodic checkpoints (`--checkpoint-every`), instead of only ever writing
the final diagnostic PDF -- see `run_store.py`'s docstring for why that
matters.
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
from vae_topology.models.euclidean_vae import train              # noqa: E402
from vae_topology.diagnostics.latent_plots import run_diagnostics_3d  # noqa: E402
from vae_topology.experiments.run_store import RunStore, new_batch_dir  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(HERE, "..", "..", "results")

EXPERIMENTS = [
    dict(key="flat_3d_no_border", dataset=make_plane_dataset_old,
         cyclic=(True, True), label="flat plane, no border contact"),
    dict(key="flat_3d_border",    dataset=make_plane_dataset,
         cyclic=(True, True), label="flat plane, border contact"),
    dict(key="torus_3d",          dataset=make_torus_dataset,
         cyclic=(True, True), label="torus"),
    dict(key="sphere_3d",         dataset=make_sphere_dataset_batched,
         cyclic=(False, True), label="sphere"),
]


def _history_arrays(history):
    if not history:
        return {}
    return {k: np.asarray([h[k] for h in history]) for k in history[0]}


def _batch_dir(args):
    """See run_store.new_batch_dir / run_euclidean_2d._batch_dir."""
    bd = getattr(args, "batch_dir", None)
    if bd is None:
        bd = new_batch_dir(RESULTS_ROOT, label="euclidean3d")
    return bd


def run_one(cfg, args):
    key, label = cfg["key"], cfg["label"]
    store = RunStore(root=_batch_dir(args), name=key, timestamp=False)
    print(f"\n=== {key}  ({label}) -> {store.path}")
    store.meta(experiment=key, description=label, latent_dim=3,
               dataset=cfg["dataset"].__name__, epochs=args.epochs,
               seed=args.seed, started=datetime.now().isoformat(timespec="seconds"))

    t0 = time.time()
    hook = None
    if args.checkpoint_every:
        def hook(model, ep):
            store.save_checkpoint(model, ep)

    model, data, angles, history = train(
        epochs=args.epochs, dataset=cfg["dataset"], latent_dim=3, seed=args.seed,
        device=args.device, log_every=args.log_every,
        eval_hook=hook, eval_every=args.checkpoint_every)
    t_train = time.time() - t0

    store.save_model(model)
    store.meta(data_dim=int(data.shape[1]))
    store.save_arrays("history", **_history_arrays(history))

    lat, sv = run_diagnostics_3d(model, data, angles, cyclic=cfg["cyclic"],
                                 fname=store.figure_path(f"latent_3d_{key}"))
    store.save_arrays("latent", lat=lat, singular_values=sv)

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
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="save a model checkpoint every N epochs (0 = final weights only)")
    p.add_argument("--only", nargs="+", metavar="KEY", help="run only these experiments")
    p.add_argument("--show", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    args = p.parse_args()

    args.batch_dir = new_batch_dir(RESULTS_ROOT, label="euclidean3d")
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
