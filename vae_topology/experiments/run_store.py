"""Persist everything a diagnostic run produces: one folder per run.

Folder layout produced by RunStore("results", name="torus_beta4"):

    results/
      torus_beta4__2026-08-10_14-31-07/
        meta.json                  run parameters and array manifest
        log.txt                    appended diagnostic printouts
        arrays/
          latents.npz              mu, true_angles, ...
          curvature_torus.npz      K, V, G, t1, t2, invariants
        figures/
          latent_torus.pdf / .png
          latent_domain.pdf / .png
        model/
          state_dict.pt

For a SET of experiments run together (e.g. all six dataset x topology
combinations), use `new_batch_dir` once to get one timestamped folder for
the whole execution, then pass it as `root=` with `timestamp=False` to each
individual `RunStore` -- see `new_batch_dir`'s docstring. `experiments/
run_all_topologies.py`, `run_euclidean_2d.py`, and `run_euclidean_3d.py` all
do this already, so re-running a notebook section produces exactly one new
top-level folder per execution:

    results/
      topologies__2026-09-17_15-30-00/     <- one execution of section 4
        border_flat_torus/
        flat_torus/
        torus_torus/
        ...

Reload with:

    from vae_topology.experiments.run_store import load_run
    r = load_run(".../torus_beta4__2026-08-10_14-31-07")
    G = r["curvature_torus"]["G"]        # (grid, grid, 2, 2)

`save_model` keeps only the FINAL weights -- fine for the arrays/figures, but
it means the original Colab driver scripts (which additionally never called
even that -- they only ever wrote `arrays_<key>.npz` with K/V/mu, not the
model) could not regenerate a different plot later without retraining from
scratch. `save_checkpoint`/`load_checkpoint` below close that gap: call
`save_checkpoint` periodically during training (`experiments.
run_all_topologies` does this from the same `eval_hook` that already updates
the validation curve, so it costs nothing extra), and `load_checkpoint` any
of those snapshots back later to make a new plot against the model as it was
at that epoch, without retraining.

This is pure storage bookkeeping; the Fisher-metric computation this module
used to also contain (`decoder_jacobian`, `fisher_metric`, `fisher_metric_grid`,
`metric_invariants`) moved to `vae_topology.geometry` as part of consolidating
three separate copies of that math into one -- see the module docstring
there. Import them from there now.

Originally `structure/runstore.py`.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

import numpy as np
import torch

__all__ = ["RunStore", "load_run", "list_runs", "append_log", "load_checkpoint",
          "new_batch_dir"]


def _np_(x):
    """Tensor / array / list -> detached numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_log(log_path, header, lines):
    """Append a timestamped block to a text file. Creates it if missing.

    Standalone version of RunStore.log, for use in plotting functions that
    do not have a store handy. Pass log_path=None to disable.
    """
    if not log_path:
        return None
    if isinstance(lines, str):
        lines = [lines]
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _ensure_parent(log_path)
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {header}\n")
        fh.write("\n".join(lines) + "\n\n")
    return os.path.abspath(log_path)


def new_batch_dir(root="results", label="batch"):
    """Create and return ONE fresh timestamped directory under `root`, meant
    to hold every experiment run in a single execution -- one notebook-cell
    run, one CLI invocation -- as plain-named subfolders.

    Without this, every individual `RunStore(root=RESULTS_ROOT, name=key)`
    stamps its OWN timestamp, so running all-six-experiments once produces
    six independently-timestamped sibling folders under `results/` with no
    way to tell "these six belong together" except by eyeballing similar
    (but not identical) timestamps. Re-running the same cell a second time
    then interleaves twelve folders. Calling `new_batch_dir` once before the
    loop, then passing `RunStore(root=batch_dir, name=key, timestamp=False)`
    for each experiment, gives instead:

        results/topologies__2026-09-17_15-30-00/
          border_flat_torus/
          flat_torus/
          torus_torus/
          ...

    one folder per execution, found by its timestamp exactly once, with
    every experiment from that run nested plainly inside it by name.
    """
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.abspath(os.path.join(root, f"{label}__{stamp}"))
    os.makedirs(path, exist_ok=True)
    return path


class RunStore:
    """One folder per run. Every save_* method returns the path it wrote."""

    def __init__(self, root="results", name="run", timestamp=True):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = f"{name}__{stamp}" if timestamp else name
        self.path = os.path.abspath(os.path.join(root, folder))
        for sub in ("arrays", "figures", "model", "checkpoints"):
            os.makedirs(os.path.join(self.path, sub), exist_ok=True)
        self._meta = {"name": name, "created": stamp, "path": self.path}
        self.meta()  # write meta.json immediately so the run is identifiable

    def __repr__(self):
        return f"RunStore({self.path!r})"

    # ---- arrays ----------------------------------------------------------
    def save_arrays(self, group, **arrays):
        """save_arrays('curvature_torus', K=K, V=V, G=G) -> arrays/curvature_torus.npz

        None values are skipped. Everything is converted to numpy and
        compressed. Re-calling with the same group overwrites that file.
        """
        clean = {k: _np_(v) for k, v in arrays.items() if v is not None}
        out = os.path.join(self.path, "arrays", group + ".npz")
        np.savez_compressed(out, **clean)
        self._meta.setdefault("arrays", {})[group] = {
            k: {"shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in clean.items()
        }
        self.meta()
        return out

    def load_arrays(self, group):
        """Read back a group written by save_arrays, as a plain dict."""
        with np.load(os.path.join(self.path, "arrays", group + ".npz")) as z:
            return {k: z[k] for k in z.files}

    # ---- figures ---------------------------------------------------------
    def save_figure(self, fig, name, ext=".pdf", dpi=150, also_png=True):
        """Vector copy for the thesis, raster copy for quick viewing."""
        if fig is None:
            return None
        out = os.path.join(self.path, "figures", name + ext)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        if also_png and ext != ".png":
            fig.savefig(os.path.join(self.path, "figures", name + ".png"),
                        dpi=dpi, bbox_inches="tight")
        self._meta.setdefault("figures", []).append(name + ext)
        self.meta()
        return out

    def adopt_figure(self, existing_path, name=None):
        """Copy a figure a plotting function already wrote into the run folder."""
        name = name or os.path.basename(existing_path)
        dst = os.path.join(self.path, "figures", name)
        shutil.copy2(existing_path, dst)
        self._meta.setdefault("figures", []).append(name)
        self.meta()
        return dst

    def figure_path(self, name, ext=".pdf"):
        """Path inside the run folder, for functions that take an fname."""
        return os.path.join(self.path, "figures", name + ext)

    # ---- model -----------------------------------------------------------
    def save_model(self, model, optimizer=None, extra=None):
        payload = {
            "state_dict": model.state_dict(),
            "class": type(model).__name__,
            "torch_version": str(torch.__version__),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if extra:
            payload.update(extra)
        out = os.path.join(self.path, "model", "state_dict.pt")
        torch.save(payload, out)
        return out

    def save_checkpoint(self, model, epoch, optimizer=None, extra=None):
        """Snapshot the model (and optionally optimizer state) at `epoch`,
        under checkpoints/epoch_{epoch:06d}.pt. Call this from a training
        loop's `eval_hook` (torus/sphere models already accept one) so that
        every diagnostic run leaves behind not just its final weights but a
        trajectory of checkpoints -- enough to regenerate ANY plot (latent
        scatter, curvature map, reconstruction) at any recorded point in
        training later, without retraining. Cheap relative to training time;
        expensive only in disk space, which is why this is opt-in via
        `eval_every` rather than every epoch.
        """
        payload = {
            "state_dict": model.state_dict(),
            "class": type(model).__name__,
            "torch_version": str(torch.__version__),
            "epoch": int(epoch),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if extra:
            payload.update(extra)
        out = os.path.join(self.path, "checkpoints", f"epoch_{int(epoch):06d}.pt")
        torch.save(payload, out)
        self._meta.setdefault("checkpoint_epochs", []).append(int(epoch))
        self.meta()
        return out

    def save_source(self, *paths):
        """Snapshot the .py files that produced the run. Cheap insurance."""
        dst_dir = os.path.join(self.path, "source")
        os.makedirs(dst_dir, exist_ok=True)
        written = []
        for p in paths:
            if os.path.isfile(p):
                dst = os.path.join(dst_dir, os.path.basename(p))
                shutil.copy2(p, dst)
                written.append(dst)
        return written

    # ---- metadata / log --------------------------------------------------
    def meta(self, **kv):
        """Merge key/values into meta.json and rewrite it."""
        self._meta.update(kv)
        with open(os.path.join(self.path, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(self._meta, fh, indent=2, default=str)
        return os.path.join(self.path, "meta.json")

    def log(self, lines, header=None):
        """Append lines to log.txt inside the run folder."""
        return append_log(os.path.join(self.path, "log.txt"), header or "", lines)


def load_run(path):
    """Return {'meta': {...}, '<group>': {'K': ndarray, ...}, ...}."""
    out = {}
    meta_p = os.path.join(path, "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p, encoding="utf-8") as fh:
            out["meta"] = json.load(fh)
    adir = os.path.join(path, "arrays")
    if os.path.isdir(adir):
        for f in sorted(os.listdir(adir)):
            if f.endswith(".npz"):
                with np.load(os.path.join(adir, f)) as z:
                    out[f[:-4]] = {k: z[k] for k in z.files}
    return out


def load_checkpoint(run_dir, epoch=None, map_location="cpu"):
    """Load a checkpoint written by `RunStore.save_checkpoint`.

    epoch=None loads the LATEST checkpoint in the run. Returns the raw
    payload dict (state_dict, class, epoch, ...); apply it to a freshly
    constructed model yourself, e.g.
        model = TorusVAE(data_dim=784)
        model.load_state_dict(load_checkpoint(run_dir)["state_dict"])
    """
    cdir = os.path.join(run_dir, "checkpoints")
    if epoch is not None:
        path = os.path.join(cdir, f"epoch_{int(epoch):06d}.pt")
    else:
        files = sorted(f for f in os.listdir(cdir) if f.startswith("epoch_"))
        if not files:
            raise FileNotFoundError(f"no checkpoints in {cdir}")
        path = os.path.join(cdir, files[-1])
    # weights_only=False: this is our own checkpoint (state_dict + a few
    # plain-data fields), not an untrusted download. PyTorch >= 2.6 defaults
    # weights_only=True, which rejects even harmless metadata like the
    # `torch.__version__` string and raises UnpicklingError.
    return torch.load(path, map_location=map_location, weights_only=False)


def list_runs(root="results"):
    """Every run folder under root (searched recursively, any depth), sorted.

    Recursive because a run folder is no longer necessarily a direct child
    of `root` -- `new_batch_dir` nests several runs one level deeper, inside
    a single per-execution batch folder. A folder counts as a "run" purely
    by containing a meta.json; sorting the resulting paths lexicographically
    also sorts by batch timestamp (then experiment name) since that
    timestamp is an ISO-ordered prefix of the parent directory name.
    """
    if not os.path.isdir(root):
        return []
    runs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "meta.json" in filenames:
            runs.append(dirpath)
    return sorted(runs)
