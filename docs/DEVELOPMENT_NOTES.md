# projects_clean

A restructured, consolidated, commented copy of `../projects/` (the Python
code behind the thesis's numerical experiments). **The original `projects/`
folder is untouched** -- this is a separate copy for you to review; nothing
here overwrote anything there.

Every numerical claim (the wrapped-normal posterior, the exponential-map
posterior on the sphere, the Christoffel/Riemann/curvature autodiff, the
Fisher metric of a Gaussian decoder) was re-derived and checked by hand
against the code, not just copied — see "What was verified" below. The whole
package was also smoke-tested end to end (train each model a few steps,
run every diagnostic, reload a checkpoint) before being handed back to you;
see "How to run" for the same commands.

## Layout

```
Master_Experiments.ipynb        ONE notebook: train + evaluate all three
                                cases (flat 2d, topological, flat 3d) on
                                every dataset, plus the full analysis
                                toolkit and the reload/replot workflow.
                                See "The master notebook" below.
vae_topology/                  the installable package -- import as
                                `from vae_topology.<module> import <name>`
  datasets.py                  synthetic plane/torus/sphere image datasets
  distributions.py             wrapped-normal (circle) / projected-normal
                                (sphere) posterior + KL utilities
  geometry.py                  Fisher metric, Christoffel/Riemann/K via
                                autodiff, Gauss-Bonnet-on-a-disk suite --
                                ONE implementation, used everywhere
  models/
    euclidean_vae.py           standard R^n-latent VAE (baseline)
    torus_vae.py                T^2-latent VAE
    sphere_vae.py                S^2-latent VAE
  diagnostics/
    curvature_plots.py         |K| / sqrt(det G) maps, Gauss-Bonnet figure
    latent_plots.py             latent-code scatter plots, per-model
                                `diagnose*`/`run_diagnostics*` entry points
  experiments/
    run_store.py                RunStore: persist a run's model, checkpoints,
                                arrays, figures, log; new_batch_dir groups a
                                whole execution's runs into one folder
    run_euclidean_2d.py          flat VAE (2d latent), all datasets
    run_all_topologies.py        the six (dataset x latent topology) runs
    run_euclidean_3d.py           flat VAE (3d latent), all datasets
    validation_curves.py          held-out error curves, worst-point plots
    render_evolution.py           latent/curvature evolution GIF from a
                                run's checkpoints

legacy/                        superseded/broken prototypes, kept for the
                                historical record -- see "Legacy" below
colab/                          the two ORIGINAL Colab notebooks this
                                restructure's `experiments/` scripts and
                                `Master_Experiments.ipynb` are modeled on,
                                plus `VSCODE_COLAB_SETUP.md` (how to point
                                VS Code at your Colab GPU compute units)
notebooks/                      your local analysis notebooks, copied as-is
results/                        created at runtime by RunStore (gitignored;
                                does not exist until you run something)
```

## The master notebook

`Master_Experiments.ipynb` is a thin orchestration layer over
`vae_topology/experiments/` -- it does not duplicate any training or
plotting logic, it just calls into the package section by section:

1. **Setup** -- GPU check, get `vae_topology` onto the runtime (Colab Drive
   mount, or already-local if you're running this in VS Code from inside
   `projects_clean/`), imports.
2. **Config** -- edit epoch counts / device here. Defaults are small (a
   smoke run finishes in well under a minute on CPU); the docstrings in
   `experiments/*.py` and the original `colab/*.ipynb` note what was
   actually used for the thesis's numbers (1000-10000 epochs).
3. **Flat VAE, 2d latent**, all datasets (`run_euclidean_2d`) -- including
   the sphere as a deliberate negative baseline (a flat latent cannot
   represent S^2 without tearing).
4. **Topological VAEs** (torus + sphere), all six combinations
   (`run_all_topologies`) -- the thesis's central experiment.
5. **Flat VAE, 3d latent**, all datasets (`run_euclidean_3d`).
6. **Analysis toolkit** -- `geometry.gauss_bonnet_selftest` (sanity-check the
   geometry code itself against metrics with a known closed-form answer),
   `geometry.gauss_bonnet_vae` (the disk-based check plus a convergence
   refinement study) on a reloaded model, and the full 2x2 Fisher metric's
   anisotropy/condition number.
7. **Reload & replot from saved history/checkpoints** -- see "How the
   history feature works" below; this section is the answer to "how do I
   make a plot from a past run without retraining."
8. **Latent/curvature evolution video** -- retrains torus/sphere with dense
   checkpointing and renders a GIF of the latent scatter + curvature map
   side by side, one frame per checkpoint. See "Evolution videos" below.

Every code cell was actually executed (not just syntax-checked) as part of
building this, with tiny epoch/dataset-size overrides for speed, including a
full pass through section 7's reload path -- see "What was verified" below.

To run it with your Colab compute units from inside VS Code rather than the
Colab web UI, see `colab/VSCODE_COLAB_SETUP.md`.

## How the history feature works, precisely

Every training run leaves behind, under `results/<key>__<timestamp>/`:
`arrays/history.npz` (per-epoch loss/recon/kl/sigma, no model needed to
replot it), `checkpoints/epoch_*.pt` (model weights at several points during
training), `model/state_dict.pt` (final weights), and `meta.json` (config,
including `data_dim` -- see below). The rule for which one you need:

- **Replotting something already computed** (the training curve, a
  curvature map that was already generated) needs only the `arrays/*.npz`
  group for that run -- `run_store.load_run(run_dir)` returns all of them as
  plain numpy arrays, and you plot directly against those (no model, no
  decoder evaluation). `Master_Experiments.ipynb` sections 7.2-7.3 do
  exactly this.
- **Making a NEW plot** (different grid resolution, a diagnostic that wasn't
  run at training time, "what did this look like at an earlier epoch")
  needs the model back: reconstruct the right class (`TorusVAE`/
  `SphereVAE`/`SimpleVAE`) using `data_dim` from `meta.json`, then
  `model.load_state_dict(run_store.load_checkpoint(run_dir, epoch=...)["state_dict"])`
  (`epoch=None` for the latest checkpoint). After that, every function in
  `diagnostics`/`geometry` that takes a model works exactly as if you'd just
  trained it. Section 7.4 does this, and section 6 reuses the same pattern
  to reload a model for the analysis toolkit rather than keeping it in
  memory from section 4 -- deliberately, since keeping it in memory is not
  how you'll actually use this after closing the notebook.

`data_dim` is recorded in `meta.json` explicitly at training time (added
during this pass) rather than reconstructed by re-calling the dataset
function and reading its output shape -- guessing that way silently builds
the wrong architecture if a dataset's `res`/`n` ever changes between runs,
and `load_state_dict` then fails with a confusing shape-mismatch error
instead of a clear one. This was caught by actually executing
`Master_Experiments.ipynb`'s reload cells end to end while building it (with
shrunk datasets, which is exactly the kind of parameter change that exposes
the bug) -- see "What was verified".

## Batch folders: one folder per execution, not per experiment

Every individual `RunStore` used to stamp its OWN timestamp, so running all
six topology experiments once produced six independently-timestamped
sibling folders under `results/` -- close together in time but not
identically named, so "everything from my last run" meant eyeballing which
six of N folders belonged together. Re-running the same notebook cell a
second time then interleaved twelve.

Fixed via `run_store.new_batch_dir`: called once before a section's loop
(sections 3-5 and 8 all do this already), it creates ONE timestamped folder
for that whole execution; each experiment's `RunStore` then nests inside it
by plain name (`timestamp=False`), so the layout is:

```
results/
  euclidean2d__2026-09-17_14-40-27/       <- one run of section 3
    flat_2d_no_border/
    flat_2d_border/
    torus_2d/
    sphere_2d/
  topologies__2026-09-17_14-41-02/        <- one run of section 4
    border_flat_torus/
    flat_torus/
    torus_torus/
    border_flat_sphere/
    flat_sphere/
    sphere_sphere/
  euclidean3d__2026-09-17_14-41-17/       <- one run of section 5
    ...
```

Re-running a section's cell later creates a brand-new batch folder with a
fresh timestamp -- it never mixes with or overwrites a previous execution's
folder. `run_store.list_runs` was updated to search recursively (any depth)
for `meta.json`, so it still finds every individual experiment folder
regardless of this extra nesting level; `Master_Experiments.ipynb` section
7's `[r for r in all_runs if "torus_torus" in r]` pattern needed no changes
at all, since the leaf folder is now literally named `torus_torus` (no
timestamp suffix to match around).

## Evolution videos: watching curvature change over training

`experiments/render_evolution.py` reloads every checkpoint saved for a run,
recomputes the latent codes and the Fisher curvature map at each one, and
assembles a GIF (latent scatter | curvature map, side by side). Two-step,
matching the rest of the package's "compute once, replot freely" split:

- `compute_evolution(run_dir, ...)` -- the expensive step, one curvature
  evaluation per checkpoint. Saves `arrays/evolution.npz`.
- `render_evolution_video(run_dir, ...)` -- cheap, just reads that back and
  plots. Re-run this alone (different `fps`/colours) without recomputing.

The colour scale for `|K|` is fixed across every frame (from percentiles of
the WHOLE stack, not each frame's own min/max) -- without this, matplotlib's
default per-frame autoscaling makes the colour scheme jump independently
every frame, and you can no longer tell whether curvature is actually
growing/shrinking over training or just being rescaled.

This needs denser checkpointing than sections 3-5's default (~50 snapshots
spread over the WHOLE run, which is coarse for a smooth video): pass
`--checkpoint-every` (a small number, e.g. 1-5 epochs) when training, or use
`Master_Experiments.ipynb` section 8, which retrains specifically for this
with a configurable cadence (`CHECKPOINT_EVERY_VIDEO`). Both `render_evolution.py`
and section 8 were executed end to end (small epoch counts, tiny grids) as
part of building this -- see "What was verified".

**You do NOT need a run trained specifically "for the video."**
`compute_evolution` only reads `checkpoints/` and `meta.json` from whatever
run folder you point it at -- any run has both, whether it came from section
3/4/5, a plain CLI call, or section 8's dedicated retrain. Section 8 only
retrains because sections 3-5's default cadence is coarse; if you already
trained something with a small `--checkpoint-every`, skip straight to
`render_evolution.compute_evolution(that_run_dir, ...)`.

Section 8.3 runs the flat case on ALL FOUR datasets in `run_euclidean_2d.
EXPERIMENTS` (one video each), not just one -- and only needs sections 1-2,
not 8.1/8.2, so you can go straight to it if you only want the flat-VAE
videos and don't need `TorusVAE`/`SphereVAE` trained at all.

**Covers all three latent geometries**, including the flat/Euclidean 2d
case (section 8.3): `MODEL_CLASSES` now also has `SimpleVAE`, dispatched via
`meta["latent_manifold"] == "euclidean"` (recorded by `run_euclidean_2d.py`,
along with `model_class`/`latent_dim`/`cyclic`, none of which it wrote to
`meta.json` before this). The flat case needs one thing the compact
manifolds don't: T^2/S^2 have a FIXED coordinate domain, identical at every
checkpoint, but a flat R^2 latent's actual extent is data-dependent and
drifts over training -- plotting each frame with its own auto-fit extent
would make the axes visibly rescale/pan between frames, indistinguishable
from the latent space actually changing. `compute_evolution` fixes this the
same way it fixes colour: one pass over all checkpoints first to find the
union of every frame's code range, then a fixed `lo`/`hi` (new optional
argument on `geometry.curvature_map_euclidean`) reused for every frame's
curvature grid and scatter plot.

## Clean heatmaps for the paper (no latent-code overlay)

`plot_abs_curvature`/`plot_abs_curvature_sphere` overlay the encoded codes
on the curvature/magnification heatmaps by default when `mu` is given (see
"What was verified" -- that overlay used to be silently absent entirely,
now it's there and, reasonably, not everyone wants it in every figure).
Pass `save_clean=True` to also save a second copy with no overlay, named
`<fname>_clean<ext>` next to the original -- e.g. `curvature_torus_torus.pdf`
(diagnostic, with codes) and `curvature_torus_torus_clean.pdf` (bare
heatmap, publication-ready). `run_all_topologies.py`'s `run_torus`/
`run_sphere` now pass this by default, so re-running section 4 already
gives you both files per experiment without any extra argument.

## Latent-point colour: "twilight" instead of "hsv"

Every plot that colours latent points by a CYCLIC true angle (torus theta1/
theta2, sphere azimuth) used `cmap="hsv"` -- the classic rainbow map, chosen
specifically because it wraps (colour at 0 matches colour at 2*pi), which a
non-cyclic map like plain `viridis` does not: swapping to `viridis` directly
would have painted a fake colour discontinuity right at the wrap point,
implying a seam in the latent space that isn't really there (exactly the
kind of visual artifact this codebase's own comments warn about elsewhere).

Changed to `"twilight"` instead -- matplotlib's perceptually-uniform CYCLIC
colormap, same modern design family as `viridis`, without `hsv`'s harsh
banding, and still wraps correctly. Non-cyclic quantities (e.g. the sphere's
polar angle) were already plotted with plain `viridis` and are unchanged.

Touched `diagnostics/latent_plots.py`, `experiments/validation_curves.py`,
`experiments/render_evolution.py`, and a comment in `models/euclidean_vae.py`
referencing the same convention -- nothing in `Master_Experiments.ipynb`
needed to change, since it only calls into these functions and never
hardcodes a colormap itself. Verified by directly exercising every changed
plotting function (`diagnose`, `diagnose_sphere`, `run_diagnostics`,
`run_diagnostics_3d`, `plot_worst_points`, and the evolution-video frame
renderers) end to end.

## TorusVAE's sigma-floor anneal is now an absolute epoch count

`models/torus_vae.py::train`'s `sigma_anneal_frac=0.5` (anneal the posterior
noise floor over half of whatever `epochs` you request) was replaced with
`sigma_anneal_epochs=50` (an absolute epoch count, matching
`models/sphere_vae.py::train`'s existing parameterization exactly).

**Why this was a real bug, not just an inconsistency**: with the old
fraction-based schedule, training for 200 vs. 1000 epochs annealed over 100
vs. 500 epochs respectively -- two genuinely DIFFERENT schedules, not a
short run and a continuation of it, since the noise floor at any given
early epoch differed between the two runs even with the same seed. Whether
the encoder's winding number healed into a continuous map or stayed torn is
exactly the kind of trajectory-sensitive, discrete outcome this would
affect -- which is what you ran into: a 1000-epoch `torus_torus` run coming
out visibly torn/discontinuous while a 200-epoch run of the same
configuration looked fine. `SphereVAE` never showed this because its anneal
window was already absolute (`sigma_anneal_epochs=50`, independent of total
`epochs`), so its early trajectory doesn't depend on the epoch budget at all.

**This changes actual training dynamics for long runs, by design**: the
default anneal window for a 1000-epoch torus run is now 50 epochs instead
of the old 500. A shorter window gives a tear less time to heal, which can
make one MORE likely on long runs, not less -- that tradeoff was a
deliberate choice (confirmed with you rather than picked silently), to
match `SphereVAE`'s parameterization exactly. If you need to reproduce
results generated under the old ~500-epoch anneal for a long run, pass
`sigma_anneal_epochs=500` explicitly:
```python
train_torus(epochs=1000, sigma_anneal_epochs=500, ...)
```
No call site in `experiments/` passed `sigma_anneal_frac` explicitly, so
this is a self-contained change to `torus_vae.py` -- nothing else needed to
change. Verified: checked the per-epoch `sigma_floor` in a small training
run's history to confirm it now reaches its floor at exactly
`sigma_anneal_epochs` regardless of total `epochs`, then re-ran the full
`Master_Experiments.ipynb` smoke test.

## A fourth Colab notebook: `colab/flat_vae_run_analyse.ipynb`

Added after the fact (you found it later on Drive) -- `Colab_run_files/
flat_vae_run_analyse (2).ipynb`, the Colab driver for the flat/Euclidean 2d
case that `run_euclidean_2d.py` didn't originally have a reference for (the
README used to note this gap explicitly). Copied into `colab/` alongside the
other three. What it changed here:

**Incorporated:**
- **Validation curve + worst-point plots for the flat VAE.** This notebook's
  own `train()` tracked a held-out validation curve inline, and every run
  cell called a near-identical `plot_worst_points` afterward -- confirming
  this was always the intended workflow for the flat case too, just never
  wired into `run_euclidean_2d.py` (which only had latent/curvature/
  Gauss-Bonnet plots). Now it calls `experiments.validation_curves.
  make_hook`/`plot_curve`/`plot_worst_points`, exactly matching
  `run_all_topologies.py`'s pattern, instead of leaving this notebook's
  separate reimplementation of the same thing uncombined.
- **A real bug, fixed at the source rather than copied.** This notebook's
  `run()` cell has to manually do `model, data = model.cpu(), data.cpu()`
  before calling `run_diagnostics`, with a comment explaining why:
  `curvature_map` (now `geometry.curvature_map_euclidean`) hardcodes
  `device="cpu"` internally but never moved the caller's `mu`/data there, so
  a GPU-trained model crashed on a device mismatch. Fixed in
  `diagnostics/latent_plots.py::run_diagnostics` (moves model/data to CPU
  itself now) and `geometry.py::curvature_map_euclidean` (moves `mu_codes`
  to `device` internally, defense in depth) -- no caller needs to remember
  this anymore.
- **A dispatch bug this surfaced while integrating, not present in the
  Colab notebook itself**: `validation_curves.plot_worst_points`'s `kind`
  parameter only branched on `kind == "torus"`, treating every other value
  as "sphere" -- fine as long as only those two ever called it, but wiring
  in the flat VAE (`kind="euclidean"`) would have silently applied the
  sphere's `[0, pi]` polar-axis limits and "theta/phi" labels to a flat R^2
  latent, which has neither. Fixed with an explicit `elif kind == "sphere"`
  and a generic `else` (auto-scaled axes, `z1`/`z2` labels) before wiring
  the flat case in -- caught before it shipped, not after.
- `val_p95` (95th-percentile held-out error) added to the tracked curve and
  plotted alongside train/val/median, matching this notebook's `report()`
  printout emphasis on tail behaviour.

**Deliberately NOT carried over** (per your instruction to skip what isn't
useful for the actual goals here):
- `run_gb`/`gb()`, a Gauss-Bonnet-only fast path skipping the latent/
  curvature figures -- redundant with the `--gauss-bonnet` flag
  `run_euclidean_2d.py` already has, which runs the identical check as part
  of the bundled diagnostics.
- `report()`, a console summary + optional error-histogram panel -- not
  needed for this project's actual focus (curvature/topology), and
  `plot_curve`'s printed one-line summary already covers the same "how bad,
  how much of a gap" question.
- The final cell (`"sphere on flat small 1"`/`"...mall 2"`, 750/1250
  epochs) -- an exploratory scratch comparison (note the typo), not a
  deliberate reusable method.

## How to run

**Easiest: open `Master_Experiments.ipynb`** and run cells top to bottom
(edit the config cell's epoch counts first). Works in VS Code locally, in
VS Code connected to a Colab GPU (`colab/VSCODE_COLAB_SETUP.md`), or in the
Colab web UI directly.

Or run any experiment from the command line, from inside `projects_clean/`:

```bash
python -m vae_topology.experiments.run_euclidean_2d            # flat, 2d latent, all datasets
python -m vae_topology.experiments.run_all_topologies          # torus + sphere, all six
python -m vae_topology.experiments.run_all_topologies --only torus_torus
python -m vae_topology.experiments.run_euclidean_3d             # flat, 3d latent, all datasets
python -m vae_topology.experiments.run_euclidean_3d --only sphere_3d
```

All three accept `--epochs`, `--seed`, `--device`, `--only <keys>`, `--show`
(open figure windows instead of just saving), `--checkpoint-every` (the
euclidean scripts) / periodic checkpoints are always on for
`run_all_topologies`, and `--stop-on-error`. Run with `--help` for the full
list. No `pip install -e .` needed -- `python -m` from this directory puts
`vae_topology/` on the import path because it is a proper package
(`__init__.py` everywhere), which is also why every import in every file
consistently reads `from vae_topology.<module> import ...` (see "Import
consistency" below).

## Why this looks different from `projects/`

**Import consistency.** The original had no `__init__.py` anywhere and three
incompatible import styles in active use simultaneously (`from
projects.architecture.X import Y`, `from architecture.X import Y`, `from X
import Y`) -- sometimes mixed within a single file
(`structure/sandbox_log.py`). It only worked via IDE- or
notebook-injected `sys.path` hacks. Every file here uses one convention:
`from vae_topology.<module> import <name>`.

**One implementation of the math that was previously copy-pasted.**
Consolidated per your instruction, into:
- `geometry.py`: the Fisher metric, Christoffel symbols, Riemann tensor /
  Gaussian curvature, and the Gauss-Bonnet-on-a-disk machinery were each
  independently implemented 2-4 times across `structure/runstore.py`,
  `visualization/curvature_plot.py`, `visualization/vaediagnostic.py`, and
  `trial/GBsphere.py`/`trial/simpleembedding.py`. One of those copies
  (`trial/GBsphere.py`) was outright broken (see Legacy). The others agreed
  with each other and with the convention stated in `thesis/main.tex`
  (checked index-by-index — see "What was verified"); `vaediagnostic.py`'s
  metric-function-generic version was the most general and became the basis
  for `geometry.py`.
- `distributions.py`: the wrapped-normal (torus) and projected-normal
  (sphere) posterior/KL math, previously duplicated across
  `architecture/new_tor_vae_charts.py`, `bin_scripts/torvae_charts.py`, and
  `bin_scripts/torvae_mod.py`'s dead code block.

**Model history and checkpoints are now saved, not just final weights and
summary numbers.** The original Colab driver (`colab/topo_vae_run_analyse
.ipynb`) only ever wrote `arrays_<key>.npz` (K, V, mu) and a curve CSV per
run -- if you wanted a different plot later, the only option was to retrain.
Every training loop (`models/*.train`) now returns a full per-epoch history
(loss/recon/kl/sigma), and `experiments/run_store.py`'s `RunStore` saves that
history plus periodic model checkpoints (`RunStore.save_checkpoint`, wired
through `experiments/validation_curves.py::make_hook`) alongside the final
model and every figure/array. `RunStore.load_checkpoint` reloads any of them
later. Concretely, one call to `run_all_topologies.py` now leaves behind,
per experiment:
```
results/<key>__<timestamp>/
  meta.json              run config + status
  log.txt                appended diagnostic printouts
  model/state_dict.pt    final weights
  checkpoints/epoch_*.pt periodic snapshots (from eval_hook)
  arrays/history.npz     per-epoch loss/recon/kl/sigma
  arrays/curvature_*.npz K, V, mu, (torus:) full 2x2 metric + invariants
  figures/*.pdf          every plot the run produced
```
This was previously missing for the sphere model specifically:
`models/sphere_vae.py::train` discarded its per-epoch history entirely
(only `models/torus_vae.py::train` tracked one) -- fixed here for parity.
`models/euclidean_vae.py::train` gained the same `history`/`eval_hook`
machinery for the same reason (it also gained a `device` argument, for
`run_euclidean_3d.py`'s GPU path).

**Colab-driven structure.** `experiments/run_all_topologies.py`,
`experiments/run_euclidean_3d.py`, and (for parity, though there was no
original Colab version of this one) `experiments/run_euclidean_2d.py` are
structured cell-for-cell after `colab/topo_vae_run_analyse.ipynb` and
`colab/latent_3d_colab.ipynb` (config -> helpers -> one driver function per
model kind -> one call per experiment), since those notebooks are what
actually produced the thesis's numbers on a GPU and fixed two real bugs
along the way that the original local scripts still had:
- `run_diagnostics_3d` (in `diagnostics/latent_plots.py`) takes no `grid`
  argument; `architecture/standardvae_3dim.py`'s `__main__` block passed one
  and would raise `TypeError`. Fixed in the ported `models/euclidean_vae.py`
  `__main__` block and in `run_euclidean_3d.py`.
- the sphere's polar angle is on `[0, pi]`, not cyclic; a default cyclic
  colormap fakes a seam at the poles. Both files now pass
  `cyclic=(False, True)` for the sphere case.
`experiments/validation_curves.py` generalizes that same notebook's
validation-curve/worst-point helpers (previously sphere-only, in the
original top-level `projects/sandbox.py`) to work for both the torus and
sphere models.

**Small fixes, applied (not archived):**
- `visualization/vaediagnostic.py` defined `_np()` and `_wrap_pm_pi()` twice
  each (identical, harmless, but confusing) -- collapsed to one each in
  `diagnostics/latent_plots.py`.
- `torch.load` of a checkpoint on PyTorch >= 2.6 raised `UnpicklingError`
  (the new default `weights_only=True` rejects the `torch.__version__`
  metadata field, which is a `TorchVersion` subclass of `str`, not a plain
  `str`) -- found while smoke-testing `RunStore.load_checkpoint` here, fixed
  by storing `str(torch.__version__)` and loading with `weights_only=False`
  (safe: this is your own checkpoint, not an untrusted download).
- **`plot_abs_curvature`/`plot_abs_curvature_sphere` accepted a `mu=`
  argument (every caller passes the encoded codes, to overlay them on the
  curvature map) and silently never plotted it -- computed, then discarded,
  no error, in BOTH the original `curvature_plot.py` and
  `vaediagnostic.py`. Every curvature figure this codebase ever produced was
  missing that overlay. Now actually implemented (`diagnostics/
  curvature_plots.py`): a small scatter of the codes on both panels.
- `meta.json` now records `data_dim` explicitly at training time. It was
  previously absent, so a from-scratch reload had to guess the model's
  input size by re-calling the dataset function and reading its output
  shape -- silently wrong (and a confusing `load_state_dict` crash) if a
  dataset's `res`/`n` differs from what was used at training time. Found by
  actually executing `Master_Experiments.ipynb`'s reload section end to end
  with a shrunk dataset while building it.

**Renamed, not removed:** `bin_scripts/sphere_data.py`'s per-sample
`sphere_render` was renamed `sphere_render_px` in `datasets.py`, purely to
avoid colliding with `datasets.py`'s own, differently-parametrized
`sphere_render` now that both live in one module (see the comment above it).
`make_sphere_dataset_try3` was renamed `make_sphere_dataset_batched`
(describes what it actually is). Nothing else was renamed: several
`_old`/`_new` dataset names look like development history but turned out,
on checking every call site, to be two DELIBERATELY different conditions
still both in active use (e.g. `make_plane_dataset` = "bordered", vs.
`make_plane_dataset_old` = "periodic", per the experiment labels in
`structure/sandbox_log.py`) -- see the docstring in `datasets.py` before
touching either.

## What was dropped (confirmed dead, not just old-looking)

Checked by grepping every call site across the entire original `projects/`
tree before removing anything:
- `datasets.py`'s `sphere_render_old`/`make_sphere_dataset` (stereographic
  renderer): imported in three files, called in none of them.
- `bin_scripts/sphervae.py`: empty file.

## Legacy

Per your call: broken/superseded files were archived into `legacy/` rather
than fixed or dropped, each with a header comment explaining its status and
what superseded it (read the top of the file before assuming it works):

| file | status |
|---|---|
| `GBsphere.py` | **broken** (`NameError` on `__main__`) -- kept exactly as broken, not fixed |
| `simpleembedding.py` | working -- the corrected follow-up to `GBsphere.py` |
| `first_fvae.py` | working, superseded -- most developed torus-VAE prototype, diagnoses & fixes the "torn encoder" bug that became `models/torus_vae.py`'s sigma-floor annealing |
| `torvae_charts.py` | working, superseded -- earliest working chart-based torus VAE |
| `torvae_mod.py` | **contains a real shadowing bug**, kept as-is -- a local `wrap()` is silently overridden by a later import; explores a different architecture (learned periodic feature grid) not carried forward |
| `plot_torus.py` | working, not wired into the live pipeline -- an independently-written 3d torus plotting toolkit with more thorough winding-number diagnostics than what is currently in `diagnostics/` |

None of these are imported by anything in `vae_topology/` (verified by grep
before archiving).

## What was verified

Read in full and checked against `thesis/main.tex` and standard references
before this restructure touched anything:
- The Riemann-tensor/Gaussian-curvature einsums in every copy match the
  convention `R^i_{jkl} = d_k Γ^i_{jl} - d_l Γ^i_{jk} + Γ^i_{ka}Γ^a_{lj} -
  Γ^i_{la}Γ^a_{kj}` and `K = R_{1212}/det(g)` stated in the thesis
  (`main.tex`, "Geometry of Information" / "Theorema Egregium to Global
  Gauss-Bonnet") -- traced index by index.
- The Fisher metric of a diagonal-Gaussian decoder
  (`G_ij = Σ_d ∂_iμ_d∂_jμ_d/σ_d² + 2Σ_d ∂_is_d∂_js_d`) matches the general
  Gaussian Fisher-information decomposition.
- `SphereVAE`'s posterior density includes the correct `r/sin(r)` Jacobian
  for pushing a tangent-plane Gaussian onto S^2 via the exponential map
  (matches Falorsi et al. 2018, cited in the thesis).
- The reconstruction NLL (learned scalar `sigma_x`) is the standard
  isotropic-Gaussian negative log-likelihood everywhere it appears.
- End-to-end smoke test (this session): trained all three models a few
  steps each, ran every diagnostic and the curvature/Gauss-Bonnet pipeline,
  reloaded a mid-training checkpoint, and ran `geometry.gauss_bonnet_selftest`
  (analytic flat/spherical/hyperbolic/torus-pullback metrics with a known
  closed-form answer) -- all closed to within quadrature error. Also ran
  all three `experiments/run_euclidean_2d.py`, `run_all_topologies.py`, and
  `run_euclidean_3d.py` through their actual CLIs and confirmed the on-disk
  run-folder structure (checkpoints, history, arrays, figures) is what this
  README claims it is.
- **`Master_Experiments.ipynb` was actually executed end to end**, all 34
  cells, not just syntax-checked: every code cell's source was extracted
  from the `.ipynb`, epoch counts and dataset sizes were overridden to tiny
  values, and the result was run as a single script covering setup through
  section 7's reload/replot path. This is what caught the `data_dim`/
  reload bug above -- a shrunk dataset is exactly the kind of
  training/reload parameter mismatch that bug needed to surface. The
  Gauss-Bonnet refinement study in section 6.3 also came back with a clean
  ~4.00x residual reduction per grid doubling on this run, matching the
  O(dr^2) convergence `geometry.py` promises.

## What still needs your judgment

This restructure did not change any modeling decision, hyperparameter, or
experimental design — only organization, naming, documentation, and the
consolidation/bugfixes listed above. In particular it did NOT:
- pick which of the two sphere dataset renderers (`sphere_render` /radians/
  vs. `sphere_render_batched(...)`'s pixel-sigma convention) should be the
  one going forward -- both are still here, documented, unchanged.
- touch any training hyperparameter, architecture width, or loss weighting.
- fix `legacy/GBsphere.py`'s bug or `legacy/torvae_mod.py`'s shadowing bug,
  since both are archived, not live, code.
