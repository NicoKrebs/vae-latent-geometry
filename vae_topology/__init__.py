"""vae_topology -- Fisher-information / Chern-Gauss-Bonnet analysis of VAE latent spaces.

Package layout:
    datasets.py       synthetic image datasets whose true generative factors
                       live on a plane, a torus, or a sphere.
    distributions.py  wrapped-normal (circle) and projected-normal (sphere)
                       posterior/prior utilities for topologically
                       constrained latent spaces.
    geometry.py        the numerical core: Fisher information metric,
                       Christoffel symbols, Riemann/Gaussian curvature via
                       autodiff, and the Gauss-Bonnet-on-a-disk verification
                       suite. One implementation of each, reused everywhere.
    models/            VAE architectures: Euclidean baseline, torus, sphere.
    diagnostics/       plotting built on top of geometry.py and models/.
    experiments/       the scripts that actually run training + evaluation
                       and persist results (RunStore).

Import convention: everything in this package is imported as
`from vae_topology.<module> import <name>`. Run experiment scripts as
`python -m vae_topology.experiments.<script>` from the `projects_clean`
directory (the parent of this package), so the package is importable
without installing it or hand-editing sys.path.

See ../docs/DEVELOPMENT_NOTES.md for how this maps onto the original `projects/` folder,
and what changed.
"""
