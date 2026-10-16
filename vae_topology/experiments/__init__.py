"""Scripts that train a model, run diagnostics, and persist results.

    run_store               RunStore: one timestamped folder per run, holding
                             the model checkpoint, arrays, figures, and a log.
    run_all_topologies       trains and evaluates all six
                             (dataset topology x latent topology) combinations
                             used in the thesis, via RunStore.
    validation_curves        held-out reconstruction error curves and
                             worst-reconstructed-point diagnostics.
"""
