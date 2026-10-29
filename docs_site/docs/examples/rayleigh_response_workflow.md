# Reusable Rayleigh response workflow

This notebook creates one precomputed response from vertical surface displacement to Rayleigh half-space gravity. It displays the input and mapping contract, including coefficient units per metre of displacement, then applies the same response to two complex displacement spectra with explicit `PhasorRepresentation` and GPS reference times. One applied result is compared with a direct `rayleigh_halfspace` evaluation, and the notebook plots vertical acceleration amplitude for selected targets.

The response describes a first-order field change at fixed observation points. Its `local_linearization` applicability is `not_assessed`, so the example makes no claim about finite-motion accuracy. Complex values are phasor amplitudes under the declared convention, not time-domain gravitational fields.

Run the source notebook in the Python environment where the project is installed:

```bash
python scripts/run_rayleigh_response_notebook.py
python scripts/run_rayleigh_response_notebook.py --output-dir /tmp/gragra-rayleigh-response-review
```

The runner creates a temporary Jupyter kernelspec pinned to its running interpreter and checks the kernel executable, environment prefix, and imported `gragra` source path. It writes the executed notebook, plot, and `provenance.json` to the external output directory. By default this is `~/.cache/gragra/rayleigh-response-workflow/`. The committed notebook is clean source with no saved cell outputs or execution counts.

The notebook source is `examples/rayleigh_response_workflow.ipynb`; the runner is `scripts/run_rayleigh_response_notebook.py`.
