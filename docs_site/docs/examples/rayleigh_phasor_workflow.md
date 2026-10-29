# Rayleigh phasor workflow

This notebook connects a complex vertical displacement phasor spectrum to the public Rayleigh half-space API. It defines the homogeneous medium, GPS phase reference, surface geometry, phase origin, propagation direction, and four targets on both sides of the free surface. It validates the target regions, displays 3D and XZ setup plots, and evaluates potential, acceleration, and acceleration gradient as native GWexpy frequency matrices with target rows, component columns, SI units, and the irregular input frequency axis.

The notebook also shifts the phase origin and compensates the input amplitudes using the documented horizontal phase factor. Agreement between these two evaluations checks phase-origin consistency. The independent physical Green-function oracle lives in the test suite and is the scientific reference check.

Run it in the intended Python environment with:

```bash
python scripts/run_rayleigh_phasor_notebook.py
python scripts/run_rayleigh_phasor_notebook.py --output-dir /tmp/gragra-rayleigh-review
```

The runner creates a temporary Jupyter kernelspec pinned to its running interpreter and checks the kernel executable, environment prefix, and imported `gragra` source path. The executed notebook, setup PNG, and `provenance.json` are written to the external output directory (by default `~/.cache/gragra/rayleigh-phasor-workflow/`). The committed source notebook has no saved outputs or execution counts.

The source files are `examples/rayleigh_phasor_workflow.ipynb` and `scripts/run_rayleigh_phasor_notebook.py` in the repository.
