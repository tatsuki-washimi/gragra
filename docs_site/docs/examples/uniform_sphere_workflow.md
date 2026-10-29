# Uniform-sphere workflow

This notebook walks through the public static-sphere API from Quantity inputs
to analytic and midpoint-source results. It uses a 5 m sphere with density
2.7 g/cm³ and two exterior targets. The three point-source spacings show how
this particular finite grid changes mass and field values; they do not define
an acceptance threshold for the two chosen targets. The repository's separate
convergence checks remain the scientific acceptance gate.

The runner pins a temporary Jupyter kernel to the Python interpreter that
launches it, then checks both `sys.executable` and the imported `gragra` path.
Executed notebook, PNG, and provenance JSON are written outside the repository
by default:

```bash
python scripts/run_uniform_sphere_notebook.py
python scripts/run_uniform_sphere_notebook.py --output-dir /tmp/gragra-sphere-review
```

The notebook keeps validation, plotting, a short visual checklist, execution,
and comparison in separate cells. It displays unit-bearing results, model
parameters, formulas and references, plus per-target error summaries for all
three observables. The plot shows at most 250 deterministically selected
source points and labels that display count against the full source count;
the calculation and mass report still use every discretized point.
