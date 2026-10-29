# Gen1 independent scientific verification

Reference generators do not import production code or read production outputs.
Fixture matrices, methods, physical scales, uncertainty components and acceptance
coefficients are declared in `docs/design/gen1/*-science-protocol.md` and
`acceptance-error-budget.md`. Raw evidence stays outside Git. The science manifest
indexes frozen data and recorded runs. Numerical estimates are not rigorous total
error bounds. Exact parity bounds apply only to certified whole observables.
Grid comparisons verify the declared discrete center-mass source, not continuum
discretization accuracy.

From the selected checkout, run collectors as modules with `PYTHONPATH=tests` in
the `gragra` conda environment. Each requires a new external `--output` directory:

- `oracles.gen1.run_sphere_reference`
- `oracles.gen1.run_sphere_high_degree_reference` (36 discovery-following cases)
- `oracles.gen1.run_surface_parallel --family cuboid --workers 2`
- `oracles.gen1.run_surface_parallel --family cylinder --workers 2`
- `oracles.gen1.run_noncavity_reference`
- `oracles.gen1.plane_static_reference`
- `oracles.gen1.run_source_reference`

The surface collector's optional `--seed` accepts prior JSONL only after checking
saved scientific-input snapshots. Omitting it regenerates the full matrix. High
precision fallback preserves acceptance coefficients; unresolved uncertainty
does not pass. Run the L0 collector directly as
`tests/oracles/gen1/derivations/run_derivations.py --evidence-root EXTERNAL_DIR`.
Run the `tests/test_gen1_*oracle.py` / science acceptance test files listed
in the science manifest for selfchecks.

The four `compare_*_production.py` scripts are a separate comparison layer. They
intentionally import the public production API, frozen JSONL and `acceptance.py`;
they do not import computational reference helpers. Each takes `--references`
and `--output`. The noncavity driver also takes `--static-plane-references` for
the independently frozen real-field supplement; the surface driver takes
`--family cuboid` or `--family cylinder`. They pin input and acceptance
hashes, record production source hashes, assert SI units, and compare baseline
plus two general Frame transformations. Source exceptions are recorded and
cause unsuccessful exit; partial successful results cannot pass the full gate.

Select the actual imported production checkout explicitly. An editable install
from another checkout can otherwise select the wrong product. Recorded runs use
the external source-pinned runner identified in the science manifest; its
`--script` mode selects a checkout and records HEAD and actual source-tree hash.
The Git comparison drivers were formatted from reviewed external versions;
their hashes are separate. Rerunning these Git versions against integrated
production is recorded separately in the science index. The high-degree Sphere
dataset uses the same frozen-data schema and Sphere comparison driver, with a
separate declaration following discovery. Its original low-degree dataset and
the other five original datasets remain unchanged.

## B1 stochastic production comparison

`compare_stochastic_production.py` is gated by an external `P_science`
authorization binding. It checks the exact production checkout, accepted
`b1-002` reference, comparator revision, frozen inputs, and runtime inventory
before it can create an output directory. The 30 science keys must map exactly;
the 12 whole-observable zero keys require exact zero across all nine matrix
entries. The source-connection gate also requires a reviewed receipt covering
the separate 56-reference and 14-consistency auxiliary keys.

`--help` does not import the production package. `--preflight-only` checks the
authorization and pinned inputs but does not import or call a production
function and does not create comparison output. Use it only with a fully
reviewed `AUTHORIZED` binding:

Run these module commands from the comparator checkout root:

```bash
conda run -n gragra-stoch-integration-20261002 python \
  -m tests.oracles.gen1.compare_stochastic_production \
  --policy tests/oracles/gen1/b1-production-comparator-policy-v1.json \
  --authorization-binding /absolute/path/to/authorization-binding.json \
  --expected-authorization-sha256 EXPECTED_BINDING_SHA256 \
  --expected-review-sha256 EXPECTED_REVIEW_SHA256 \
  --preflight-only
```

The `--execute` mode accepts only the same authorization inputs and writes a
new directory outside both Git worktrees. It stores each full 3×3 result and
its decision evidence durably. This mode must run as a separate, explicitly
authorized step; adapter and driver implementation or review work must not
invoke it.

The accepted review sidecar records the total nine-source allocation but not
the individual source values. The adapter therefore preserves exact `u_up`
and `3*u_up` totals and passes `3*u_up` through one aggregate kernel transport
slot whose method states that it is not source attribution. The rounded
`uncertainty_matrix_upper` display is not used as a decision value.
