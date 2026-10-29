# Gen1 Wave 1--3 Analytic Verification Oracles

This directory contains independent verification oracles, SymPy CAS checks, and test fixtures for the Gen1 Wave 1--3 analytic catalogue:

- **Step 6a**: Cylindrical Density-Mode Gravity
- **Step 6b (AC3)**: Acoustic Specialization (isentropic sound-speed density coupling ($A = P/c_s^2$), rigid wall Neumann boundary)
- **Step 7 (T4)**: Deterministic Temperature Specialization (isobaric thermal expansion coupling, insulating Neumann boundary; for a given temperature amplitude $\Theta$ the spatial gravity transfer does not depend on the modulation frequency, which is a structural property of the linear Poisson problem, not a numerical check; $\Theta(\omega)$ itself is an external input)

## Contents

- `gravity.py`: Candidate closed-form evaluation ($\Phi$, $a$, $\Gamma$) in cylindrical coordinates and Cartesian tensor transformation.
- `oracle.py`: Independent split Green function numerical quadrature oracle.
- `ac3_case_matrix.json`: 13 test modes across angular modes $m \in \{0, 1, 2, -1\}$, axial wavenumbers $k_z$, and radial roots $\kappa = \alpha_{mn} / R$.
- `ac3_cas_checks.py`: SymPy algebraic verification of the acoustic wave equation, rigid-wall condition, density coupling, and Poisson/trace identity.
- `ac3_evaluation.py`: Multi-precision (40, 55, 70 dps) candidate-vs-oracle numerical evaluation.
- `t4_case_matrix.json`: 15 test modes with serialized 80-dps Neumann root constants and physical parameters.
- `t4_cas_checks.py`: SymPy algebraic verification of insulating Neumann condition, isobaric density coupling, Poisson/trace identity, and zero-mode classification.
- `t4_evaluation.py`: Multi-precision (40, 55, 70 dps) candidate-vs-oracle numerical evaluation across 9 scientific quality gates.
- `verify_wave1_3.py`: Unified one-command runner executing all four suites.

## Reproduction Commands

Run all checks via the unified runner:

```bash
conda run -n gragra python tests/oracles/gen1/wave1_3/verify_wave1_3.py
```

Or run individual suites:

```bash
conda run -n gragra python tests/oracles/gen1/wave1_3/ac3_cas_checks.py
conda run -n gragra python tests/oracles/gen1/wave1_3/ac3_evaluation.py
conda run -n gragra python tests/oracles/gen1/wave1_3/t4_cas_checks.py
conda run -n gragra python tests/oracles/gen1/wave1_3/t4_evaluation.py
```

Optional `--output-dir <path>` can be passed to `ac3_evaluation.py` and `t4_evaluation.py` to write full evaluation records (`result.json`, `numerical-summary.json`, `numerical-results.json`).

## Independence Boundary Note

- **Radial field computation**: The radial potential $f(r)$ and its derivatives are computed independently by the split Green numerical quadrature oracle versus the closed-form Bessel function expressions.
- **Coordinate transformation**: Cylindrical-to-Cartesian vector and tensor transformations use shared geometric routines (`field_from_radial` in `gravity.py`).
