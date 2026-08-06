"""Canonical list of required symbols for the compiled C++ backend.

Single source of truth for the symbol names that a complete ``gragra._cpp_backend``
extension must export. The per-wrapper guards
(:mod:`gragra.kernels._cpp_planewave`, :mod:`gragra.kernels._cpp_inverse_square`,
:mod:`gragra.kernels._cpp_parametric`) and the CI/wheel sanity checks all reference
these tuples so that the set is defined in exactly one place. Adding a new compiled
symbol therefore only requires updating this module.

This module is intentionally pure: it imports nothing from the compiled backend (and
no third-party packages), so it stays importable even when the C++ backend is absent
or stale, which is exactly when the sanity checks need to run.
"""

PLANEWAVE_REQUIRED = ("plane_wave_weight_factor",)

INVERSE_SQUARE_REQUIRED = (
    "potential_contract_real",
    "potential_contract_complex",
    "acceleration_contract_real",
    "acceleration_contract_complex",
    "gradient_contract_real",
    "gradient_contract_complex",
    "dipole_contract_real",
    "dipole_contract_complex",
)

PARAMETRIC_REQUIRED = (
    "potential_contract_parametric_real",
    "potential_contract_parametric_complex",
    "acceleration_contract_parametric_real",
    "acceleration_contract_parametric_complex",
    "gradient_contract_parametric_real",
    "gradient_contract_parametric_complex",
)

#: Every symbol a complete C++ backend build must export.
ALL_REQUIRED = PLANEWAVE_REQUIRED + INVERSE_SQUARE_REQUIRED + PARAMETRIC_REQUIRED
