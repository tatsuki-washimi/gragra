"""Immutable displacement series on a fixed point mesh."""

from dataclasses import dataclass

import numpy as np

from gragra.fields._series import _array, _common, _finite_real, _Series


@dataclass(frozen=True, init=False)
class FixedMeshDisplacementSeries(_Series):
    mass_kg: np.ndarray
    displacement_m: np.ndarray
    t_ref_s: float | None
    phase_origin_m: np.ndarray | None
    fourier_convention: str | None

    def __init__(
        self,
        positions_m,
        mass_kg,
        displacement_m,
        axis_values,
        axis_kind,
        coordinate_frame,
        coordinate_origin_m,
        source_ids,
        source_metadata=None,
        *,
        t_ref_s=None,
        phase_origin_m=None,
    ):
        n, s = len(positions_m), len(axis_values)
        _common(
            self,
            positions_m,
            axis_values,
            axis_kind,
            coordinate_frame,
            coordinate_origin_m,
            source_ids,
            source_metadata,
        )
        mass = _array("mass_kg", mass_kg, (n,))
        disp = _array(
            "displacement_m",
            displacement_m,
            (s, n, 3),
            complex_ok=axis_kind == "frequency",
        )
        if (mass < 0).any():
            raise ValueError("mass_kg must be non-negative")
        if axis_kind == "frequency":
            if t_ref_s is None or phase_origin_m is None:
                raise ValueError("frequency inputs require t_ref_s and phase_origin_m")
            t_ref_s = _finite_real("t_ref_s", t_ref_s)
            phase_origin_m = _array("phase_origin_m", phase_origin_m, (3,))
            convention = "Re[q_hat exp(-2pi i f(t-t_ref_s))]"
        else:
            if t_ref_s is not None or phase_origin_m is not None:
                raise ValueError(
                    "time inputs must not declare frequency phase metadata"
                )
            convention = None
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "displacement_m", disp)
        object.__setattr__(self, "t_ref_s", t_ref_s)
        object.__setattr__(self, "phase_origin_m", phase_origin_m)
        object.__setattr__(self, "fourier_convention", convention)
