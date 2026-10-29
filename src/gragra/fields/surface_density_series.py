"""Immutable fixed-quadrature surface-density series."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from gragra.fields._series import _array, _common, _finite_real, _freeze, _Series


@dataclass(frozen=True, init=False)
class FixedMeshSurfaceDensitySeries(_Series):
    areas_m2: np.ndarray
    normals: np.ndarray
    surface_density_kg_m2: np.ndarray
    face_metadata: Mapping[str, object]
    t_ref_s: float | None
    phase_origin_m: np.ndarray | None
    fourier_convention: str | None

    def __init__(
        self,
        positions_m,
        areas_m2,
        normals,
        surface_density_kg_m2,
        axis_values,
        axis_kind,
        face_metadata,
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
        areas = _array("areas_m2", areas_m2, (n,))
        normals_array = _array("normals", normals, (n, 3))
        density = _array(
            "surface_density_kg_m2",
            surface_density_kg_m2,
            (s, n),
            complex_ok=axis_kind == "frequency",
        )
        if (areas <= 0).any():
            raise ValueError("areas_m2 must be positive")
        if not np.allclose(
            np.linalg.norm(normals_array, axis=1), 1.0, rtol=0, atol=1e-12
        ):
            raise ValueError("normals must be unit vectors")
        if not isinstance(face_metadata, Mapping):
            raise TypeError("face_metadata must be a mapping")
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
        object.__setattr__(self, "areas_m2", areas)
        object.__setattr__(self, "normals", normals_array)
        object.__setattr__(self, "surface_density_kg_m2", density)
        object.__setattr__(self, "face_metadata", _freeze(face_metadata))
        object.__setattr__(self, "t_ref_s", t_ref_s)
        object.__setattr__(self, "phase_origin_m", phase_origin_m)
        object.__setattr__(self, "fourier_convention", convention)
