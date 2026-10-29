"""One fixed Rayleigh request for inspection and phasor execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from platform import python_version
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeriesMatrix

from gragra._evaluation_setup import OBSERVABLES, EvaluationRecord
from gragra._phasor_matrix import _phasor_matrices
from gragra._rayleigh_contract import (
    RayleighResolvedEvaluation,
    check_surface_precision,
    rotate_rayleigh_tensor,
    rotate_rayleigh_vector,
)
from gragra.constants import G_SI

if TYPE_CHECKING:
    from gragra.geometry import Appearance


@dataclass(frozen=True)
class RayleighValidationReport:
    valid: bool
    observables: tuple[str, ...]
    target_count: int
    target_regions: Mapping[str, str]


_COMPONENTS = {
    "potential": ("potential",),
    "acceleration": ("x", "y", "z"),
    "gradient": ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
}
_UNITS = {
    "potential": u.m**2 / u.s**2,
    "acceleration": u.m / u.s**2,
    "gradient": u.s**-2,
}


def _rayleigh_generation_record() -> EvaluationRecord:
    """Record the package and runtime versions used for Rayleigh coefficients."""
    return EvaluationRecord(
        backend="rayleigh_literature",
        versions={
            "gragra": version("gragra"),
            "python": python_version(),
            "numpy": np.__version__,
            "astropy": version("astropy"),
            "gwexpy": version("gwexpy"),
        },
        gravitational_constant_si=G_SI,
        g_applications=1,
    )


def _native_matrices(
    arrays: Mapping[str, np.ndarray],
    target_ids: tuple[str, ...],
    frequencies_hz: tuple[float, ...],
) -> dict[str, FrequencySeriesMatrix]:
    """Pack numerical (frequency,target,component) arrays as native matrices."""
    count, bins = len(target_ids), len(frequencies_hz)
    matrices = {}
    for name in OBSERVABLES:
        raw = np.asarray(arrays[name], dtype=np.complex128)
        shape = (bins, count) + (
            () if name == "potential" else (3,) if name == "acceleration" else (3, 3)
        )
        if raw.shape != shape or not np.isfinite(raw).all():
            raise ValueError(
                f"{name} must have finite numerical shape {shape}, got {raw.shape}"
            )
        components = _COMPONENTS[name]
        data = raw.reshape(bins, count, len(components)).transpose(1, 2, 0)
        matrices[name] = data
    return _phasor_matrices(
        matrices,
        target_ids,
        frequencies_hz,
        units=_UNITS,
        cell_name=lambda name, target, component, _row, _column: (
            f"{target} {component} {name}"
        ),
        cell_channel=lambda name, _target, _component, row, column: (
            f"GRAGRA:{name}:{row}:{column}"
        ),
        matrix_name=lambda name: f"Rayleigh {name}",
    )


class RayleighEvaluationSetup:
    __slots__ = ("_resolved", "_context")

    def __init__(self, resolved: RayleighResolvedEvaluation) -> None:
        if not isinstance(resolved, RayleighResolvedEvaluation):
            raise TypeError("resolved must be a RayleighResolvedEvaluation")
        self._resolved = resolved
        from gragra.results._context import EvaluationContext

        spec = resolved.model_spec
        self._context = EvaluationContext(
            model_spec=spec,
            targets=resolved.targets,
            frame=resolved.output_frame,
            observables=resolved.observables,
            representation=spec.representation,
            frequencies=np.asarray(spec.frequencies_hz) * u.Hz,
            reference_time=spec.reference_time_gps,
            g_convention=spec.g_convention,
        )

    @property
    def context(self):
        return self._context

    @property
    def resolved(self) -> RayleighResolvedEvaluation:
        return self._resolved

    @property
    def evaluation(self) -> RayleighResolvedEvaluation:
        return self._resolved

    @property
    def parameters(self):
        return self._resolved.parameters

    @property
    def formulas(self):
        return self._resolved.formulas

    @property
    def references(self):
        return self._resolved.references

    def validate(self) -> RayleighValidationReport:
        resolved = self._resolved
        regions = {
            target_id: check_surface_precision(
                trace,
                resolved.targets.frame,
                resolved.computation_frame,
                target_id=target_id,
            )
            for target_id, trace in zip(
                resolved.targets.ids, resolved.target_traces, strict=True
            )
        }
        return RayleighValidationReport(
            True, OBSERVABLES, len(resolved.targets), MappingProxyType(regions)
        )

    def plot_setup(
        self,
        *,
        view: str = "3d",
        unit: str = "m",
        ax=None,
        appearance: Appearance | None = None,
    ):
        from gragra._plotting._rayleigh_setup import plot_rayleigh_setup

        return plot_rayleigh_setup(
            self, view=view, unit=unit, ax=ax, appearance=appearance
        )

    def run(self):
        """Evaluate all three SI phasors and bind native frequency matrices."""
        self.validate()
        resolved = self._resolved
        spec = resolved.model_spec
        from gragra.observables.rayleigh_literature import (
            rayleigh_halfspace_allspace_observables,
        )
        from gragra.results import GravityResult
        from gragra.results._evaluation import EvaluationResult

        values = rayleigh_halfspace_allspace_observables(
            resolved.target_positions_m,
            density_kg_m3=spec.density_kg_m3,
            v_p_m_s=spec.vp_m_s,
            v_s_m_s=spec.vs_m_s,
            frequency_hz=np.asarray(spec.frequencies_hz),
            vertical_surface_amplitude_m=spec.vertical_displacement_m,
            direction=spec.direction,
            phase_origin_m=spec.phase_origin_m,
        )
        arrays = {
            name: np.asarray(getattr(values, name), dtype=np.complex128)
            for name in OBSERVABLES
        }
        arrays["acceleration"] = rotate_rayleigh_vector(
            arrays["acceleration"], resolved.computation_frame, resolved.output_frame
        )
        arrays["gradient"] = rotate_rayleigh_tensor(
            arrays["gradient"], resolved.computation_frame, resolved.output_frame
        )
        matrices = _native_matrices(arrays, resolved.targets.ids, spec.frequencies_hz)
        record = _rayleigh_generation_record()
        return EvaluationResult(
            values=GravityResult(**matrices),
            evaluation=self,
            record=record,
        )
