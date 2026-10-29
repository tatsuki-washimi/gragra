"""Target resolution and coefficient generation for Rayleigh responses."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from gragra._rayleigh_contract import (
    RayleighResponseEvaluation,
    check_surface_precision,
    rotate_rayleigh_tensor,
    rotate_rayleigh_vector,
)
from gragra._rayleigh_evaluation_setup import (
    RayleighValidationReport,
    _rayleigh_generation_record,
)

if TYPE_CHECKING:
    from gragra.geometry import Appearance


class RayleighResponseSetup:
    """One resolved Rayleigh request for inspection and coefficient generation."""

    __slots__ = ("_resolved",)

    def __init__(self, resolved: RayleighResponseEvaluation) -> None:
        if not isinstance(resolved, RayleighResponseEvaluation):
            raise TypeError("resolved must be a RayleighResponseEvaluation")
        self._resolved = resolved

    @property
    def resolved(self) -> RayleighResponseEvaluation:
        return self._resolved

    @property
    def evaluation(self) -> RayleighResponseEvaluation:
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
        """Classify every target relative to the half-space surface."""
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
            True,
            resolved.observables,
            len(resolved.targets),
            MappingProxyType(regions),
        )

    def plot_setup(
        self,
        *,
        view: str = "3d",
        unit: str = "m",
        ax=None,
        appearance: Appearance | None = None,
    ):
        """Draw the response geometry without applying the surface gate."""
        from gragra._plotting._rayleigh_setup import plot_rayleigh_setup

        return plot_rayleigh_setup(
            self, view=view, unit=unit, ax=ax, appearance=appearance
        )

    def run(self):
        """Generate SI transfer coefficients from one one-metre kernel call."""
        self.validate()
        resolved = self._resolved
        spec = resolved.model_spec
        from gragra.observables.rayleigh_literature import (
            rayleigh_halfspace_allspace_observables,
        )
        from gragra.results._response import GravityResponse

        frequencies = np.asarray(spec.frequencies_hz, dtype=np.float64)
        basis_amplitude_m = float(spec.contract.basis_amplitude.to_value(u.m))
        values = rayleigh_halfspace_allspace_observables(
            resolved.target_positions_m,
            density_kg_m3=spec.density_kg_m3,
            v_p_m_s=spec.vp_m_s,
            v_s_m_s=spec.vs_m_s,
            frequency_hz=frequencies,
            vertical_surface_amplitude_m=np.full(
                len(frequencies), basis_amplitude_m, dtype=np.complex128
            ),
            direction=spec.direction,
            phase_origin_m=spec.phase_origin_m,
        )

        potential = np.asarray(values.potential, dtype=np.complex128)
        acceleration = np.asarray(values.acceleration, dtype=np.complex128)
        gradient = np.asarray(values.gradient, dtype=np.complex128)
        acceleration = rotate_rayleigh_vector(
            acceleration, resolved.computation_frame, resolved.output_frame
        )
        gradient = rotate_rayleigh_tensor(
            gradient, resolved.computation_frame, resolved.output_frame
        )

        target_count = len(resolved.targets)
        frequency_count = len(frequencies)
        coefficients = {
            "potential": potential.transpose(1, 0)[:, None, None, :]
            / basis_amplitude_m,
            "acceleration": acceleration.transpose(1, 2, 0)[:, :, None, :]
            / basis_amplitude_m,
            "gradient": gradient.reshape(frequency_count, target_count, 9).transpose(
                1, 2, 0
            )[:, :, None, :]
            / basis_amplitude_m,
        }
        coefficients = {
            name: np.asarray(array, dtype=np.complex128)
            for name, array in coefficients.items()
        }
        return GravityResponse(
            **resolved._constructor_kwargs(
                coefficients=coefficients,
                record=_rayleigh_generation_record(),
            )
        )


__all__ = ["RayleighResponseSetup"]
