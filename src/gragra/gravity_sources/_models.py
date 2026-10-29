"""High-level models backed by an already resolved gravity source."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from gragra._evaluation_setup import (
    OBSERVABLES,
    DiscreteSource,
    ResolvedEvaluation,
    UniformSphereModelSpec,
    _StaticPointSourceBinding,
    _StaticPointSumSpec,
    _validate_field_definition,
)
from gragra.geometry._frame import Frame
from gragra.gravity_sources.volume import VolumeElementSource
from gragra.gravity_sources.weighted_points import (
    PointMass,
    PointMassCloud,
    WeightedPointSource,
)
from gragra.results import Contribution, GravityFieldDefinition, PotentialReference
from gragra.targets._high_points import TargetPoints

StaticPointInput: TypeAlias = (
    DiscreteSource
    | WeightedPointSource
    | PointMass
    | PointMassCloud
    | VolumeElementSource
)
ModelSpec: TypeAlias = UniformSphereModelSpec | _StaticPointSumSpec


@dataclass(frozen=True, eq=False, init=False)
class PointSumModel:
    """Evaluate one immutable static point source in its explicit frame."""

    source: StaticPointInput
    spec: ModelSpec
    binding: _StaticPointSourceBinding | None

    def __init__(
        self,
        source: StaticPointInput,
        *,
        frame: Frame | None = None,
        field_definition: GravityFieldDefinition | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> None:
        if isinstance(source, DiscreteSource):
            if frame is not None and frame is not source.frame:
                raise ValueError("frame must be the DiscreteSource geometry.frame")
            if provenance is not None:
                raise ValueError(
                    "provenance is supported for bare point sources only; "
                    "DiscreteSource already carries discretization metadata"
                )
            spec = UniformSphereModelSpec(source.geometry, source.density)
            if field_definition is not None:
                definition = _validate_field_definition(field_definition)
                if definition != spec.field_definition:
                    raise ValueError(
                        "field_definition must preserve the DiscreteSource "
                        "Sphere meaning"
                    )
                object.__setattr__(spec, "field_definition", definition)
            object.__setattr__(self, "source", source)
            object.__setattr__(self, "spec", spec)
            object.__setattr__(self, "binding", None)
            return

        convenience_source = isinstance(source, (PointMass, PointMassCloud))
        if isinstance(source, (PointMass, PointMassCloud)):
            weighted = source.as_weighted_source()
        elif isinstance(source, WeightedPointSource):
            weighted = source
        elif isinstance(source, VolumeElementSource):
            weighted = source.as_weighted_source()
        else:
            raise TypeError(
                "source must be a DiscreteSource, WeightedPointSource, "
                "PointMass, PointMassCloud, or VolumeElementSource"
            )

        if frame is None:
            raise TypeError("frame must be explicitly provided for a bare point source")
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        if field_definition is None:
            if not convenience_source:
                raise TypeError(
                    "field_definition must be provided for WeightedPointSource "
                    "and VolumeElementSource"
                )
            field_definition = GravityFieldDefinition.source_field(
                type(source).__name__,
                Contribution("total"),
                PotentialReference.ZERO_AT_INFINITY,
            )
        definition = _validate_field_definition(field_definition)
        binding = _StaticPointSourceBinding(
            weighted.positions_m,
            weighted.weights_kg,
            frame,
            definition,
            type(source).__name__,
            {} if provenance is None else provenance,
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "spec", _StaticPointSumSpec(frame, definition))

    def at(
        self,
        targets: TargetPoints,
        *,
        observables: tuple[str, ...] = OBSERVABLES,
        output_frame: Frame | None = None,
    ):
        """Fix targets, requested fields, and output component frame."""
        from gragra.evaluation import EvaluationSetup

        resolved = ResolvedEvaluation(
            self,
            targets,
            (
                self.spec.geometry.frame
                if isinstance(self.spec, UniformSphereModelSpec)
                else self.spec.frame
            ),
            targets.frame if output_frame is None else output_frame,
            observables,
            self.source if self.binding is None else self.binding,
        )
        return EvaluationSetup(resolved)


def point_sum(
    source: StaticPointInput,
    *,
    frame: Frame | None = None,
    field_definition: GravityFieldDefinition | None = None,
    provenance: Mapping[str, object] | None = None,
) -> PointSumModel:
    """Create a static point-sum model.

    ``DiscreteSource`` supplies its geometry frame and Sphere field meaning.
    Bare sources require ``frame``; generic signed weights and volume elements
    also require ``field_definition``. ``PointMass`` and ``PointMassCloud``
    default to a total source field with potential zero at infinity. Optional
    provenance is an immutable snapshot and never establishes frame identity.
    """
    return PointSumModel(
        source,
        frame=frame,
        field_definition=field_definition,
        provenance=provenance,
    )


__all__ = ["PointSumModel", "point_sum"]
