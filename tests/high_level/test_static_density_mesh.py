"""Static high-level evaluation of regular-grid and volume-mesh density data."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gragra.adapters.meshio_adapter import VolumeMesh, to_volume_element_source
from gragra.constants import G_SI
from gragra.fields import DensityPerturbationGrid
from gragra.geometry import Frame
from gragra.gravity_sources import VolumeElementSource, point_sum
from gragra.observables import (
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from gragra.results import (
    ChangeDefinition,
    ChangeForm,
    Contribution,
    EvaluationResult,
    FieldKind,
    GravityFieldDefinition,
    PotentialReference,
    ReferenceDefinition,
    ReferenceStatus,
    ReferenceTemporalKind,
)
from gragra.targets import TargetPoints


def _field_change() -> GravityFieldDefinition:
    return GravityFieldDefinition.field_change(
        "density_perturbation",
        Contribution("total"),
        PotentialReference.ZERO_AT_INFINITY,
        ChangeDefinition(
            ChangeForm.FINITE_STATE_DIFFERENCE,
            ReferenceDefinition(
                ReferenceStatus.DEFINED,
                ReferenceTemporalKind.STATE,
                "reference_density",
            ),
            "density_perturbation",
        ),
    )


def _source_field(name: str) -> GravityFieldDefinition:
    return GravityFieldDefinition.source_field(
        name, Contribution("total"), PotentialReference.ZERO_AT_INFINITY
    )


def _grid_volume_source() -> VolumeElementSource:
    grid = DensityPerturbationGrid(
        x_m=np.array([0.0, 1.0]),
        y_m=np.array([0.0]),
        z_m=np.array([0.0]),
        delta_density_kg_m3=np.array([[[2.0]], [[-1.0]]]),
        cell_size_m=(1.0, 1.0, 1.0),
    )
    return grid.as_volume_element_source()


def _tetra_mesh() -> VolumeMesh:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return VolumeMesh(vertices, (("tetra", np.array([[0, 1, 2, 3]])),))


def _assert_result_contract(
    result: EvaluationResult,
    definition: GravityFieldDefinition,
    target_ids: tuple[str, ...],
    provenance: dict[str, object],
) -> None:
    assert isinstance(result, EvaluationResult)
    assert result.field_definition is definition
    assert result.field_definition.kind in {
        FieldKind.FIELD_CHANGE,
        FieldKind.SOURCE_FIELD,
    }
    assert (
        result.field_definition.potential_reference
        is PotentialReference.ZERO_AT_INFINITY
    )
    assert result.observables == ("potential", "acceleration", "gradient")
    assert result.potential.unit == u.m**2 / u.s**2
    assert result.acceleration.unit == u.m / u.s**2
    assert result.gradient.unit == u.s**-2
    assert result.g_applications == 1
    assert tuple(binding.target_ids for binding in result.bindings.values()) == (
        target_ids,
        target_ids,
        target_ids,
    )

    binding = result.evaluation.resolved.source
    assert dict(binding.provenance) == provenance
    with pytest.raises(TypeError):
        binding.provenance["added_after_binding"] = "no"
    assert not binding.positions_m.flags.writeable
    assert not binding.weights_kg.flags.writeable
    with pytest.raises(ValueError):
        binding.positions_m.setflags(write=True)
    with pytest.raises(ValueError):
        binding.weights_kg.setflags(write=True)
    assert result.parameters.meta["source_provenance"] == provenance

    selected = result.select("potential", "gradient")
    assert selected.observables == ("potential", "gradient")
    assert selected.evaluation is result.evaluation
    assert selected.field_definition is definition
    assert tuple(binding.target_ids for binding in selected.bindings.values()) == (
        target_ids,
        target_ids,
    )


def _assert_native_kernel_parity(
    result: EvaluationResult,
    weighted_source,
    source_frame_targets_m: np.ndarray,
) -> None:
    reference = (
        point_potential(weighted_source, source_frame_targets_m, backend="numpy"),
        point_acceleration(weighted_source, source_frame_targets_m, backend="numpy"),
        point_gravity_gradient(
            weighted_source, source_frame_targets_m, backend="numpy"
        ),
    )
    for actual, expected in zip(
        (
            result.potential.to_value(u.m**2 / u.s**2),
            result.acceleration.to_value(u.m / u.s**2),
            result.gradient.to_value(u.s**-2),
        ),
        reference,
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0.0)


def test_density_grid_static_result_signed_oracle_provenance() -> None:
    root = Frame("grid-world")
    source_frame = Frame(
        "grid-local", origin=np.array([100.0, 0.0, 0.0]) * u.m, parent=root
    )
    volume_source = _grid_volume_source()
    weighted_source = volume_source.as_weighted_source()
    np.testing.assert_array_equal(volume_source.positions_m, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(volume_source.volumes_m3, [1.0, 1.0])
    np.testing.assert_array_equal(weighted_source.weights_kg, [2.0, -1.0])
    assert weighted_source.weights_kg.sum() == 1.0

    provenance = {
        "coordinate_frame": "grid-local",
        "coordinate_origin_m": (100.0, 0.0, 0.0),
        "density_kind": "delta_density",
    }
    expected_provenance = provenance.copy()
    definition = _field_change()
    model = point_sum(
        volume_source,
        frame=source_frame,
        field_definition=definition,
        provenance=provenance,
    )
    provenance["density_kind"] = "caller-mutated-after-binding"
    assert model.source is volume_source
    assert model.binding.source_type == "VolumeElementSource"
    assert model.binding.frame is source_frame
    assert model.binding.weight_sum_kg == weighted_source.weights_kg.sum()
    np.testing.assert_array_equal(model.binding.positions_m, volume_source.positions_m)
    np.testing.assert_array_equal(model.binding.weights_kg, weighted_source.weights_kg)

    target_ids = ("grid-near", "grid-next")
    targets_root_m = np.array([[103.0, 0.0, 0.0], [104.0, 0.0, 0.0]])
    result = model.at(TargetPoints(targets_root_m * u.m, target_ids, root)).run()
    _assert_result_contract(result, definition, target_ids, expected_provenance)
    assert result.field_definition.kind is FieldKind.FIELD_CHANGE

    # Independent point-mass sum at target x=103 m: the cells have masses +2 kg
    # at x=100 m and -1 kg at x=101 m after the explicit Frame transform.
    expected_gradient = G_SI * np.diag([-11.0 / 108.0, 11.0 / 216.0, 11.0 / 216.0])
    np.testing.assert_allclose(
        result.potential.to_value(u.m**2 / u.s**2)[0],
        -G_SI / 6.0,
        rtol=2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.acceleration.to_value(u.m / u.s**2)[0],
        [G_SI / 36.0, 0.0, 0.0],
        rtol=2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.gradient.to_value(u.s**-2)[0],
        expected_gradient,
        rtol=2e-14,
        atol=0.0,
    )
    assert (
        np.count_nonzero(
            result.gradient.to_value(u.s**-2)[0]
            - np.diag(np.diag(result.gradient.to_value(u.s**-2)[0]))
        )
        == 0
    )

    # The adapter output is a centroid point-mass representation, so compare
    # against the same native weights passed to the existing numpy kernels.
    _assert_native_kernel_parity(
        result, weighted_source, np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    )


def test_volume_mesh_static_result_tetra_oracle_provenance() -> None:
    mesh = _tetra_mesh()
    volume_source = to_volume_element_source(mesh, density_kg_m3=6.0)
    weighted_source = volume_source.as_weighted_source()
    np.testing.assert_allclose(
        volume_source.positions_m, [[0.25, 0.25, 0.25]], rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        volume_source.volumes_m3, [1.0 / 6.0], rtol=2e-14, atol=0.0
    )
    np.testing.assert_allclose(weighted_source.weights_kg, [1.0], rtol=2e-14, atol=0.0)

    frame = Frame("tetra-local")
    provenance = {
        "coordinate_frame": "tetra-import-frame",
        "coordinate_origin_m": (0.0, 0.0, 0.0),
        "mesh_kind": "unit_tetrahedron",
    }
    expected_provenance = provenance.copy()
    definition = _source_field("tetra_volume_density")
    model = point_sum(
        volume_source,
        frame=frame,
        field_definition=definition,
        provenance=provenance,
    )
    provenance["mesh_kind"] = "caller-mutated-after-binding"
    assert model.source is volume_source
    assert model.binding.source_type == "VolumeElementSource"
    assert model.binding.frame is frame
    assert model.binding.weight_sum_kg == pytest.approx(1.0, rel=2e-14, abs=0.0)
    np.testing.assert_array_equal(model.binding.positions_m, volume_source.positions_m)
    np.testing.assert_allclose(model.binding.weights_kg, weighted_source.weights_kg)

    target_ids = ("tetra-centroid-plus-two",)
    target_m = np.array([[2.25, 0.25, 0.25]])
    result = model.at(TargetPoints(target_m * u.m, target_ids, frame)).run()
    _assert_result_contract(result, definition, target_ids, expected_provenance)
    assert result.field_definition.kind is FieldKind.SOURCE_FIELD

    # This oracle checks the discrete centroid-mass representation of the tetra,
    # not the continuous-body field accuracy of a centroid quadrature.
    np.testing.assert_allclose(
        result.potential.to_value(u.m**2 / u.s**2),
        [-G_SI / 2.0],
        rtol=2e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.acceleration.to_value(u.m / u.s**2),
        [[-G_SI / 4.0, 0.0, 0.0]],
        rtol=2e-14,
        atol=0.0,
    )
    expected_gradient = G_SI * np.diag([1.0 / 4.0, -1.0 / 8.0, -1.0 / 8.0])
    np.testing.assert_allclose(
        result.gradient.to_value(u.s**-2),
        [expected_gradient],
        rtol=2e-14,
        atol=0.0,
    )
    assert (
        np.count_nonzero(
            result.gradient.to_value(u.s**-2)[0]
            - np.diag(np.diag(result.gradient.to_value(u.s**-2)[0]))
        )
        == 0
    )

    _assert_native_kernel_parity(result, weighted_source, target_m)


@pytest.mark.parametrize(
    "source_factory",
    [_grid_volume_source, lambda: to_volume_element_source(_tetra_mesh(), 6.0)],
    ids=["density-grid", "volume-mesh"],
)
def test_external_frame_provenance_never_supplies_or_connects_frame(
    source_factory,
) -> None:
    source = source_factory()
    provenance = {
        "coordinate_frame": "caller-frame-name",
        "coordinate_origin_m": (12.0, -3.0, 5.0),
    }
    definition = _field_change()

    with pytest.raises(TypeError, match="frame must be explicitly provided"):
        point_sum(source, field_definition=definition, provenance=provenance)

    source_frame = Frame("caller-frame-name")
    disconnected_target_frame = Frame("caller-frame-name")
    model = point_sum(
        source,
        frame=source_frame,
        field_definition=definition,
        provenance=provenance,
    )
    assert model.binding.frame is source_frame
    assert dict(model.binding.provenance) == provenance
    with pytest.raises(ValueError, match="disconnected roots"):
        model.at(
            TargetPoints(
                [[20.0, 0.0, 0.0]] * u.m, ("disconnected",), disconnected_target_frame
            )
        )
