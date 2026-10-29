"""Evaluate static density data from a regular grid and a volume mesh.

Run with the project environment, for example::

    python examples/static_sources_workflow.py

Both adapters produce centroid point-mass representations. The integration
test checks the mesh result against its point-mass oracle; this example does not
measure continuous-tetrahedron quadrature accuracy.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u

from gragra.adapters.meshio_adapter import VolumeMesh, to_volume_element_source
from gragra.fields import DensityPerturbationGrid
from gragra.geometry import Frame
from gragra.gravity_sources import point_sum
from gragra.results import (
    ChangeDefinition,
    ChangeForm,
    Contribution,
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


def _source_field() -> GravityFieldDefinition:
    return GravityFieldDefinition.source_field(
        "tetrahedron_density",
        Contribution("total"),
        PotentialReference.ZERO_AT_INFINITY,
    )


def _show_result(label: str, result) -> None:
    selected = result.select("potential", "acceleration")
    print(f"{label}: {selected.field_definition.kind.value}")
    print(f"  target IDs: {next(iter(selected.bindings.values())).target_ids}")
    print(f"  potential: {selected.potential}")
    print(f"  acceleration: {selected.acceleration}")
    print(f"  source provenance: {selected.parameters.meta['source_provenance']}")


def evaluate_density_grid() -> None:
    root = Frame("grid-world")
    grid_frame = Frame(
        "grid-local", origin=np.array([100.0, 0.0, 0.0]) * u.m, parent=root
    )
    density_grid = DensityPerturbationGrid(
        x_m=np.array([0.0, 1.0]),
        y_m=np.array([0.0]),
        z_m=np.array([0.0]),
        delta_density_kg_m3=np.array([[[2.0]], [[-1.0]]]),
        cell_size_m=(1.0, 1.0, 1.0),
    )
    volume_source = density_grid.as_volume_element_source()
    provenance = {
        "coordinate_frame": "grid-local",
        "coordinate_origin_m": (100.0, 0.0, 0.0),
        "input": "regular-grid density perturbation",
    }
    model = point_sum(
        volume_source,
        frame=grid_frame,
        field_definition=_field_change(),
        provenance=provenance,
    )
    print(f"grid mass sum: {model.binding.weight_sum_kg} kg")
    result = model.at(
        TargetPoints(
            np.array([[103.0, 0.0, 0.0], [104.0, 0.0, 0.0]]) * u.m,
            ("grid-near", "grid-next"),
            root,
        )
    ).run()
    _show_result("regular grid", result)


def evaluate_volume_mesh() -> None:
    vertices_m = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mesh = VolumeMesh(vertices_m, (("tetra", np.array([[0, 1, 2, 3]])),))
    volume_source = to_volume_element_source(mesh, density_kg_m3=6.0)
    frame = Frame("mesh-local")
    provenance = {
        "coordinate_frame": "tetra-import-frame",
        "coordinate_origin_m": (0.0, 0.0, 0.0),
        "input": "unit tetrahedron volume mesh",
    }
    model = point_sum(
        volume_source,
        frame=frame,
        field_definition=_source_field(),
        provenance=provenance,
    )
    print(f"mesh mass sum: {model.binding.weight_sum_kg} kg")
    result = model.at(
        TargetPoints(
            np.array([[2.25, 0.25, 0.25]]) * u.m,
            ("tetra-centroid-plus-two",),
            frame,
        )
    ).run()
    _show_result("volume mesh", result)


if __name__ == "__main__":
    evaluate_density_grid()
    evaluate_volume_mesh()
