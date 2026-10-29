"""Hex8 metadata over the existing fixed-mesh-series HDF5 version 1.

The conversion constructor materializes its finite input series. HDF5 writing
and replay use the existing sample/source slabs; the static mesh remains
resident. The exchange stores interpolated quadrature values, not native
solver nodal output. No solver-native format or hidden frame transform is used.
"""

import json
from dataclasses import dataclass

import numpy as np

from gragra.adapters.field_series_hdf5 import (
    _h5py,
    read_fixed_mesh_series_hdf5,
)
from gragra.fields.displacement_series import FixedMeshDisplacementSeries
from gragra.fields.material_hex8 import (
    Hex8DisplacementField,
    MaterialRegions,
    _rule,
    _shape,
)

HEX8_METADATA_PROFILE = "gragra.hex8-q1"
HEX8_METADATA_VERSION = 1
HEX8_NODE_ORDER = "VTK:000,100,110,010,001,101,111,011"


def hex8_displacement_series(
    field,
    materials,
    displacement_m,
    axis_values,
    axis_kind,
    *,
    coordinate_origin_m=(0.0, 0.0, 0.0),
    t_ref_s=None,
    phase_origin_m=None,
    quadrature_order=3,
):
    """Interpolate a finite nodal time/frequency array into an existing series.

    Input has shape (samples, vertices, 3); output has (samples, Gauss points,
    3). This in-memory constructor is not a bounded nodal-file importer.
    Frequency inputs obey Re[u_hat exp(-2pi i f(t-t_ref_s))].
    """
    if not isinstance(field, Hex8DisplacementField):
        raise TypeError("field must be a Hex8DisplacementField")
    sources = field.sources(materials, quadrature_order=quadrature_order)
    values = np.asarray(displacement_m)
    if values.dtype.kind not in "iufc":
        raise TypeError("displacement_m must be numeric and non-boolean")
    if values.shape != (len(axis_values), len(field.vertices_m), 3):
        raise ValueError("displacement_m must have shape (samples,vertices,3)")
    if not np.isfinite(values).all():
        raise ValueError("displacement_m must be finite")
    if axis_kind == "time" and np.iscomplexobj(values):
        raise TypeError("time displacement must be real")
    shape = _shape(_rule(quadrature_order)[0])[0]
    interpolated = np.einsum("qn,scni->scqi", shape, values[:, field.hex8]).reshape(
        len(axis_values), -1, 3
    )
    metadata = {
        "profile": HEX8_METADATA_PROFILE,
        "version": HEX8_METADATA_VERSION,
        "cell_type": "hex8",
        "node_order": HEX8_NODE_ORDER,
        "unit_system": "SI",
        "density_convention": "nonnegative_background_rho",
        "representation": "mass_displacement",
        "quadrature_order": int(quadrature_order),
        "vertices_m": field.vertices_m,
        "hex8": field.hex8,
        "cell_ids": materials.cell_ids,
        "region_ids": materials.region_ids,
        "region_density": [
            [int(key), value]
            for key, value in materials.density_by_region_kg_m3.items()
        ],
        "exterior_density_kg_m3": materials.exterior_density_kg_m3,
    }
    return FixedMeshDisplacementSeries(
        sources.positions_m,
        sources.mass_kg,
        interpolated,
        np.asarray(axis_values),
        axis_kind,
        field.coordinate_frame,
        np.asarray(coordinate_origin_m),
        sources.source_ids,
        metadata,
        t_ref_s=t_ref_s,
        phase_origin_m=None if phase_origin_m is None else np.asarray(phase_origin_m),
    )


@dataclass(frozen=True)
class Hex8SeriesExchange:
    """Validated mesh metadata and replayable fixed-source HDF5 provider.

    ``geometry.displacement_m`` is zero: sample values live only in provider
    blocks and have already been interpolated to quadrature positions.
    """

    geometry: Hex8DisplacementField
    materials: MaterialRegions
    quadrature_order: int
    provider: object


def read_hex8_series_hdf5(path, *, metadata_source_block_size=8192):
    """Read the optional hex8 metadata profile without changing v1 readers.

    Validate geometry, region map, quadrature IDs/positions, and mass weights
    before exposing a provider. Numeric sample values remain HDF5-backed and
    are only read through bounded sample/source slabs.
    """
    provider = read_fixed_mesh_series_hdf5(
        path, metadata_source_block_size=metadata_source_block_size
    )
    if provider.metadata.quantity != "displacement":
        raise ValueError("hex8 exchange requires displacement series")
    with _h5py().File(path, "r") as h5:
        try:
            data = json.loads(h5["metadata"].attrs["source_metadata_json"])
            if not isinstance(data, dict):
                raise ValueError("hex8 metadata must be an object")
            expected = {
                "profile": HEX8_METADATA_PROFILE,
                "version": HEX8_METADATA_VERSION,
                "cell_type": "hex8",
                "node_order": HEX8_NODE_ORDER,
                "unit_system": "SI",
                "density_convention": "nonnegative_background_rho",
                "representation": "mass_displacement",
            }
            if any(data.get(k) != v for k, v in expected.items()):
                raise ValueError("unsupported hex8 metadata profile")
            if type(data["version"]) is not int:
                raise ValueError("hex8 metadata version must be an integer")
            density_pairs = data["region_density"]
            if len({pair[0] for pair in density_pairs}) != len(density_pairs):
                raise ValueError("hex8 metadata has duplicate region IDs")
            materials = MaterialRegions(
                data["cell_ids"],
                np.asarray(data["region_ids"]),
                dict(density_pairs),
                exterior_density_kg_m3=data["exterior_density_kg_m3"],
                coordinate_frame=provider.metadata.coordinate_frame,
                unit_system=data["unit_system"],
            )
            vertices = np.asarray(data["vertices_m"])
            field = Hex8DisplacementField(
                vertices,
                np.asarray(data["hex8"]),
                np.zeros_like(vertices),
                coordinate_frame=provider.metadata.coordinate_frame,
            )
            order = data["quadrature_order"]
            sources = field.sources(materials, quadrature_order=order)
        except (KeyError, TypeError, IndexError, json.JSONDecodeError) as error:
            raise ValueError("invalid hex8 metadata") from error
        if sources.source_ids != provider.metadata.source_ids or not np.array_equal(
            sources.positions_m, provider.metadata.positions_m
        ):
            raise ValueError("hex8 quadrature geometry or IDs differ from series")
        mass = h5["metadata/mass_kg"]
        for start in range(0, len(sources.mass_kg), metadata_source_block_size):
            stop = min(start + metadata_source_block_size, len(sources.mass_kg))
            if not np.array_equal(mass[start:stop], sources.mass_kg[start:stop]):
                raise ValueError("hex8 background mass differs from region density")
    return Hex8SeriesExchange(field, materials, order, provider)
