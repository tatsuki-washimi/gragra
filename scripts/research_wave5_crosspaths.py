#!/usr/bin/env python3
# ruff: noqa: E501
"""Wave 5 moving-density input-path comparison.

The exact-exchange panel uses all nine accepted OpenFOAM snapshots through
matched in-memory, HDF5, and GWexpy routes. The prescribed-bell panel compares
Eulerian and material grids to an independent continuous spherical reference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import resource
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
sys.path.insert(0, str(ROOT / "src"))

import gragra  # noqa: E402
from gragra.adapters import (  # noqa: E402
    from_gwexpy_density_series,
    from_gwexpy_displacement_series,
    from_gwexpy_surface_density_series,
    load_openfoam_density,
    read_fixed_mesh_series_hdf5,
    to_gwexpy_density_series,
    to_gwexpy_displacement_series,
    to_gwexpy_surface_density_series,
    write_fixed_mesh_series_hdf5,
)
from gragra.fields import (  # noqa: E402
    DensityTimeSeries,
    FixedMeshDisplacementSeries,
    FixedMeshSurfaceDensitySeries,
)
from gragra.gravity_sources import WeightedPointSource  # noqa: E402
from gragra.integrators.streaming import as_series_provider  # noqa: E402
from gragra.observables import (  # noqa: E402
    collect_observable_blocks,
    point_acceleration,
    point_gravity_gradient,
    point_potential,
    stream_observable,
)
from gragra.observables.wave456_reference import point_mass_observables  # noqa: E402

OBSERVABLES = ("potential", "acceleration", "gradient")
SAME_DISCRETE_GATE = 1e-11
ANALYTIC_REFERENCE_SHA256 = (
    "bda1d6075ad042511d51624d8bca8e9b244d7e7063f401c12f2c4eba4cee11be"
)
ANALYTIC_WRAPPER_SHA256 = (
    "7ebe9850fa6fa242fcc1be3f3635c6436c3d60c8bd3878a75423ba6c7701af47"
)
E_DELIVERY_SHA256 = "e146d87b8b132befe5bb4e181f8ab1aa3c9476abdc4e5808767a9fa3889ac297"
E_TRANSPORT_MANIFEST_SHA256 = (
    "28e4ef6255e8321c3e1ad0ef42ddcd04f0ebb5b17c79b9b370e6e05a847f5dc0"
)
E_ROOT = Path("/home/washimi/work/gragra-research/2026-09-12-wave4-6/E")
E_DELIVERY = E_ROOT / "delivery.json"
E_TRANSPORT_ROOT = E_ROOT / "wave5-transport-v3"
E_TRANSPORT_MANIFEST = E_TRANSPORT_ROOT / "transport-manifest.json"
E_CASES = ("n24-dt0.01", "n48-dt0.01", "n48-dt0.005")
E_TIMES = ("0", "0.5", "1")
E_TARGETS_M = np.array(
    [[3.0, 0.7, -0.4], [-2.8, 1.3, 1.7], [0.25, -3.0, 2.6]],
    dtype=np.float64,
)
EXECUTED_MODULE_NAMES = (
    "gragra",
    "gragra.constants",
    "gragra.gravity_sources.weighted_points",
    "gragra.fields._series",
    "gragra.fields.density_series",
    "gragra.adapters.openfoam_density",
    "gragra.adapters.field_series_hdf5",
    "gragra.adapters._gwexpy_series",
    "gragra.adapters.gwexpy_density_series",
    "gragra.integrators.point_direct",
    "gragra.integrators.streaming",
    "gragra.kernels.inverse_square",
    "gragra.observables.potential",
    "gragra.observables.acceleration",
    "gragra.observables.gradient",
    "gragra.observables.streaming",
    "gragra.analytic.wave456_reference",
    "gragra.observables.wave456_reference",
)


class StudyAcceptanceError(RuntimeError):
    """The completed payload does not meet the study's publication gates."""

    def __init__(self, findings):
        self.findings = tuple(findings)
        super().__init__("; ".join(self.findings))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _observable_dict(value):
    if isinstance(value, dict):
        return value
    return dict(zip(OBSERVABLES, value, strict=True))


def _raw_observables(values):
    return {name: np.asarray(values[name]).tolist() for name in OBSERVABLES}


def _bell_grid(resolution: int):
    edges = np.linspace(-0.5, 0.5, resolution + 1)
    axis = (edges[:-1] + edges[1:]) / 2
    positions = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    return positions, np.full(len(positions), (1.0 / resolution) ** 3)


def _bell_density(positions, center_x):
    ratio = np.linalg.norm(np.asarray(positions) - [center_x, 0.0, 0.0], axis=1) / 0.15
    return np.where(ratio < 1.0, -0.005 * (1.0 + np.cos(np.pi * ratio)), 0.0)


def _absolute_observables(positions, weights, targets):
    source = WeightedPointSource(positions, weights)
    return {
        "potential": point_potential(source, targets, chunk_size=1024),
        "acceleration": point_acceleration(source, targets, chunk_size=1024),
        "gradient": point_gravity_gradient(source, targets, chunk_size=1024),
    }


def _physical_norm(values, observable):
    if observable == "potential":
        return np.abs(values)
    if observable == "acceleration":
        return np.linalg.norm(values, axis=-1)
    return np.sqrt(np.sum(np.abs(values) ** 2, axis=(-2, -1)))


def _comparison_metrics(candidate, reference, *, physical_scales=None):
    """Return physical norms without collapsing sample or target axes."""
    if physical_scales is not None and set(physical_scales) != set(OBSERVABLES):
        raise ValueError("physical_scales must provide all three observables")
    result = {}
    for observable in OBSERVABLES:
        candidate_array = np.asarray(candidate[observable])
        reference_array = np.asarray(reference[observable])
        if candidate_array.shape != reference_array.shape:
            raise ValueError(f"{observable} candidate/reference shapes differ")
        absolute = _physical_norm(candidate_array - reference_array, observable)
        reference_norm = _physical_norm(reference_array, observable)
        largest_reference = float(np.max(reference_norm, initial=0.0))
        control_scale = (
            float(physical_scales[observable])
            if physical_scales is not None
            else largest_reference
        )
        if not np.isfinite(control_scale) or control_scale <= 0.0:
            raise ValueError(
                f"{observable} requires a positive physical scale when its reference is zero"
            )
        small_reference_threshold = control_scale * np.finfo(np.float64).eps
        scale = np.where(
            reference_norm > small_reference_threshold,
            reference_norm,
            control_scale,
        )
        normalized = absolute / scale
        result[observable] = {
            "absolute_norm": absolute.tolist(),
            "reference_norm": reference_norm.tolist(),
            "normalization_scale": scale.tolist(),
            "zero_or_small_reference_physical_scale": control_scale,
            "small_reference_threshold": small_reference_threshold,
            "normalized": normalized.tolist(),
            "summary": {
                "maximum_absolute_norm": float(np.max(absolute, initial=0.0)),
                "maximum_normalized": float(np.max(normalized, initial=0.0)),
            },
        }
    return result


def moving_bell_comparison(*, resolution: int):
    """Absolute Eulerian-versus-material gravity for one translated 3-D bell."""
    positions, volumes = _bell_grid(resolution)
    targets = np.array([[1.0, 0.2, 0.3], [-1.0, -0.2, 0.4]])
    times = np.array([0.0, 0.5, 1.0])
    initial_weights = _bell_density(positions, -0.25) * volumes
    analytic_mass = float(2.0 * -0.01 * 0.15**3 * (np.pi**2 - 6.0) / (3.0 * np.pi))
    records = []
    eulerian_masses = []
    material_masses = []
    initial_material = None
    for time_s in times:
        center_x = -0.25 + 0.2 * time_s
        eulerian_weights = _bell_density(positions, center_x) * volumes
        eulerian = _absolute_observables(positions, eulerian_weights, targets)
        material_positions = positions + [0.2 * time_s, 0.0, 0.0]
        material_weights = initial_weights
        material = _absolute_observables(material_positions, material_weights, targets)
        continuous_value = point_mass_observables(
            np.array([[center_x, 0.0, 0.0]]),
            np.array([analytic_mass]),
            targets,
        )
        continuous = _observable_dict(continuous_value)
        if initial_material is None:
            initial_material = material
        eulerian_masses.append(float(np.sum(eulerian_weights)))
        material_masses.append(float(np.sum(material_weights)))
        records.append(
            {
                "time_s": float(time_s),
                "center_m": [center_x, 0.0, 0.0],
                "mass_kg": {
                    "analytic": analytic_mass,
                    "eulerian": eulerian_masses[-1],
                    "material": material_masses[-1],
                },
                "observables": {
                    "eulerian": _raw_observables(eulerian),
                    "material": _raw_observables(material),
                    "continuous": _raw_observables(continuous),
                },
                "comparisons": {
                    "eulerian_vs_material": _comparison_metrics(eulerian, material),
                    "eulerian_vs_continuous": _comparison_metrics(eulerian, continuous),
                    "material_vs_continuous": _comparison_metrics(material, continuous),
                },
            }
        )
    material_mass_array = np.asarray(material_masses)
    return {
        "resolution": resolution,
        "comparison_kind": "different finite-grid discretizations of a translating radial cosine bell",
        "material_model": "signed t0 point masses translated in position; not a displacement transfer",
        "continuous_reference_scope": "exact exterior field of the prescribed translated spherical bell; not the evolved CFD density",
        "analytic_mass_kg": analytic_mass,
        "analytic_reference": {
            "module_path": str(
                (ROOT / "src/gragra/analytic/wave456_reference.py").resolve()
            ),
            "module_sha256": ANALYTIC_REFERENCE_SHA256,
            "wrapper_path": str(
                (ROOT / "src/gragra/observables/wave456_reference.py").resolve()
            ),
            "wrapper_sha256": ANALYTIC_WRAPPER_SHA256,
            "formula": "2*A*R^3*(pi^2-6)/(3*pi)",
            "radius_m": 0.15,
            "amplitude_kg_m3": -0.01,
        },
        "eulerian_mass_kg_by_time": eulerian_masses,
        "material_mass_kg_by_time": material_masses,
        "material_mass_range_kg": float(np.ptp(material_mass_array)),
        "material_mass_constant": bool(
            np.array_equal(
                material_mass_array,
                np.full(len(material_mass_array), material_mass_array[0]),
            )
        ),
        "t0_eulerian_vs_material_normalized": {
            name: records[0]["comparisons"]["eulerian_vs_material"][name]["summary"][
                "maximum_normalized"
            ]
            for name in OBSERVABLES
        },
        "nonzero_translation_change": bool(
            np.linalg.norm(
                np.asarray(records[-1]["observables"]["material"]["acceleration"])
                - np.asarray(records[0]["observables"]["material"]["acceleration"])
            )
            > 0.0
        ),
        "per_time_per_target": records,
    }


def build_small_crosspath_inputs():
    """Return a deterministic, non-colliding density/material/surface panel."""
    positions = np.array(
        [[-0.75, 0.0, 0.0], [-0.25, 0.0, 0.0], [0.25, 0.0, 0.0], [0.75, 0.0, 0.0]]
    )
    times = np.array([0.0, 0.5, 1.0])
    # Eulerian samples of a translating radius-0.75 cosine bell.
    centers = -0.30 + 0.20 * times
    radius = 0.75
    density_values = np.array(
        [
            0.5
            * (
                1.0
                + np.cos(np.pi * np.minimum(np.abs(positions[:, 0] - c) / radius, 1.0))
            )
            for c in centers
        ]
    )
    density = DensityTimeSeries(
        positions,
        np.full(4, 0.125),
        density_values,
        times,
        "delta_rho",
        "study",
        np.zeros(3),
        tuple(f"cell-{i}" for i in range(4)),
        {"model": "Eulerian translating cosine bell"},
    )
    displacement = FixedMeshDisplacementSeries(
        positions,
        density_values[0] * 0.125,
        np.stack(
            [np.broadcast_to([c - centers[0], 0.0, 0.0], (4, 3)) for c in centers]
        ),
        times,
        "time",
        "study",
        np.zeros(3),
        tuple(f"material-{i}" for i in range(4)),
        {"model": "Lagrangian material points"},
    )
    surface = FixedMeshSurfaceDensitySeries(
        positions,
        np.full(4, 0.125),
        np.tile([0.0, 0.0, 1.0], (4, 1)),
        density_values,
        times,
        "time",
        {"representation": "signed sheet; not added to bulk"},
        "study",
        np.zeros(3),
        tuple(f"face-{i}" for i in range(4)),
    )
    return (
        density,
        displacement,
        surface,
        np.array([[2.0, 0.2, 0.3], [-2.2, -0.1, 0.4]]),
    )


def _evaluate(series, targets):
    provider = series if hasattr(series, "blocks") else as_series_provider(series)
    source_block_size = min(8192, provider.source_count)
    target_block_size = min(16, len(targets))
    result = {}
    for observable in OBSERVABLES:
        blocks = stream_observable(
            provider,
            targets,
            observable=observable,
            sample_block_size=1,
            source_block_size=source_block_size,
            target_block_size=target_block_size,
            kernel_chunk_size=min(1024, source_block_size),
        )
        result[observable] = collect_observable_blocks(
            blocks,
            sample_count=provider.sample_count,
            target_count=len(targets),
            axis_kind=provider.axis_kind,
        )
    return result


def _observable_difference(candidate, reference):
    return {
        name: (np.asarray(candidate[name]) - np.asarray(reference[name])).tolist()
        for name in OBSERVABLES
    }


def hdf5_density_exchange(series, targets, hdf5_path: Path, *, direct=None):
    """Run one density series through the bounded HDF5 exchange."""
    import h5py

    direct_values = _evaluate(series, targets) if direct is None else direct
    source_block_size = min(8192, series.source_count)
    write_fixed_mesh_series_hdf5(
        hdf5_path,
        series,
        sample_block_size=1,
        source_block_size=source_block_size,
    )
    provider = read_fixed_mesh_series_hdf5(
        hdf5_path, metadata_source_block_size=min(8192, series.source_count)
    )
    hdf_values = _evaluate(provider, targets)
    with h5py.File(hdf5_path, "r") as h5:
        metadata = h5["metadata"]
        stored_positions = np.asarray(metadata["positions_m"])
        stored_volumes = np.asarray(metadata["volumes_m3"])
        stored_values = np.asarray(h5["values"])
        stored_ids = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in metadata["source_ids"][:]
        )
        stored_axis = np.asarray(metadata["axis_values"])
        stored_origin = np.asarray(metadata["coordinate_origin_m"])
        stored_density_kind = _decode(metadata.attrs["density_kind"])
        stored_axis_kind = _decode(metadata.attrs["axis_kind"])
        stored_coordinate_frame = _decode(metadata.attrs["coordinate_frame"])
        expected_axis_unit = "s" if series.axis_kind == "time" else "Hz"
        stored_units_equal = all(
            (
                _decode(metadata["positions_m"].attrs.get("unit", "")) == "m",
                _decode(metadata["coordinate_origin_m"].attrs.get("unit", "")) == "m",
                _decode(metadata["source_ids"].attrs.get("unit", "")) == "1",
                _decode(metadata["axis_values"].attrs.get("unit", ""))
                == expected_axis_unit,
                _decode(metadata["volumes_m3"].attrs.get("unit", "")) == "m3",
                _decode(h5["values"].attrs.get("unit", "")) == "kg/m3",
            )
        )
    return {
        "hdf5_path": str(hdf5_path),
        "hdf5_sha256": _sha256(hdf5_path),
        "metadata_equal": {
            "geometry": bool(
                np.array_equal(stored_positions, series.positions_m)
                and np.array_equal(provider.positions_m, series.positions_m)
            ),
            "volumes": bool(np.array_equal(stored_volumes, series.volumes_m3)),
            "ids": stored_ids == series.source_ids
            and provider.source_ids == series.source_ids,
            "axis": bool(
                np.array_equal(stored_axis, series.axis_values)
                and np.array_equal(provider.axis_values, series.axis_values)
            ),
            "axis_kind": bool(
                stored_axis_kind == series.axis_kind
                and provider.axis_kind == series.axis_kind
            ),
            "coordinate_frame": bool(
                stored_coordinate_frame == series.coordinate_frame
                and provider.coordinate_frame == series.coordinate_frame
            ),
            "coordinate_origin": bool(
                np.array_equal(stored_origin, series.coordinate_origin_m)
                and np.array_equal(
                    provider.coordinate_origin_m, series.coordinate_origin_m
                )
            ),
            "density_kind": bool(
                stored_density_kind == series.density_kind
                and provider.density_kind == series.density_kind
            ),
            "units": stored_units_equal,
            "values": bool(np.array_equal(stored_values, series.density_kg_m3)),
        },
        "array_sha256": {
            "positions_m": _array_sha256(stored_positions),
            "volumes_m3": _array_sha256(stored_volumes),
            "density_kg_m3": _array_sha256(stored_values),
        },
        "observables": {
            "direct": _raw_observables(direct_values),
            "hdf5": _raw_observables(hdf_values),
        },
        "raw_difference": _observable_difference(hdf_values, direct_values),
        "comparison": _comparison_metrics(hdf_values, direct_values),
    }


def gwexpy_density_exchange(
    series,
    targets,
    *,
    direct=None,
    current_value_probe: tuple[int, int, float] | None = None,
):
    """Run one density series through native GWexpy and inspect current values."""
    from astropy import units

    direct_values = _evaluate(series, targets) if direct is None else direct
    bundle = to_gwexpy_density_series(series)
    restored = from_gwexpy_density_series(bundle)
    native_values = _evaluate(restored, targets)
    companion = bundle.companion
    field = bundle.field
    expected_axis_unit = units.s if series.axis_kind == "time" else units.Hz
    native_units_equal = bool(
        field.unit == units.Unit("kg/m^3")
        and field._axis0_index.unit == expected_axis_unit
        and all(
            getattr(field, f"_axis{index}_index").unit == units.dimensionless_unscaled
            for index in (1, 2, 3)
        )
    )
    record = {
        "gwexpy_version": importlib.metadata.version("gwexpy"),
        "metadata_equal": {
            "geometry": bool(
                np.array_equal(companion["positions_m"], series.positions_m)
                and np.array_equal(restored.positions_m, series.positions_m)
            ),
            "volumes": bool(
                np.array_equal(companion["volumes_m3"], series.volumes_m3)
                and np.array_equal(restored.volumes_m3, series.volumes_m3)
            ),
            "ids": companion["source_ids"] == series.source_ids
            and restored.source_ids == series.source_ids,
            "axis": bool(
                np.array_equal(companion["axis_values"], series.axis_values)
                and np.array_equal(restored.axis_values, series.axis_values)
                and np.array_equal(
                    field._axis0_index.to_value(expected_axis_unit),
                    series.axis_values,
                )
            ),
            "axis_kind": bool(
                companion["axis_kind"] == series.axis_kind
                and restored.axis_kind == series.axis_kind
                and field.axis0_domain == series.axis_kind
            ),
            "coordinate_frame": bool(
                companion["coordinate_frame"] == series.coordinate_frame
                and restored.coordinate_frame == series.coordinate_frame
            ),
            "coordinate_origin": bool(
                np.array_equal(
                    companion["coordinate_origin_m"], series.coordinate_origin_m
                )
                and np.array_equal(
                    restored.coordinate_origin_m, series.coordinate_origin_m
                )
            ),
            "density_kind": companion["density_kind"] == series.density_kind
            and restored.density_kind == series.density_kind,
            "units": native_units_equal,
            "values": bool(
                np.array_equal(
                    np.asarray(field.value)[:, :, 0, 0], series.density_kg_m3
                )
                and np.array_equal(restored.density_kg_m3, series.density_kg_m3)
            ),
        },
        "array_sha256": {
            "positions_m": _array_sha256(restored.positions_m),
            "volumes_m3": _array_sha256(restored.volumes_m3),
            "density_kg_m3": _array_sha256(restored.density_kg_m3),
        },
        "observables": {
            "direct": _raw_observables(direct_values),
            "gwexpy": _raw_observables(native_values),
        },
        "raw_difference": _observable_difference(native_values, direct_values),
        "comparison": _comparison_metrics(native_values, direct_values),
    }
    if current_value_probe is not None:
        sample, source, delta = current_value_probe
        before = float(bundle.field.value[sample, source, 0, 0])
        bundle.field.value[sample, source, 0, 0] = before + delta
        mutated = from_gwexpy_density_series(bundle)
        mutated_values = _evaluate(mutated, targets)
        record["native_current_value_probe"] = {
            "sample": sample,
            "source": source,
            "delta_kg_m3": delta,
            "value_before_kg_m3": before,
            "value_after_kg_m3": float(mutated.density_kg_m3[sample, source]),
            "restored_density_matches_mutated_native_value": bool(
                mutated.density_kg_m3[sample, source] == before + delta
            ),
            "observable_changed": {
                name: bool(
                    np.any(
                        np.asarray(mutated_values[name])
                        != np.asarray(native_values[name])
                    )
                )
                for name in OBSERVABLES
            },
            "observables_after_mutation": _raw_observables(mutated_values),
            "change_from_unmutated": _comparison_metrics(mutated_values, native_values),
        }
    return record


def same_discrete_exchange(
    density,
    targets,
    directory: Path,
    name: str = "density",
    *,
    current_value_probe: tuple[int, int, float] | None = None,
):
    """Compare direct, HDF5, and native GWexpy for one discrete density series."""
    direct = _evaluate(density, targets)
    hdf = hdf5_density_exchange(
        density, targets, directory / f"{name}.h5", direct=direct
    )
    native = gwexpy_density_exchange(
        density,
        targets,
        direct=direct,
        current_value_probe=current_value_probe,
    )
    return {
        "comparison_kind": "same discrete input exchange",
        "hdf5": hdf,
        "gwexpy": native,
        "all_metadata_equal": all(hdf["metadata_equal"].values())
        and all(native["metadata_equal"].values()),
        "maximum_normalized_difference": {
            observable: max(
                hdf["comparison"][observable]["summary"]["maximum_normalized"],
                native["comparison"][observable]["summary"]["maximum_normalized"],
            )
            for observable in OBSERVABLES
        },
    }


def _phase_metadata_equal(series, *, stored, restored):
    if series.axis_kind == "time":
        return bool(
            stored.get("t_ref_s") is None
            and stored.get("phase_origin_m") is None
            and stored.get("fourier_convention") is None
            and getattr(restored, "t_ref_s", None) is None
            and getattr(restored, "phase_origin_m", None) is None
            and getattr(restored, "fourier_convention", None) is None
        )
    return bool(
        stored.get("t_ref_s") == series.t_ref_s
        and np.array_equal(stored.get("phase_origin_m"), series.phase_origin_m)
        and stored.get("fourier_convention") == series.fourier_convention
        and getattr(restored, "t_ref_s", None) == series.t_ref_s
        and np.array_equal(
            getattr(restored, "phase_origin_m", None), series.phase_origin_m
        )
        and getattr(restored, "fourier_convention", series.fourier_convention)
        == series.fourier_convention
    )


def _hdf5_series_metadata_equal(series, provider, hdf5_path, quantity):
    import h5py

    with h5py.File(hdf5_path, "r") as h5:
        metadata = h5["metadata"]
        stored_ids = tuple(_decode(value) for value in metadata["source_ids"][:])
        stored = {
            "t_ref_s": metadata.attrs.get("t_ref_s"),
            "phase_origin_m": (
                np.asarray(metadata["phase_origin_m"])
                if "phase_origin_m" in metadata
                else None
            ),
            "fourier_convention": (
                _decode(metadata.attrs["fourier_convention"])
                if "fourier_convention" in metadata.attrs
                else None
            ),
        }
        expected_axis_unit = "s" if series.axis_kind == "time" else "Hz"
        value_unit = "m" if quantity == "displacement" else "kg/m2"
        geometry_name = "mass_kg" if quantity == "displacement" else "areas_m2"
        geometry_unit = "kg" if quantity == "displacement" else "m2"
        result = {
            "positions": bool(
                np.array_equal(metadata["positions_m"][:], series.positions_m)
                and np.array_equal(provider.positions_m, series.positions_m)
            ),
            "ids": stored_ids == series.source_ids
            and provider.source_ids == series.source_ids,
            "axis": bool(
                np.array_equal(metadata["axis_values"][:], series.axis_values)
                and np.array_equal(provider.axis_values, series.axis_values)
            ),
            "axis_kind": bool(
                _decode(metadata.attrs["axis_kind"]) == series.axis_kind
                and provider.axis_kind == series.axis_kind
            ),
            "coordinate_frame": bool(
                _decode(metadata.attrs["coordinate_frame"]) == series.coordinate_frame
                and provider.coordinate_frame == series.coordinate_frame
            ),
            "coordinate_origin": bool(
                np.array_equal(
                    metadata["coordinate_origin_m"][:], series.coordinate_origin_m
                )
                and np.array_equal(
                    provider.coordinate_origin_m, series.coordinate_origin_m
                )
            ),
            "quantity": bool(
                _decode(h5.attrs["quantity"]) == quantity
                and provider.quantity == quantity
            ),
            "phase": _phase_metadata_equal(series, stored=stored, restored=provider),
            "units": all(
                (
                    _decode(metadata["positions_m"].attrs.get("unit", "")) == "m",
                    _decode(metadata["coordinate_origin_m"].attrs.get("unit", ""))
                    == "m",
                    _decode(metadata["source_ids"].attrs.get("unit", "")) == "1",
                    _decode(metadata["axis_values"].attrs.get("unit", ""))
                    == expected_axis_unit,
                    _decode(metadata[geometry_name].attrs.get("unit", ""))
                    == geometry_unit,
                    _decode(h5["values"].attrs.get("unit", "")) == value_unit,
                )
            ),
        }
        if quantity == "displacement":
            result["background_mass"] = bool(
                np.array_equal(metadata["mass_kg"][:], series.mass_kg)
            )
            result["values"] = bool(
                np.array_equal(h5["values"][:], series.displacement_m)
            )
        else:
            result["areas"] = bool(
                np.array_equal(metadata["areas_m2"][:], series.areas_m2)
            )
            result["normals"] = bool(
                np.array_equal(metadata["normals"][:], series.normals)
                and _decode(metadata["normals"].attrs.get("unit", "")) == "1"
            )
            result["values"] = bool(
                np.array_equal(h5["values"][:], series.surface_density_kg_m2)
            )
    return result


def _gwexpy_series_metadata_equal(series, bundle, restored, quantity):
    from astropy import units

    companion = bundle.companion
    field = bundle.field
    components = (
        tuple(field[key] for key in ("x", "y", "z"))
        if quantity == "displacement"
        else (field,)
    )
    expected_axis_unit = units.s if series.axis_kind == "time" else units.Hz
    expected_value_unit = (
        units.m if quantity == "displacement" else units.Unit("kg/m^2")
    )
    native_values = (
        np.stack(
            [np.asarray(component.value)[:, :, 0, 0] for component in components],
            axis=-1,
        )
        if quantity == "displacement"
        else np.asarray(field.value)[:, :, 0, 0]
    )
    expected_values = (
        series.displacement_m
        if quantity == "displacement"
        else series.surface_density_kg_m2
    )
    stored = {
        "t_ref_s": companion.get("t_ref_s"),
        "phase_origin_m": companion.get("phase_origin_m"),
        "fourier_convention": companion.get("fourier_convention"),
    }
    result = {
        "positions": bool(
            np.array_equal(companion["positions_m"], series.positions_m)
            and np.array_equal(restored.positions_m, series.positions_m)
        ),
        "ids": companion["source_ids"] == series.source_ids
        and restored.source_ids == series.source_ids,
        "axis": bool(
            np.array_equal(companion["axis_values"], series.axis_values)
            and np.array_equal(restored.axis_values, series.axis_values)
            and all(
                np.array_equal(
                    component._axis0_index.to_value(expected_axis_unit),
                    series.axis_values,
                )
                for component in components
            )
        ),
        "axis_kind": bool(
            companion["axis_kind"] == series.axis_kind
            and restored.axis_kind == series.axis_kind
            and all(
                component.axis0_domain == series.axis_kind for component in components
            )
        ),
        "coordinate_frame": bool(
            companion["coordinate_frame"] == series.coordinate_frame
            and restored.coordinate_frame == series.coordinate_frame
        ),
        "coordinate_origin": bool(
            np.array_equal(companion["coordinate_origin_m"], series.coordinate_origin_m)
            and np.array_equal(restored.coordinate_origin_m, series.coordinate_origin_m)
        ),
        "quantity": companion["quantity"] == quantity,
        "phase": _phase_metadata_equal(series, stored=stored, restored=restored),
        "units": bool(
            all(component.unit == expected_value_unit for component in components)
            and all(
                component._axis0_index.unit == expected_axis_unit
                for component in components
            )
            and all(
                getattr(component, f"_axis{index}_index").unit
                == units.dimensionless_unscaled
                for component in components
                for index in (1, 2, 3)
            )
        ),
        "values": bool(
            np.array_equal(native_values, expected_values)
            and np.array_equal(
                restored.displacement_m
                if quantity == "displacement"
                else restored.surface_density_kg_m2,
                expected_values,
            )
        ),
    }
    if quantity == "displacement":
        result["background_mass"] = bool(
            np.array_equal(companion["mass_kg"], series.mass_kg)
            and np.array_equal(restored.mass_kg, series.mass_kg)
        )
    else:
        result["areas"] = bool(
            np.array_equal(companion["areas_m2"], series.areas_m2)
            and np.array_equal(restored.areas_m2, series.areas_m2)
        )
        result["normals"] = bool(
            np.array_equal(companion["normals"], series.normals)
            and np.array_equal(restored.normals, series.normals)
        )
    return result


def _small_series_metadata_observed(
    series, provider, hdf5_path, bundle, restored, quantity
):
    """Return the small raw metadata/value records used by equality checks."""
    import h5py

    value_name = (
        "displacement_m" if quantity == "displacement" else "surface_density_kg_m2"
    )
    geometry_name = "mass_kg" if quantity == "displacement" else "areas_m2"

    def phase_record(value):
        return {
            "t_ref_s": getattr(value, "t_ref_s", None),
            "phase_origin_m": (
                np.asarray(value.phase_origin_m).tolist()
                if getattr(value, "phase_origin_m", None) is not None
                else None
            ),
            "fourier_convention": getattr(value, "fourier_convention", None),
        }

    def series_record(value):
        record = {
            "positions_m": np.asarray(value.positions_m).tolist(),
            "source_ids": list(value.source_ids),
            "axis_values": np.asarray(value.axis_values).tolist(),
            "axis_kind": value.axis_kind,
            "coordinate_frame": value.coordinate_frame,
            "coordinate_origin_m": np.asarray(value.coordinate_origin_m).tolist(),
            "quantity": quantity,
            "phase": phase_record(value),
            geometry_name: np.asarray(getattr(value, geometry_name)).tolist(),
            "values": np.asarray(getattr(value, value_name)).tolist(),
        }
        if quantity == "surface_density":
            record["normals"] = np.asarray(value.normals).tolist()
        return record

    companion = bundle.companion
    field = bundle.field
    components = (
        tuple(field[key] for key in ("x", "y", "z"))
        if quantity == "displacement"
        else (field,)
    )
    native_values = (
        np.stack(
            [np.asarray(component.value)[:, :, 0, 0] for component in components],
            axis=-1,
        )
        if quantity == "displacement"
        else np.asarray(field.value)[:, :, 0, 0]
    )
    native_record = {
        "positions_m": np.asarray(companion["positions_m"]).tolist(),
        "source_ids": list(companion["source_ids"]),
        "axis_values": np.asarray(companion["axis_values"]).tolist(),
        "axis_kind": companion["axis_kind"],
        "coordinate_frame": companion["coordinate_frame"],
        "coordinate_origin_m": np.asarray(companion["coordinate_origin_m"]).tolist(),
        "quantity": companion["quantity"],
        "phase": {
            "t_ref_s": companion.get("t_ref_s"),
            "phase_origin_m": (
                np.asarray(companion["phase_origin_m"]).tolist()
                if companion.get("phase_origin_m") is not None
                else None
            ),
            "fourier_convention": companion.get("fourier_convention"),
        },
        geometry_name: np.asarray(companion[geometry_name]).tolist(),
        "values": native_values.tolist(),
        "native_units": {
            "values": [str(component.unit) for component in components],
            "axis0": [str(component._axis0_index.unit) for component in components],
            "axis1": [str(component._axis1_index.unit) for component in components],
            "axis2": [str(component._axis2_index.unit) for component in components],
            "axis3": [str(component._axis3_index.unit) for component in components],
        },
    }
    if quantity == "surface_density":
        native_record["normals"] = np.asarray(companion["normals"]).tolist()

    with h5py.File(hdf5_path, "r") as h5:
        metadata = h5["metadata"]
        hdf_record = {
            "positions_m": metadata["positions_m"][:].tolist(),
            "source_ids": [_decode(value) for value in metadata["source_ids"][:]],
            "axis_values": metadata["axis_values"][:].tolist(),
            "axis_kind": _decode(metadata.attrs["axis_kind"]),
            "coordinate_frame": _decode(metadata.attrs["coordinate_frame"]),
            "coordinate_origin_m": metadata["coordinate_origin_m"][:].tolist(),
            "quantity": _decode(h5.attrs["quantity"]),
            "phase": {
                "t_ref_s": metadata.attrs.get("t_ref_s"),
                "phase_origin_m": (
                    metadata["phase_origin_m"][:].tolist()
                    if "phase_origin_m" in metadata
                    else None
                ),
                "fourier_convention": (
                    _decode(metadata.attrs["fourier_convention"])
                    if "fourier_convention" in metadata.attrs
                    else None
                ),
            },
            geometry_name: metadata[geometry_name][:].tolist(),
            "values": h5["values"][:].tolist(),
            "stored_units": {
                name: _decode(dataset.attrs.get("unit", ""))
                for name, dataset in (
                    ("positions_m", metadata["positions_m"]),
                    ("coordinate_origin_m", metadata["coordinate_origin_m"]),
                    ("source_ids", metadata["source_ids"]),
                    ("axis_values", metadata["axis_values"]),
                    (geometry_name, metadata[geometry_name]),
                    ("values", h5["values"]),
                )
            },
        }
        if quantity == "surface_density":
            hdf_record["normals"] = metadata["normals"][:].tolist()
            hdf_record["stored_units"]["normals"] = _decode(
                metadata["normals"].attrs.get("unit", "")
            )
        if "phase_origin_m" in metadata:
            hdf_record["stored_units"]["phase_origin_m"] = _decode(
                metadata["phase_origin_m"].attrs.get("unit", "")
            )
    return {
        "input": series_record(series),
        "hdf5_stored": hdf_record,
        "hdf5_restored_common": {
            "positions_m": provider.positions_m.tolist(),
            "source_ids": list(provider.source_ids),
            "axis_values": provider.axis_values.tolist(),
            "axis_kind": provider.axis_kind,
            "coordinate_frame": provider.coordinate_frame,
            "coordinate_origin_m": provider.coordinate_origin_m.tolist(),
            "quantity": provider.quantity,
            "phase": phase_record(provider),
        },
        "gwexpy_native": native_record,
        "gwexpy_restored": series_record(restored),
    }


def surface_and_displacement_crosschecks(
    displacement, surface, targets, directory: Path
):
    def roundtrip(series, name, quantity, to_native, from_native):
        direct = _evaluate(series, targets)
        hdf5_path = directory / f"{name}.h5"
        write_fixed_mesh_series_hdf5(
            hdf5_path,
            series,
            sample_block_size=1,
            source_block_size=min(8192, series.source_count),
        )
        provider = read_fixed_mesh_series_hdf5(hdf5_path)
        hdf = _evaluate(provider, targets)
        bundle = to_native(series)
        restored = from_native(bundle)
        native = _evaluate(restored, targets)
        return {
            "hdf5_path": str(hdf5_path),
            "hdf5_sha256": _sha256(hdf5_path),
            "metadata_equal": {
                "hdf5": _hdf5_series_metadata_equal(
                    series, provider, hdf5_path, quantity
                ),
                "gwexpy": _gwexpy_series_metadata_equal(
                    series, bundle, restored, quantity
                ),
            },
            "metadata_observed": _small_series_metadata_observed(
                series, provider, hdf5_path, bundle, restored, quantity
            ),
            "hdf5_comparison": _comparison_metrics(hdf, direct),
            "gwexpy_comparison": _comparison_metrics(native, direct),
        }

    return {
        "displacement": roundtrip(
            displacement,
            "displacement",
            "displacement",
            to_gwexpy_displacement_series,
            from_gwexpy_displacement_series,
        ),
        "surface_density": roundtrip(
            surface,
            "surface",
            "surface_density",
            to_gwexpy_surface_density_series,
            from_gwexpy_surface_density_series,
        ),
    }


def _openfoam_provenance(result_payload, time_index: int):
    export_record = result_payload["commands"][5 + time_index]
    return {
        "distribution": "OpenFOAM Foundation",
        "version": "14",
        "patch": "20260724",
        "solver": "incompressibleFluid+scalarTransport-function-object",
        "compressibility": "incompressible",
        "region": "region0",
        "reconstructed": True,
        "exporter": "foamToVTK",
        "export_command": shlex.join(export_record["argv"]),
    }


def _load_openfoam_case(case_name, case_manifest, source_hashes):
    result_path = Path(case_manifest["result_path"]).resolve()
    if not result_path.is_relative_to(E_TRANSPORT_ROOT.resolve()):
        raise ValueError(f"{case_name} result is outside accepted E transport root")
    result_hash = _sha256(result_path)
    if result_hash != case_manifest["result_sha256"]:
        raise ValueError(f"{case_name} result hash differs from the E manifest")
    source_hashes[str(result_path)] = result_hash
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    science = result_payload["science"]
    if (
        science["case"] != case_name
        or science["status"] != "complete"
        or tuple(science["times"]) != E_TIMES
    ):
        raise ValueError(f"{case_name} does not declare the accepted time series")

    first_snapshot = None
    positions = None
    volumes = None
    source_ids = None
    densities = []
    raw_inputs = []
    e_observables = {observable: [] for observable in OBSERVABLES}
    reader_assertions = {
        "all_declared_times_loaded": True,
        "successor_geometry_volume_id_order_exact": True,
        "float32_ascii_9_significant_digit_profile": True,
        "reader_arrays_read_only": True,
        "si_unit_contract": True,
    }
    loaded_time_names = []
    for time_index, time_name in enumerate(E_TIMES):
        entry = science["times"][time_name]
        vtk_record = entry["vtk"]
        vtk_path = Path(vtk_record["path"]).resolve()
        if not vtk_path.is_relative_to(E_TRANSPORT_ROOT.resolve()):
            raise ValueError(f"{case_name} {time_name} VTK is outside E v3")
        actual_vtk_hash = _sha256(vtk_path)
        if (
            actual_vtk_hash != vtk_record["sha256"]
            or vtk_path.stat().st_size != vtk_record["bytes"]
            or vtk_record["payload"]
            != "float32 legacy VTK ASCII with 9 significant digits"
        ):
            raise ValueError(f"{case_name} {time_name} VTK identity differs")
        source_hashes[str(vtk_path)] = actual_vtk_hash
        snapshot = load_openfoam_density(
            vtk_path,
            field_name="delta_rho",
            time_s=float(time_name),
            time_name=time_name,
            coordinate_frame="openfoam-periodic-cube",
            coordinate_origin_m=np.zeros(3),
            density_unit="kg/m^3",
            coordinate_unit="m",
            volume_unit="m^3",
            time_unit="s",
            provenance=_openfoam_provenance(result_payload, time_index),
        )
        loaded_time_names.append(snapshot.time_name)
        expected_dtypes = {
            "points_dtype": "float32",
            "centre_dtype": "float32",
            "volume_dtype": "float32",
            "density_dtype": "float32",
            "cell_id_dtype": "int32",
        }
        if any(
            snapshot.provenance[name] != dtype
            for name, dtype in expected_dtypes.items()
        ):
            raise ValueError(f"{case_name} {time_name} reader lost float32 profile")
        reader_assertions["float32_ascii_9_significant_digit_profile"] &= bool(
            vtk_record["payload"]
            == "float32 legacy VTK ASCII with 9 significant digits"
            and all(
                snapshot.provenance[name] == dtype
                for name, dtype in expected_dtypes.items()
            )
        )
        if snapshot.provenance["input_sha256"] != actual_vtk_hash:
            raise ValueError(f"{case_name} {time_name} reader hash differs")
        arrays_read_only = not any(
            array.flags.writeable
            for array in (
                snapshot.positions_m,
                snapshot.volumes_m3,
                snapshot.density_kg_m3,
                snapshot.cell_ids,
                snapshot.coordinate_origin_m,
                snapshot.mesh.vertices_m,
            )
        )
        reader_assertions["reader_arrays_read_only"] &= arrays_read_only
        if not arrays_read_only:
            raise ValueError(f"{case_name} {time_name} reader arrays are mutable")
        si_unit_contract = (
            snapshot.density_unit == "kg/m^3"
            and snapshot.coordinate_unit == "m"
            and snapshot.volume_unit == "m^3"
            and snapshot.time_unit == "s"
            and snapshot.density_kind == "delta_rho"
            and snapshot.coordinate_frame == "openfoam-periodic-cube"
            and np.array_equal(snapshot.coordinate_origin_m, np.zeros(3))
        )
        reader_assertions["si_unit_contract"] &= bool(si_unit_contract)
        if not si_unit_contract:
            raise ValueError(f"{case_name} {time_name} reader SI contract differs")
        if first_snapshot is None:
            first_snapshot = snapshot
            positions = np.array(snapshot.positions_m, copy=True)
            volumes = np.array(snapshot.volumes_m3, copy=True)
            source_ids = tuple(f"cell-{int(value)}" for value in snapshot.cell_ids)
        else:
            same_mesh = len(first_snapshot.mesh.cell_blocks) == len(
                snapshot.mesh.cell_blocks
            ) and all(
                left[0] == right[0] and np.array_equal(left[1], right[1])
                for left, right in zip(
                    first_snapshot.mesh.cell_blocks,
                    snapshot.mesh.cell_blocks,
                    strict=True,
                )
            )
            same_geometry = bool(
                same_mesh
                and np.array_equal(
                    first_snapshot.mesh.vertices_m, snapshot.mesh.vertices_m
                )
                and all(
                    np.array_equal(
                        getattr(first_snapshot, name), getattr(snapshot, name)
                    )
                    for name in (
                        "positions_m",
                        "volumes_m3",
                        "cell_ids",
                        "coordinate_origin_m",
                    )
                )
            )
            reader_assertions["successor_geometry_volume_id_order_exact"] &= (
                same_geometry
            )
            first_snapshot.assert_compatible_successor(snapshot)
        densities.append(np.array(snapshot.density_kg_m3, copy=True))
        raw_inputs.append(
            {
                "time_name": time_name,
                "time_s": snapshot.time_s,
                "vtk_path": str(vtk_path),
                "vtk_sha256": actual_vtk_hash,
                "vtk_bytes": vtk_path.stat().st_size,
                "payload": vtk_record["payload"],
                "reader_profile": dict(snapshot.provenance),
                "positions_m_sha256": _array_sha256(snapshot.positions_m),
                "volumes_m3_sha256": _array_sha256(snapshot.volumes_m3),
                "cell_ids_sha256": _array_sha256(snapshot.cell_ids),
                "density_kg_m3_sha256": _array_sha256(snapshot.density_kg_m3),
            }
        )
        for observable in OBSERVABLES:
            e_observables[observable].append(
                np.asarray(entry["exported"]["observables_production"][observable])
            )

    if positions is None or volumes is None or source_ids is None:
        raise ValueError(f"{case_name} contains no snapshots")
    reader_assertions["all_declared_times_loaded"] = tuple(loaded_time_names) == E_TIMES
    series = DensityTimeSeries(
        positions,
        volumes,
        np.stack(densities),
        np.array([float(value) for value in E_TIMES]),
        "delta_rho",
        "openfoam-periodic-cube",
        np.zeros(3),
        source_ids,
        {
            "classification": "actual OpenFOAM 14 scalar-transport VTK snapshots",
            "case": case_name,
            "source_result_path": str(result_path),
            "source_result_sha256": result_hash,
        },
    )
    return (
        series,
        raw_inputs,
        {name: np.stack(values) for name, values in e_observables.items()},
        reader_assertions,
    )


def actual_openfoam_exchange(transport_manifest: Path, directory: Path):
    """Compare all nine accepted E snapshots through direct/HDF5/GWexpy paths."""
    manifest = json.loads(transport_manifest.read_text(encoding="utf-8"))
    if set(manifest["cases"]) != set(E_CASES):
        raise ValueError("E manifest case set differs from the accepted three cases")
    source_hashes = {str(transport_manifest.resolve()): _sha256(transport_manifest)}
    cases = {}
    raw_inputs = []
    for case_index, case_name in enumerate(E_CASES):
        series, case_inputs, e_recorded, reader_assertions = _load_openfoam_case(
            case_name, manifest["cases"][case_name], source_hashes
        )
        direct = _evaluate(series, E_TARGETS_M)
        hdf5 = hdf5_density_exchange(
            series,
            E_TARGETS_M,
            directory / f"{case_name}.h5",
            direct=direct,
        )
        gwexpy = gwexpy_density_exchange(
            series,
            E_TARGETS_M,
            direct=direct,
            current_value_probe=(1, 0, 0.001) if case_index == 0 else None,
        )
        if not all(hdf5["metadata_equal"].values()) or not all(
            gwexpy["metadata_equal"].values()
        ):
            raise ValueError(f"{case_name} exchange metadata differs")
        cases[case_name] = {
            "sample_count": series.sample_count,
            "source_count": series.source_count,
            "targets_m": E_TARGETS_M.tolist(),
            "reader_assertions": reader_assertions,
            "raw_inputs": case_inputs,
            "direct_vs_e_recorded": {
                "e_recorded_observables": _raw_observables(e_recorded),
                "raw_difference": _observable_difference(direct, e_recorded),
                "comparison": _comparison_metrics(direct, e_recorded),
            },
            "hdf5_exchange": hdf5,
            "gwexpy_exchange": gwexpy,
        }
        raw_inputs.extend(case_inputs)
    return {
        "classification": "actual accepted E OpenFOAM solver/export snapshots",
        "scope": "same discrete reader output through direct, HDF5, and native GWexpy paths",
        "transport_manifest": str(transport_manifest.resolve()),
        "transport_manifest_sha256": source_hashes[str(transport_manifest.resolve())],
        "raw_input_count": len(raw_inputs),
        "raw_input_identity": raw_inputs,
        "source_hashes": source_hashes,
        "cases": cases,
    }


def _version(distribution: str):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _assert_source(path: Path, expected_sha256: str):
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"source hash differs for {path}: {actual}")
    return actual


def _executed_module_hashes():
    """Record the selected repository modules that execute the science paths."""
    source_root = (ROOT / "src").resolve()
    records = {}
    for module_name in EXECUTED_MODULE_NAMES:
        module = importlib.import_module(module_name)
        module_file = Path(module.__file__).resolve()
        if not module_file.is_relative_to(source_root):
            raise RuntimeError(
                f"executed module {module_name} is outside the worktree: {module_file}"
            )
        records[module_name] = {
            "path": str(module_file),
            "sha256": _sha256(module_file),
        }
    return records


def _comparison_findings(label, comparison):
    findings = []
    if set(comparison) != set(OBSERVABLES):
        return [f"{label}: comparison does not contain exactly three observables"]
    for observable in OBSERVABLES:
        try:
            maximum = float(comparison[observable]["summary"]["maximum_normalized"])
        except (KeyError, TypeError, ValueError):
            findings.append(f"{label}/{observable}: maximum is missing or invalid")
            continue
        if not np.isfinite(maximum) or not 0.0 <= maximum <= SAME_DISCRETE_GATE:
            findings.append(
                f"{label}/{observable}: {maximum!r} exceeds {SAME_DISCRETE_GATE}"
            )
    return findings


def _validate_acceptance(payload):
    """Reject any payload that cannot support a complete publication."""
    findings = []
    metadata_names = {
        "geometry",
        "volumes",
        "ids",
        "axis",
        "axis_kind",
        "coordinate_frame",
        "coordinate_origin",
        "density_kind",
        "units",
        "values",
    }
    reader_names = {
        "all_declared_times_loaded",
        "successor_geometry_volume_id_order_exact",
        "float32_ascii_9_significant_digit_profile",
        "reader_arrays_read_only",
        "si_unit_contract",
    }
    small_density = payload.get("same_discrete_exchange", {})
    if small_density.get("all_metadata_equal") is not True:
        findings.append("small_density: aggregate metadata equality is false")
    for route_key in ("hdf5", "gwexpy"):
        route = small_density.get(route_key, {})
        metadata = route.get("metadata_equal", {})
        if set(metadata) != metadata_names or not all(
            value is True for value in metadata.values()
        ):
            findings.append(
                f"small_density/{route_key}: metadata is incomplete or false"
            )
        findings.extend(
            _comparison_findings(
                f"small_density/{route_key}", route.get("comparison", {})
            )
        )

    small_other = payload.get("displacement_surface_exchange", {})
    expected_small_other = {
        "displacement": "small_displacement",
        "surface_density": "small_surface_density",
    }
    per_type_metadata = {
        "displacement": {
            "positions",
            "ids",
            "axis",
            "axis_kind",
            "coordinate_frame",
            "coordinate_origin",
            "quantity",
            "phase",
            "units",
            "background_mass",
            "values",
        },
        "surface_density": {
            "positions",
            "ids",
            "axis",
            "axis_kind",
            "coordinate_frame",
            "coordinate_origin",
            "quantity",
            "phase",
            "units",
            "areas",
            "normals",
            "values",
        },
    }
    if set(small_other) != set(expected_small_other):
        findings.append("small displacement/surface panels are incomplete")
    for panel_key, label in expected_small_other.items():
        panel = small_other.get(panel_key, {})
        metadata_by_route = panel.get("metadata_equal", {})
        if set(metadata_by_route) != {"hdf5", "gwexpy"}:
            findings.append(f"{label}: HDF5/GWexpy metadata records are incomplete")
        for route_key in ("hdf5", "gwexpy"):
            metadata = metadata_by_route.get(route_key, {})
            if set(metadata) != per_type_metadata[panel_key] or not all(
                value is True for value in metadata.values()
            ):
                findings.append(f"{label}/{route_key}: metadata is incomplete or false")
        observed = panel.get("metadata_observed", {})
        endpoint_names = {
            "input",
            "hdf5_stored",
            "hdf5_restored_common",
            "gwexpy_native",
            "gwexpy_restored",
        }
        if set(observed) != endpoint_names:
            findings.append(f"{label}: observed metadata endpoints are incomplete")
        else:
            input_record = observed["input"]
            common_names = {
                "positions_m",
                "source_ids",
                "axis_values",
                "axis_kind",
                "coordinate_frame",
                "coordinate_origin_m",
                "quantity",
                "phase",
            }
            complete_names = (
                common_names
                | {"values"}
                | (
                    {"mass_kg"}
                    if panel_key == "displacement"
                    else {"areas_m2", "normals"}
                )
            )
            hdf_record = observed["hdf5_stored"]
            native_record = observed["gwexpy_native"]
            axis_unit = "s" if input_record.get("axis_kind") == "time" else "Hz"
            hdf_units = {
                "positions_m": "m",
                "coordinate_origin_m": "m",
                "source_ids": "1",
                "axis_values": axis_unit,
                "values": "m" if panel_key == "displacement" else "kg/m2",
                **(
                    {"mass_kg": "kg"}
                    if panel_key == "displacement"
                    else {"areas_m2": "m2", "normals": "1"}
                ),
                **(
                    {"phase_origin_m": "m"}
                    if input_record.get("axis_kind") == "frequency"
                    else {}
                ),
            }
            component_count = 3 if panel_key == "displacement" else 1
            native_units = {
                "values": ["m" if panel_key == "displacement" else "kg / m2"]
                * component_count,
                "axis0": [axis_unit] * component_count,
                "axis1": [""] * component_count,
                "axis2": [""] * component_count,
                "axis3": [""] * component_count,
            }
            consistent = bool(
                isinstance(input_record, dict)
                and set(input_record) == complete_names
                and observed["gwexpy_restored"] == input_record
                and set(observed["hdf5_restored_common"]) == common_names
                and observed["hdf5_restored_common"]
                == {name: input_record[name] for name in common_names}
                and {name: hdf_record.get(name) for name in complete_names}
                == input_record
                and {name: native_record.get(name) for name in complete_names}
                == input_record
                and hdf_record.get("stored_units") == hdf_units
                and native_record.get("native_units") == native_units
            )
            if not consistent:
                findings.append(
                    f"{label}: observed metadata is internally inconsistent"
                )
        for route_key in ("hdf5_comparison", "gwexpy_comparison"):
            route_label = "hdf5" if route_key == "hdf5_comparison" else "gwexpy"
            findings.extend(
                _comparison_findings(f"{label}/{route_label}", panel.get(route_key, {}))
            )

    actual = payload.get("actual_openfoam_same_discrete_exchange", {})
    cases = actual.get("cases", {})
    if actual.get("raw_input_count") != 9 or set(cases) != set(E_CASES):
        findings.append(
            "actual E panel must contain the declared three cases and nine inputs"
        )
    flattened_inputs = []
    for case_name in E_CASES:
        case = cases.get(case_name, {})
        case_inputs = case.get("raw_inputs")
        if not isinstance(case_inputs, list) or len(case_inputs) != len(E_TIMES):
            findings.append(f"{case_name}: raw input identity must contain three times")
            case_inputs = []
        else:
            observed_times = []
            for identity in case_inputs:
                if not isinstance(identity, dict):
                    findings.append(f"{case_name}: raw input identity is invalid")
                    continue
                time_name = identity.get("time_name")
                try:
                    time_s = float(identity.get("time_s"))
                except (TypeError, ValueError):
                    time_s = np.nan
                if time_name not in E_TIMES or not np.isfinite(time_s):
                    findings.append(f"{case_name}: raw input time is invalid")
                else:
                    observed_times.append((time_name, time_s))
                required_identity = {
                    "vtk_path",
                    "vtk_sha256",
                    "vtk_bytes",
                    "payload",
                    "reader_profile",
                    "positions_m_sha256",
                    "volumes_m3_sha256",
                    "cell_ids_sha256",
                    "density_kg_m3_sha256",
                }
                if not required_identity <= set(identity) or any(
                    not isinstance(identity.get(key), str)
                    or len(identity[key]) != 64
                    or any(char not in "0123456789abcdef" for char in identity[key])
                    for key in (
                        "vtk_sha256",
                        "positions_m_sha256",
                        "volumes_m3_sha256",
                        "cell_ids_sha256",
                        "density_kg_m3_sha256",
                    )
                ):
                    findings.append(f"{case_name}: raw input hashes are incomplete")
            if tuple(observed_times) != tuple(
                (time_name, float(time_name)) for time_name in E_TIMES
            ):
                findings.append(f"{case_name}: raw input time coverage differs")
        flattened_inputs.extend(case_inputs)
        reader_assertions = case.get("reader_assertions", {})
        if set(reader_assertions) != reader_names or not all(
            value is True for value in reader_assertions.values()
        ):
            findings.append(f"{case_name}: reader assertions are incomplete or false")
        for route_key in ("hdf5_exchange", "gwexpy_exchange"):
            route = case.get(route_key, {})
            metadata = route.get("metadata_equal", {})
            if set(metadata) != metadata_names or not all(
                value is True for value in metadata.values()
            ):
                findings.append(
                    f"{case_name}/{route_key}: metadata is incomplete or false"
                )
            findings.extend(
                _comparison_findings(
                    f"{case_name}/{route_key}", route.get("comparison", {})
                )
            )
        findings.extend(
            _comparison_findings(
                f"{case_name}/direct_vs_e_recorded",
                case.get("direct_vs_e_recorded", {}).get("comparison", {}),
            )
        )
    global_inputs = actual.get("raw_input_identity")
    if (
        not isinstance(global_inputs, list)
        or len(global_inputs) != 9
        or global_inputs != flattened_inputs
        or len(
            {item.get("vtk_path") for item in global_inputs if isinstance(item, dict)}
        )
        != 9
    ):
        findings.append("actual E global raw input identity is missing or inconsistent")
    probe = (
        cases.get(E_CASES[0], {})
        .get("gwexpy_exchange", {})
        .get("native_current_value_probe", {})
    )
    if probe.get("restored_density_matches_mutated_native_value") is not True:
        findings.append("native current-value edit was not restored as density")
    changed = probe.get("observable_changed", {})
    if set(changed) != set(OBSERVABLES) or not all(
        changed.get(name) is True for name in OBSERVABLES
    ):
        findings.append(
            "native current-value edit did not change all three observables"
        )

    moving = payload.get("moving_bell_different_discretizations", {})
    if set(moving) != {"24", "48"}:
        findings.append("prescribed bell panel must contain resolutions 24 and 48")
    for resolution in ("24", "48"):
        record = moving.get(resolution, {})
        if record.get("material_mass_constant") is not True:
            findings.append(f"bell n{resolution}: material mass is not constant")
        if record.get("material_mass_range_kg") != 0.0:
            findings.append(
                f"bell n{resolution}: material mass range is not exactly zero"
            )
        t0 = record.get("t0_eulerian_vs_material_normalized", {})
        if set(t0) != set(OBSERVABLES):
            findings.append(f"bell n{resolution}: t0 identity is incomplete")
        else:
            for observable, value in t0.items():
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = np.nan
                if not np.isfinite(value) or not 0.0 <= value <= SAME_DISCRETE_GATE:
                    findings.append(
                        f"bell n{resolution}/{observable}: t0 identity exceeds gate"
                    )

    resource_record = payload.get("resource", {})
    peak = resource_record.get("rss_peak_bytes")
    limit = resource_record.get("rss_limit_bytes")
    if (
        isinstance(peak, bool)
        or isinstance(limit, bool)
        or not isinstance(peak, (int, np.integer))
        or not isinstance(limit, (int, np.integer))
        or peak < 0
        or limit != 2147483648
        or peak > limit
    ):
        findings.append("peak RSS is invalid or exceeds the 2 GiB gate")

    source = payload.get("source")
    if not isinstance(source, dict):
        findings.append("source provenance is missing")
    else:
        start_sha = source.get("script_sha256_at_start")
        end_sha = source.get("script_sha256_at_end")
        matching_execution_sha = bool(
            isinstance(start_sha, str)
            and len(start_sha) == 64
            and all(char in "0123456789abcdef" for char in start_sha)
            and start_sha == end_sha
        )
        if (
            source.get("script_unchanged_during_run") is not True
            or not matching_execution_sha
        ):
            findings.append("study script changed during execution")
        script_path = Path(source.get("script", ""))
        gragra_path = Path(source.get("gragra", ""))
        execution_root = (
            script_path.parent.parent if script_path.is_absolute() else None
        )
        if (
            execution_root is None
            or script_path != execution_root / "scripts/research_wave5_crosspaths.py"
            or gragra_path != execution_root / "src/gragra/__init__.py"
            or not script_path.is_file()
            or not gragra_path.is_file()
        ):
            findings.append("historical script or gragra path is invalid")
        expected_dependencies = (
            {
                str(
                    execution_root / "src/gragra/analytic/wave456_reference.py"
                ): ANALYTIC_REFERENCE_SHA256,
                str(
                    execution_root / "src/gragra/observables/wave456_reference.py"
                ): ANALYTIC_WRAPPER_SHA256,
                str(E_DELIVERY.resolve()): E_DELIVERY_SHA256,
                str(E_TRANSPORT_MANIFEST.resolve()): E_TRANSPORT_MANIFEST_SHA256,
            }
            if execution_root is not None
            else {}
        )
        if source.get("accepted_dependencies") != expected_dependencies:
            findings.append(
                "accepted dependency provenance is incomplete or contradictory"
            )
        modules = source.get("executed_modules", {})
        if set(modules) != set(EXECUTED_MODULE_NAMES):
            findings.append("executed module provenance is incomplete")
        else:
            if modules["gragra"].get("path") != str(gragra_path):
                findings.append("gragra module path contradicts execution provenance")
            for module_name, record in modules.items():
                path = Path(record.get("path", ""))
                digest = record.get("sha256")
                if (
                    not path.is_file()
                    or not isinstance(digest, str)
                    or _sha256(path) != digest
                ):
                    findings.append(
                        f"executed module provenance differs: {module_name}"
                    )
    if findings:
        raise StudyAcceptanceError(findings)
    return {
        "status": "accepted",
        "same_discrete_maximum_normalized_gate": SAME_DISCRETE_GATE,
        "small_crosspath_panel_count": 3,
        "actual_case_count": len(E_CASES),
        "actual_snapshot_count": 9,
        "resource_gate_bytes": 2147483648,
    }


def _publish_json_create_new(path: Path, payload):
    """Verify a new partial file before linking it to the final name."""
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    partial = path.with_name(f"{path.name}.partial")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with partial.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    readback = partial.read_text(encoding="utf-8")
    if readback != encoded or json.loads(readback) != payload:
        raise OSError(f"JSON publication readback differs for {partial}")
    os.link(partial, path)
    partial.unlink()
    return path


def _file_identity(directory: Path, *, delivery_name: str):
    return [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != delivery_name
    ]


def _preprocessing_step(output, started_utc, script_path, phase, operation):
    """Run one preprocessing step and retain an incomplete exception record."""
    try:
        return operation()
    except Exception as error:
        diagnostic_payload = {
            "study": "wave5 moving density crosspaths",
            "status": "incomplete",
            "started_utc": started_utc,
            "ended_utc": datetime.now(UTC).isoformat(),
            "execution": {"argv": list(sys.argv)},
            "phase": phase,
            "exception": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "source": {
                "script": str(script_path),
                "script_sha256_at_start": _sha256(script_path),
            },
        }
        diagnostic = _publish_json_create_new(
            output / "results.incomplete.json", diagnostic_payload
        )
        _publish_json_create_new(
            output / "delivery.json",
            {
                "base_commit": "bc502b025f070f8d0aab80d18e314b0a97c45482",
                "status": "incomplete",
                "created_utc": datetime.now(UTC).isoformat(),
                "execution_argv": list(sys.argv),
                "phase": phase,
                "exception": diagnostic_payload["exception"],
                "diagnostic": {
                    "path": str(diagnostic.resolve()),
                    "sha256": _sha256(diagnostic),
                },
                "files": _file_identity(output, delivery_name="delivery.json"),
            },
        )
        raise


def run(output: Path, *, e_delivery: Path, e_transport_manifest: Path):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    started_utc = datetime.now(UTC).isoformat()
    script_path = Path(__file__).resolve()
    script_sha256_start = _sha256(script_path)
    analytic_reference_path = ROOT / "src/gragra/analytic/wave456_reference.py"
    analytic_wrapper_path = ROOT / "src/gragra/observables/wave456_reference.py"
    accepted_source_hashes = {
        str(analytic_reference_path): _assert_source(
            analytic_reference_path, ANALYTIC_REFERENCE_SHA256
        ),
        str(analytic_wrapper_path): _assert_source(
            analytic_wrapper_path, ANALYTIC_WRAPPER_SHA256
        ),
        str(e_delivery.resolve()): _assert_source(e_delivery, E_DELIVERY_SHA256),
        str(e_transport_manifest.resolve()): _assert_source(
            e_transport_manifest, E_TRANSPORT_MANIFEST_SHA256
        ),
    }
    expected_gragra = (ROOT / "src/gragra/__init__.py").resolve()
    if Path(gragra.__file__).resolve() != expected_gragra:
        raise RuntimeError(
            f"imported gragra from {gragra.__file__}, expected {expected_gragra}"
        )
    output.mkdir(parents=True)
    density, displacement, surface, targets = _preprocessing_step(
        output,
        started_utc,
        script_path,
        "build_small_crosspath_inputs",
        build_small_crosspath_inputs,
    )
    exact = _preprocessing_step(
        output,
        started_utc,
        script_path,
        "same_discrete_exchange",
        lambda: same_discrete_exchange(density, targets, output),
    )
    cross = _preprocessing_step(
        output,
        started_utc,
        script_path,
        "surface_and_displacement_crosschecks",
        lambda: surface_and_displacement_crosschecks(
            displacement, surface, targets, output
        ),
    )
    moving_bell = _preprocessing_step(
        output,
        started_utc,
        script_path,
        "moving_bell_comparison",
        lambda: {
            str(resolution): moving_bell_comparison(resolution=resolution)
            for resolution in (24, 48)
        },
    )
    actual_openfoam = _preprocessing_step(
        output,
        started_utc,
        script_path,
        "actual_openfoam_exchange",
        lambda: actual_openfoam_exchange(e_transport_manifest, output),
    )
    executed_modules = _preprocessing_step(
        output,
        started_utc,
        script_path,
        "executed_module_provenance",
        _executed_module_hashes,
    )
    script_sha256_end = _sha256(script_path)
    payload = {
        "study": "wave5 moving density crosspaths",
        "started_utc": started_utc,
        "ended_utc": datetime.now(UTC).isoformat(),
        "execution": {
            "argv": list(sys.argv),
            "analysis_only": True,
            "openfoam_solver_rerun": False,
        },
        "source": {
            "gragra": str(expected_gragra),
            "script": str(script_path),
            "script_sha256_at_start": script_sha256_start,
            "script_sha256_at_end": script_sha256_end,
            "script_unchanged_during_run": script_sha256_start == script_sha256_end,
            "accepted_dependencies": accepted_source_hashes,
            "executed_modules": executed_modules,
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "h5py": _version("h5py"),
            "meshio": _version("meshio"),
            "gwexpy": _version("gwexpy"),
        },
        "same_discrete_exchange": exact,
        "actual_openfoam_same_discrete_exchange": actual_openfoam,
        "displacement_surface_exchange": cross,
        "moving_bell_different_discretizations": moving_bell,
        "supersedes": [
            "run-20260912-1229 used an invalid first-order displacement versus absolute-density comparison",
            "run-20260912-1250 added absolute moving-mass paths but lacked actual E exchange and an independent continuous reference",
        ],
        "resource": {
            "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "rss_limit_bytes": 2147483648,
            "single_process": True,
        },
    }
    try:
        acceptance = _validate_acceptance(payload)
    except StudyAcceptanceError as error:
        payload["status"] = "incomplete"
        payload["acceptance"] = {
            "status": "rejected",
            "same_discrete_maximum_normalized_gate": SAME_DISCRETE_GATE,
            "findings": list(error.findings),
        }
        diagnostic = _publish_json_create_new(
            output / "results.incomplete.json", payload
        )
        _publish_json_create_new(
            output / "delivery.json",
            {
                "base_commit": "bc502b025f070f8d0aab80d18e314b0a97c45482",
                "status": "incomplete",
                "created_utc": datetime.now(UTC).isoformat(),
                "execution_argv": list(sys.argv),
                "diagnostic": {
                    "path": str(diagnostic.resolve()),
                    "sha256": _sha256(diagnostic),
                },
                "findings": list(error.findings),
                "files": _file_identity(output, delivery_name="delivery.json"),
                "rss_peak_bytes": payload["resource"]["rss_peak_bytes"],
            },
        )
        raise

    payload["status"] = "complete"
    payload["acceptance"] = acceptance
    result = _publish_json_create_new(output / "results.json", payload)
    files = _file_identity(output, delivery_name="delivery.json")
    delivery = output / "delivery.json"
    _publish_json_create_new(
        delivery,
        {
            "base_commit": "bc502b025f070f8d0aab80d18e314b0a97c45482",
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "execution_argv": list(sys.argv),
            "source_hashes": payload["source"],
            "raw_output_identity": actual_openfoam["raw_input_identity"],
            "files": files,
            "rss_peak_bytes": payload["resource"]["rss_peak_bytes"],
            "acceptance": acceptance,
        },
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--e-delivery", type=Path, default=E_DELIVERY)
    parser.add_argument(
        "--e-transport-manifest", type=Path, default=E_TRANSPORT_MANIFEST
    )
    arguments = parser.parse_args()
    print(
        run(
            arguments.output,
            e_delivery=arguments.e_delivery,
            e_transport_manifest=arguments.e_transport_manifest,
        )
    )
