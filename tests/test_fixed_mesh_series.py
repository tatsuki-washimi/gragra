"""Behavioral contracts for Lane G immutable fixed-mesh input types."""

import subprocess
import sys

import numpy as np
import pytest

from gragra.fields.density_series import DensityPhasors, DensityTimeSeries
from gragra.fields.displacement_series import FixedMeshDisplacementSeries
from gragra.fields.surface_density_series import FixedMeshSurfaceDensitySeries


def _common():
    return np.zeros((2, 3)), np.ones(2), np.array(["left", "right"])


def test_density_time_series_is_defensive_and_requires_ids():
    positions, volumes, ids = _common()
    series = DensityTimeSeries(
        positions,
        volumes,
        np.ones((2, 2)),
        np.array([0.0, 0.3]),
        "rho",
        "local",
        np.zeros(3),
        ids,
    )
    assert series.axis_kind == "time" and not series.density_kg_m3.flags.writeable
    positions[0, 0] = 9
    assert series.positions_m[0, 0] == 0
    with pytest.raises(TypeError, match="source_ids"):
        DensityTimeSeries(
            positions,
            volumes,
            np.ones((2, 2)),
            np.array([0.0, 0.3]),
            "rho",
            "local",
            np.zeros(3),
            None,
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda p, scale, ids: DensityTimeSeries(
            p,
            scale,
            np.ones((2, 2)),
            np.array([0.0, 1.0]),
            "rho",
            "local",
            np.zeros(3),
            ids,
        ),
        lambda p, scale, ids: DensityPhasors(
            p,
            scale,
            np.ones((2, 2), dtype=np.complex128),
            np.array([1.0, 2.0]),
            0.0,
            np.zeros(3),
            "local",
            np.zeros(3),
            ids,
        ),
        lambda p, scale, ids: FixedMeshDisplacementSeries(
            p,
            scale,
            np.ones((2, 2, 3)),
            np.array([0.0, 1.0]),
            "time",
            "local",
            np.zeros(3),
            ids,
        ),
        lambda p, scale, ids: FixedMeshSurfaceDensitySeries(
            p,
            scale,
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.ones((2, 2)),
            np.array([0.0, 1.0]),
            "time",
            {},
            "local",
            np.zeros(3),
            ids,
        ),
    ],
    ids=["density-time", "density-phasor", "displacement", "surface-density"],
)
def test_all_fixed_mesh_series_reject_empty_source_ids(factory):
    positions, scale, _ = _common()
    with pytest.raises(ValueError, match="non-empty"):
        factory(positions, scale, np.array(["left", ""]))


def test_frequency_types_require_explicit_phase_contract():
    positions, masses, ids = _common()
    with pytest.raises(ValueError, match="require t_ref_s"):
        FixedMeshDisplacementSeries(
            positions,
            masses,
            np.ones((2, 2, 3), complex),
            np.array([1.0, 2.0]),
            "frequency",
            "local",
            np.zeros(3),
            ids,
        )
    series = FixedMeshDisplacementSeries(
        positions,
        masses,
        np.ones((2, 2, 3), complex),
        np.array([1.0, 2.0]),
        "frequency",
        "local",
        np.zeros(3),
        ids,
        t_ref_s=1.5,
        phase_origin_m=np.zeros(3),
    )
    assert series.fourier_convention == "Re[q_hat exp(-2pi i f(t-t_ref_s))]"
    with pytest.raises(ValueError, match="must not declare"):
        FixedMeshDisplacementSeries(
            positions,
            masses,
            np.ones((2, 2, 3)),
            np.array([0.0, 1.0]),
            "time",
            "local",
            np.zeros(3),
            ids,
            t_ref_s=0.0,
        )


def test_surface_density_keeps_area_and_normal_separate():
    positions, areas, ids = _common()
    series = FixedMeshSurfaceDensitySeries(
        positions,
        areas,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.ones((2, 2)),
        np.array([0.0, 1.0]),
        "time",
        {"side": ["inside", "outside"]},
        "local",
        np.zeros(3),
        ids,
    )
    assert series.areas_m2.flags.writeable is False
    assert series.face_metadata["side"] == ("inside", "outside")


def test_phasors_are_complex_and_positive_frequency_only():
    positions, volumes, ids = _common()
    with pytest.raises(ValueError, match="positive"):
        DensityPhasors(
            positions,
            volumes,
            np.ones((2, 2), complex),
            np.array([0.0, 1.0]),
            0.0,
            np.zeros(3),
            "local",
            np.zeros(3),
            ids,
        )


def test_series_core_has_no_adapter_import_boundary():
    """Core input types remain importable before any optional GWexpy adapter."""
    code = (
        "import sys; "
        "sys.meta_path=[x for x in sys.meta_path "
        "if '_editable' not in type(x).__module__]; "
        "import gragra.fields.density_series; "
        "assert not any(name.startswith('gragra.adapters') for name in sys.modules)"
    )
    completed = subprocess.run([sys.executable, "-c", code], check=False)
    assert completed.returncode == 0


def test_metadata_rejects_nonstring_keys_and_self_referential_sequences():
    positions, volumes, ids = _common()
    with pytest.raises(TypeError, match="keys must be strings"):
        DensityTimeSeries(
            positions,
            volumes,
            np.ones((2, 2)),
            np.array([0.0, 1.0]),
            "rho",
            "local",
            np.zeros(3),
            ids,
            {1: "bad"},
        )
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic"):
        DensityTimeSeries(
            positions,
            volumes,
            np.ones((2, 2)),
            np.array([0.0, 1.0]),
            "rho",
            "local",
            np.zeros(3),
            ids,
            {"cycle": cyclic},
        )
