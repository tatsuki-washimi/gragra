"""Actual native GWexpy round trips for all Lane G bridges."""
# ruff: noqa: E402, I001

import pathlib

import numpy as np
import pytest

import gwexpy
import gragra  # noqa: E402
from gragra.adapters._gwexpy_series import GwexpySeriesBundle  # noqa: E402
from gragra.adapters.gwexpy_density_series import (  # noqa: E402
    from_gwexpy_density_series,
    to_gwexpy_density_series,
)
from gragra.adapters.gwexpy_displacement_series import (  # noqa: E402
    from_gwexpy_displacement_series,
    to_gwexpy_displacement_series,
)
from gragra.adapters.gwexpy_surface_density_series import (  # noqa: E402
    from_gwexpy_surface_density_series,
    to_gwexpy_surface_density_series,
)
from gragra.fields.density_series import DensityPhasors, DensityTimeSeries  # noqa: E402
from gragra.fields.displacement_series import FixedMeshDisplacementSeries  # noqa: E402
from gragra.fields.surface_density_series import FixedMeshSurfaceDensitySeries  # noqa: E402
from gragra.observables import (  # noqa: E402
    displacement_transfer_acceleration,
    displacement_transfer_gravity_gradient,
    displacement_transfer_potential,
    point_acceleration,
    point_gravity_gradient,
    point_potential,
)
from gragra.gravity_sources import WeightedPointSource  # noqa: E402


def _args():
    return np.zeros((2, 3)), np.ones(2), ["a", "b"]


def test_native_bridges_read_current_values_not_sidecar():
    assert pathlib.Path(gragra.__file__).resolve().parents[1].name == "src"
    positions, weights, ids = _args()
    density = DensityTimeSeries(
        positions,
        weights,
        np.ones((2, 2)),
        np.array([0.0, 1.0]),
        "rho",
        "local",
        np.zeros(3),
        ids,
    )
    bundle = to_gwexpy_density_series(density)
    bundle.field.value[1, 1, 0, 0] = 7.0
    assert from_gwexpy_density_series(bundle).density_kg_m3[1, 1] == 7.0
    displacement = FixedMeshDisplacementSeries(
        positions,
        weights,
        np.ones((2, 2, 3)),
        np.array([0.0, 1.0]),
        "time",
        "local",
        np.zeros(3),
        ids,
    )
    vector = to_gwexpy_displacement_series(displacement)
    vector.field["z"].value[0, 0, 0, 0] = 8.0
    assert from_gwexpy_displacement_series(vector).displacement_m[0, 0, 2] == 8.0
    surface = FixedMeshSurfaceDensitySeries(
        positions,
        weights,
        np.array([[1.0, 0, 0], [0, 1.0, 0]]),
        np.ones((2, 2)),
        np.array([0.0, 1.0]),
        "time",
        {},
        "local",
        np.zeros(3),
        ids,
    )
    scalar = to_gwexpy_surface_density_series(surface)
    scalar.field.value[0, 1, 0, 0] = -2.0
    assert (
        from_gwexpy_surface_density_series(scalar).surface_density_kg_m2[0, 1] == -2.0
    )


def test_density_bridge_rejects_native_axis_and_sidecar_contract_mutations():
    """A sidecar may not be used to mask a reordered native frequency/time axis."""
    positions, weights, ids = _args()
    series = DensityTimeSeries(
        positions,
        weights,
        np.ones((2, 2)),
        np.array([0.0, 1.0]),
        "rho",
        "local",
        np.zeros(3),
        ids,
    )
    bundle = to_gwexpy_density_series(series)
    companion = dict(bundle.companion)
    companion["quantity"] = "surface_density"
    with pytest.raises(ValueError, match="quantity"):
        from_gwexpy_density_series(GwexpySeriesBundle(bundle.field, companion))
    bundle.field._axis0_index = bundle.field._axis0_index[::-1]
    with pytest.raises(ValueError, match="axis values or ordering"):
        from_gwexpy_density_series(bundle)


@pytest.mark.parametrize("component", ["x", "y", "z"])
def test_displacement_bridge_checks_each_native_vector_component(component):
    positions, masses, ids = _args()
    series = FixedMeshDisplacementSeries(
        positions,
        masses,
        np.ones((2, 2, 3), dtype=np.complex128),
        np.array([1.0, 2.0]),
        "frequency",
        "local",
        np.zeros(3),
        ids,
        t_ref_s=0.5,
        phase_origin_m=np.array([1.0, 0.0, 0.0]),
    )
    bundle = to_gwexpy_displacement_series(series)
    bundle.field[component]._axis0_index = bundle.field[component]._axis0_index[::-1]
    with pytest.raises(ValueError, match="axis values or ordering"):
        from_gwexpy_displacement_series(bundle)


def test_frequency_sidecar_rejects_ids_and_fourier_convention_mutations():
    positions, volumes, ids = _args()
    series = DensityPhasors(
        positions,
        volumes,
        np.ones((2, 2), dtype=np.complex128),
        np.array([1.0, 2.0]),
        0.5,
        np.array([1.0, 0.0, 0.0]),
        "local",
        np.zeros(3),
        ids,
    )
    bundle = to_gwexpy_density_series(series)
    bad_ids = dict(bundle.companion)
    bad_ids["source_ids"] = ("a", "a")
    with pytest.raises(ValueError, match="source_ids"):
        from_gwexpy_density_series(GwexpySeriesBundle(bundle.field, bad_ids))
    empty_id = dict(bundle.companion)
    empty_id["source_ids"] = ("a", "")
    with pytest.raises(ValueError, match="source_ids"):
        from_gwexpy_density_series(GwexpySeriesBundle(bundle.field, empty_id))
    bad_phase = dict(bundle.companion)
    bad_phase["fourier_convention"] = "opposite sign"
    with pytest.raises(ValueError, match="Fourier convention"):
        from_gwexpy_density_series(GwexpySeriesBundle(bundle.field, bad_phase))


def test_density_bridge_rejects_native_axis_unit_and_domain_mutations():
    from astropy import units

    positions, volumes, ids = _args()
    series = DensityTimeSeries(
        positions,
        volumes,
        np.ones((2, 2)),
        np.array([0.0, 1.0]),
        "rho",
        "local",
        np.zeros(3),
        ids,
    )
    unit_bundle = to_gwexpy_density_series(series)
    unit_bundle.field._axis0_index = unit_bundle.field._axis0_index * units.m / units.s
    with pytest.raises(ValueError, match="axis unit"):
        from_gwexpy_density_series(unit_bundle)
    domain_bundle = to_gwexpy_density_series(series)
    domain_bundle.field._space_domains["point"] = "frequency"
    with pytest.raises(ValueError, match="spatial domains"):
        from_gwexpy_density_series(domain_bundle)


@pytest.mark.parametrize("axis_kind", ["time", "frequency"])
def test_displacement_inverse_maps_noncanonical_vector_order_by_component_name(
    axis_kind,
):
    """Named VectorField components must never be interpreted by insertion order."""
    positions = np.array([[0.2, 0.3, -0.4], [-0.5, 0.7, 0.6]])
    values = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
        ],
        dtype=np.complex128 if axis_kind == "frequency" else np.float64,
    )
    kwargs = {}
    axis = np.array([0.0, 0.7])
    if axis_kind == "frequency":
        values *= 1.0 + 0.5j
        axis = np.array([1.25, 3.75])
        kwargs = {"t_ref_s": 0.2, "phase_origin_m": np.array([1.0, 0.0, 0.0])}
    series = FixedMeshDisplacementSeries(
        positions,
        np.array([2.0, 3.0]),
        values,
        axis,
        axis_kind,
        "local",
        np.zeros(3),
        ["first", "second"],
        **kwargs,
    )
    bundle = to_gwexpy_displacement_series(series)
    reordered = gwexpy.fields.VectorField(
        {"z": bundle.field["z"], "x": bundle.field["x"], "y": bundle.field["y"]}
    )
    restored = from_gwexpy_displacement_series(
        GwexpySeriesBundle(reordered, bundle.companion)
    )
    np.testing.assert_allclose(restored.displacement_m, values, rtol=0.0, atol=0.0)
    _assert_observables_equal(
        _displacement_observables(series, values),
        _displacement_observables(restored, restored.displacement_m),
    )


def test_density_phasor_rejects_rho_and_time_rejects_phase_sidecar_fields():
    positions, volumes, ids = _args()
    phasor = DensityPhasors(
        positions,
        volumes,
        np.ones((2, 2), complex),
        np.array([1.0, 2.0]),
        0.0,
        np.zeros(3),
        "local",
        np.zeros(3),
        ids,
    )
    phasor_bundle = to_gwexpy_density_series(phasor)
    bad_kind = dict(phasor_bundle.companion)
    bad_kind["density_kind"] = "rho"
    with pytest.raises(ValueError, match="density_kind"):
        from_gwexpy_density_series(GwexpySeriesBundle(phasor_bundle.field, bad_kind))
    time = DensityTimeSeries(
        positions,
        volumes,
        np.ones((2, 2)),
        np.array([0.0, 1.0]),
        "rho",
        "local",
        np.zeros(3),
        ids,
    )
    time_bundle = to_gwexpy_density_series(time)
    bad_time = dict(time_bundle.companion)
    bad_time["t_ref_s"] = 0.0
    with pytest.raises(ValueError, match="must not contain"):
        from_gwexpy_density_series(GwexpySeriesBundle(time_bundle.field, bad_time))


def _scalar_observables(series, values, multiplier):
    target = np.array([[4.0, -2.0, 3.0]])
    result = []
    for sample in values:
        source = WeightedPointSource(series.positions_m, sample * multiplier)
        result.append(
            (
                point_potential(source, target),
                point_acceleration(source, target),
                point_gravity_gradient(source, target),
            )
        )
    return result


def _displacement_observables(series, values):
    target = np.array([[4.0, -2.0, 3.0]])
    return [
        (
            displacement_transfer_potential(
                series.positions_m, series.mass_kg, sample, target
            ),
            displacement_transfer_acceleration(
                series.positions_m, series.mass_kg, sample, target
            ),
            displacement_transfer_gravity_gradient(
                series.positions_m, series.mass_kg, sample, target
            ),
        )
        for sample in values
    ]


def _physical_norm(value, observable):
    array = np.asarray(value)
    if observable == "potential":
        return np.abs(array)
    if observable == "acceleration":
        return np.linalg.norm(array, axis=-1)
    return np.linalg.norm(array, axis=(-2, -1))


def _assert_one_observable_equal(before, after, observable, *, zero_control_scale=None):
    """Compare per target against its physical nonzero reference norm.

    A true zero reference must supply the norm of a matched nonzero control as
    ``zero_control_scale``; an arbitrary unit floor would hide relative errors.
    """
    for expected, actual in zip(before, after, strict=True):
        reference = _physical_norm(expected, observable)
        if np.any(reference == 0.0):
            if zero_control_scale is None:
                raise ValueError("zero reference requires a nonzero control scale")
            control = np.broadcast_to(np.asarray(zero_control_scale), reference.shape)
            if not np.isfinite(control).all() or np.any(control <= 0.0):
                raise ValueError("zero control scale must be finite and positive")
            scale = np.where(reference > 0.0, reference, control)
        else:
            scale = reference
        normalized = _physical_norm(actual - expected, observable) / scale
        assert np.all(normalized <= 1e-11)


def _assert_observables_equal(before, after):
    for index, observable in enumerate(("potential", "acceleration", "gradient")):
        _assert_one_observable_equal(
            [sample[index] for sample in before],
            [sample[index] for sample in after],
            observable,
        )


@pytest.mark.parametrize("kind", ["density", "displacement", "surface"])
def test_frequency_bridges_preserve_complex_three_observables_and_native_edits(kind):
    """Actual complex frequency bridges preserve all three observables."""
    positions = np.array([[0.2, 0.3, -0.4], [-0.5, 0.7, 0.6]])
    axis = np.array([1.25, 3.75])  # intentionally irregular frequency axis
    ids = ["first", "second"]
    complex_values = np.array([[1.0 + 2.0j, -0.5 + 0.25j], [0.3 - 0.7j, 2.0 + 1.0j]])
    common = {"t_ref_s": 0.125, "phase_origin_m": np.array([1.0, -2.0, 0.5])}
    if kind == "density":
        series = DensityPhasors(
            positions,
            np.array([2.0, 3.0]),
            complex_values,
            axis,
            common["t_ref_s"],
            common["phase_origin_m"],
            "local",
            np.zeros(3),
            ids,
        )
        bundle = to_gwexpy_density_series(series)
        restored = from_gwexpy_density_series(bundle)
        before = _scalar_observables(series, series.density_kg_m3, series.volumes_m3)
        after = _scalar_observables(
            restored, restored.density_kg_m3, restored.volumes_m3
        )
        np.testing.assert_array_equal(restored.density_kg_m3, series.density_kg_m3)
        np.testing.assert_array_equal(restored.volumes_m3, series.volumes_m3)
        native = bundle.field.value.copy()
        bundle.field.value[...] *= 1.1
        amplitude_changed = from_gwexpy_density_series(bundle)
        amplitude_edited = _scalar_observables(
            amplitude_changed,
            amplitude_changed.density_kg_m3,
            amplitude_changed.volumes_m3,
        )
        bundle.field.value[...] = native
        bundle.field.value[0, 0, 0, 0] += 0.75j
        changed = from_gwexpy_density_series(bundle)
        edited = _scalar_observables(changed, changed.density_kg_m3, changed.volumes_m3)
    elif kind == "displacement":
        values = np.stack(
            (complex_values, complex_values * (1.0 - 0.5j), complex_values * 0.3),
            axis=-1,
        )
        series = FixedMeshDisplacementSeries(
            positions,
            np.array([2.0, 3.0]),
            values,
            axis,
            "frequency",
            "local",
            np.zeros(3),
            ids,
            **common,
        )
        bundle = to_gwexpy_displacement_series(series)
        restored = from_gwexpy_displacement_series(bundle)
        before = _displacement_observables(series, series.displacement_m)
        after = _displacement_observables(restored, restored.displacement_m)
        np.testing.assert_array_equal(restored.displacement_m, series.displacement_m)
        np.testing.assert_array_equal(restored.mass_kg, series.mass_kg)
        native = {
            component: bundle.field[component].value.copy()
            for component in ("x", "y", "z")
        }
        for component in ("x", "y", "z"):
            bundle.field[component].value[...] *= 1.1
        amplitude_changed = from_gwexpy_displacement_series(bundle)
        amplitude_edited = _displacement_observables(
            amplitude_changed, amplitude_changed.displacement_m
        )
        for component in ("x", "y", "z"):
            bundle.field[component].value[...] = native[component]
        for component in ("x", "y", "z"):
            bundle.field[component].value[0, 0, 0, 0] += 0.25j
        changed = from_gwexpy_displacement_series(bundle)
        edited = _displacement_observables(changed, changed.displacement_m)
    else:
        series = FixedMeshSurfaceDensitySeries(
            positions,
            np.array([2.0, 3.0]),
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            complex_values,
            axis,
            "frequency",
            {"side": ["a", "b"]},
            "local",
            np.zeros(3),
            ids,
            **common,
        )
        bundle = to_gwexpy_surface_density_series(series)
        restored = from_gwexpy_surface_density_series(bundle)
        before = _scalar_observables(
            series, series.surface_density_kg_m2, series.areas_m2
        )
        after = _scalar_observables(
            restored, restored.surface_density_kg_m2, restored.areas_m2
        )
        np.testing.assert_array_equal(
            restored.surface_density_kg_m2, series.surface_density_kg_m2
        )
        np.testing.assert_array_equal(restored.areas_m2, series.areas_m2)
        np.testing.assert_array_equal(restored.normals, series.normals)
        native = bundle.field.value.copy()
        bundle.field.value[...] *= 1.1
        amplitude_changed = from_gwexpy_surface_density_series(bundle)
        amplitude_edited = _scalar_observables(
            amplitude_changed,
            amplitude_changed.surface_density_kg_m2,
            amplitude_changed.areas_m2,
        )
        bundle.field.value[...] = native
        bundle.field.value[0, 0, 0, 0] += 0.75j
        changed = from_gwexpy_surface_density_series(bundle)
        edited = _scalar_observables(
            changed, changed.surface_density_kg_m2, changed.areas_m2
        )
    _assert_observables_equal(before, after)
    for index, observable in enumerate(("potential", "acceleration", "gradient")):
        with pytest.raises(AssertionError):
            _assert_one_observable_equal(
                [sample[index] for sample in before],
                [sample[index] for sample in amplitude_edited],
                observable,
            )
    assert any(
        np.linalg.norm(a - b) > 0.0 for a, b in zip(before[0], edited[0], strict=True)
    )
