"""Baseline public workflows that C2 comparison migration must preserve."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries

from gragra.analytic.objects import uniform_sphere
from gragra.analytic.seismic import rayleigh_halfspace
from gragra.geometry import Frame, HalfSpace, Sphere
from gragra.gravity_sources import discretize, point_sum
from gragra.results import PhasorRepresentation, compare
from gragra.targets import TargetPoints


def test_public_sphere_results_compare_candidate_minus_reference():
    world = Frame("world")
    sphere = Sphere(np.zeros(3) * u.m, 1 * u.m, world)
    targets = TargetPoints([[0, 0, 2]] * u.m, ("outside",), world)
    density = 2500 * u.kg / u.m**3

    reference = uniform_sphere(sphere, density).at(targets).run()
    candidate = point_sum(discretize(sphere, density, 0.2 * u.m)).at(targets).run()
    comparison = compare(reference=reference, candidate=candidate)

    assert comparison.reference is reference
    assert comparison.candidate is candidate
    assert comparison.differences.name == "candidate - reference"
    analytic = reference.potential.to_value(u.m**2 / u.s**2)
    discrete = candidate.potential.to_value(u.m**2 / u.s**2)
    delta = comparison.differences.potential.to_value(u.m**2 / u.s**2)
    np.testing.assert_allclose(delta, discrete - analytic, rtol=1e-12, atol=0.0)
    assert delta[0] < 0


def _rayleigh_result(
    *,
    ids=("A", "B"),
    positions=None,
    frequencies=(1.0, 2.0),
    density=2500.0,
    vp=4000.0,
    vs=2300.0,
    amplitudes=(1e-9 + 2e-9j, 3e-9 - 1e-9j),
    reference_time=(1_000_000_000, 0),
    phase_origin=(0.0, 0.0, 0.0),
    output_frame=None,
    root=None,
):
    world = root or Frame("surface")
    targets = TargetPoints(
        (positions if positions is not None else [[0, 0, -10], [3, 0, -15]]) * u.m,
        ids,
        world,
    )
    model = rayleigh_halfspace(
        geometry=HalfSpace(world),
        density=density * u.kg / u.m**3,
        vp=vp * u.m / u.s,
        vs=vs * u.m / u.s,
        vertical_displacement=FrequencySeries(
            np.asarray(amplitudes, dtype=np.complex128),
            frequencies=np.asarray(frequencies) * u.Hz,
            unit=u.m,
        ),
        representation=PhasorRepresentation(),
        reference_time=reference_time,
        phase_origin=np.asarray(phase_origin) * u.m,
        direction=(1, 0, 0),
    )
    return model.at(targets, output_frame=output_frame or world).run()


def test_rayleigh_comparison_accepts_model_parameter_and_phase_origin_differences():
    world = Frame("surface")
    reference = _rayleigh_result(root=world)
    candidate = _rayleigh_result(
        density=2600.0,
        vp=4100.0,
        vs=2400.0,
        amplitudes=(2e-9 + 3e-9j, 4e-9 - 1e-9j),
        root=world,
    )
    # Phase origin is model input and does not change compare compatibility.
    shifted = _rayleigh_result(phase_origin=(2.0, -1.0, 0.0), root=world)
    comparison = compare(reference=reference, candidate=candidate)

    delta = np.asarray(comparison.differences.potential)
    expected_delta = np.asarray(candidate.potential) - np.asarray(reference.potential)
    np.testing.assert_allclose(delta, expected_delta, rtol=1e-12, atol=0.0)
    assert np.iscomplexobj(delta)
    assert np.any(delta.imag != 0)
    assert np.any(delta != 0)
    shifted_comparison = compare(reference=reference, candidate=shifted)
    assert np.any(np.asarray(shifted_comparison.differences.potential) != 0)


def test_rayleigh_compare_aligns_ids_and_rejects_key_mismatches():
    world = Frame("surface")
    reference = _rayleigh_result(root=world)
    reordered = _rayleigh_result(
        ids=("B", "A"), positions=[[3, 0, -15], [0, 0, -10]], root=world
    )
    comparison = compare(reference=reference, candidate=reordered)
    assert comparison.summary["target_id"].tolist() == ["A", "A", "B", "B"]
    np.testing.assert_allclose(
        np.asarray(comparison.differences.potential),
        0.0,
        rtol=0.0,
        atol=0.0,
    )

    with pytest.raises(ValueError):
        compare(
            reference=reference,
            candidate=_rayleigh_result(reference_time=(1_000_000_000, 1), root=world),
        )
    with pytest.raises(ValueError):
        compare(
            reference=reference,
            candidate=_rayleigh_result(frequencies=(1, 2.1), root=world),
        )
    with pytest.raises(ValueError):
        compare(
            reference=reference,
            candidate=_rayleigh_result(ids=("A", "C"), root=world),
        )
    with pytest.raises(ValueError):
        compare(
            reference=reference,
            candidate=_rayleigh_result(
                positions=[[0, 0, -10], [3, 0, -15.1]], root=world
            ),
        )


def test_rayleigh_compare_rejects_different_output_basis():
    world = Frame("surface")
    rotated = Frame("rotated", basis=[[0, -1, 0], [1, 0, 0], [0, 0, 1]], parent=world)
    reference = _rayleigh_result(root=world)
    candidate = _rayleigh_result(output_frame=rotated, root=world)

    with pytest.raises(ValueError):
        compare(reference=reference, candidate=candidate)

    static = (
        uniform_sphere(Sphere(np.zeros(3) * u.m, 1 * u.m, world), 2500 * u.kg / u.m**3)
        .at(TargetPoints([[0, 0, -10], [3, 0, -15]] * u.m, ("A", "B"), world))
        .run()
    )
    with pytest.raises(ValueError):
        compare(reference=reference, candidate=static)
