"""Import contracts for accepted first-wave subpackage APIs."""

import gragra.adapters as adapters
import gragra.gravity_sources as sources
import gragra.observables as observables


def test_displacement_observables_are_subpackage_exports() -> None:
    expected = {
        "displacement_transfer_potential",
        "displacement_transfer_gravity_gradient",
    }
    assert expected <= set(observables.__all__)
    for name in expected:
        assert callable(getattr(observables, name))


def test_gwexpy_observable_bridge_is_lazy_subpackage_export() -> None:
    expected = {
        "GwexpyObservableBundle",
        "ObservableArrayBundle",
        "to_gwexpy_observable",
        "from_gwexpy_observable",
    }
    assert expected <= set(adapters.__all__)
    for name in expected:
        assert getattr(adapters, name) is not None


def test_prescribed_motion_sources_and_observables_are_subpackage_exports() -> None:
    source_names = {"PrescribedPointTrajectory", "PrescribedRigidTransform"}
    observable_names = {
        "trajectory_potential",
        "trajectory_acceleration",
        "trajectory_gravity_gradient",
        "rigid_transform_potential",
        "rigid_transform_acceleration",
        "rigid_transform_gravity_gradient",
    }
    assert source_names <= set(sources.__all__)
    assert observable_names <= set(observables.__all__)
    for name in source_names:
        assert getattr(sources, name) is not None
    for name in observable_names:
        assert callable(getattr(observables, name))
