"""Wave 6-A external mesh -> observables -> exchange E2E tests."""

import gwexpy  # noqa: F401 - required dependency must fail before optional skips
import numpy as np
import pytest

pytest.importorskip("meshio")
pytest.importorskip("h5py")

from gragra.adapters.field_series_hdf5 import (
    read_fixed_mesh_series_hdf5,
    write_fixed_mesh_series_hdf5,
)
from gragra.adapters.gwexpy_displacement_series import (
    from_gwexpy_displacement_series,
    to_gwexpy_displacement_series,
)
from gragra.adapters.meshio_adapter import from_meshio_volume
from scripts.research_wave6_mesh_exchange_e2e import (
    OBSERVABLES,
    build_host_cavity_volume_mesh,
    build_mass_displacement_sources,
    evaluate_exchange_paths,
    mass_displacement_series,
    write_host_cavity_gmsh_file,
)


@pytest.mark.parametrize("wavelength_m", [20.0, 40.0, 80.0])
@pytest.mark.parametrize("observable", OBSERVABLES)
def test_wave6_a_host_cavity_evaluation_and_exchange_paths_agree(
    tmp_path, wavelength_m, observable
):
    """(b) Evaluation/exchange comparison for the wave6-A host+cavity
    representative case's mass-displacement representation: direct
    ``displacement_transfer_*`` calls, the ``observables.series`` dispatch
    entry point, an HDF5 round trip, and a GWexpy round trip must agree
    bit-identically.
    """
    volume_mesh = build_host_cavity_volume_mesh()
    mesh_path = write_host_cavity_gmsh_file(volume_mesh, tmp_path / "host_cavity.msh")
    file_mesh = from_meshio_volume(mesh_path)

    sources = build_mass_displacement_sources(file_mesh, wavelength_m=wavelength_m)
    series = mass_displacement_series(sources, wavelength_m=wavelength_m)

    hdf5_path = tmp_path / "mass_displacement_series.h5"
    write_fixed_mesh_series_hdf5(hdf5_path, series)
    provider = read_fixed_mesh_series_hdf5(hdf5_path)
    recovered_series = from_gwexpy_displacement_series(
        to_gwexpy_displacement_series(series)
    )

    targets = np.array(
        ((2.31, -0.27, -11.81), (24.0, 3.0, -14.0)),
    )
    paths = evaluate_exchange_paths(
        sources, series, provider, recovered_series, targets, observable=observable
    )

    np.testing.assert_array_equal(paths["direct"], paths["series"])
    np.testing.assert_array_equal(paths["direct"], paths["hdf5"])
    np.testing.assert_array_equal(paths["direct"], paths["gwexpy"])
