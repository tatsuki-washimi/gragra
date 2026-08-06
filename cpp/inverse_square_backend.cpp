#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <algorithm>
#include "kernels.hpp"

namespace py = pybind11;

template <typename W>
void validate_inputs_cpp(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& sources_xyz,
    const py::array_t<W, py::array::c_style | py::array::forcecast>& weights,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& targets_xyz,
    double eps2,
    const py::array_t<int64_t, py::array::c_style>& collision_source_index
) {
    if (sources_xyz.ndim() != 2 || sources_xyz.shape(1) != 3) {
        throw py::value_error("sources_xyz must have shape (N, 3)");
    }
    if (targets_xyz.ndim() != 2 || targets_xyz.shape(1) != 3) {
        throw py::value_error("targets_xyz must have shape (M, 3)");
    }
    if (weights.ndim() != 1 || weights.shape(0) != sources_xyz.shape(0)) {
        throw py::value_error("weights length must match N (matching sources_xyz)");
    }
    if (collision_source_index.ndim() != 1 || collision_source_index.shape(0) != targets_xyz.shape(0)) {
        throw py::value_error("collision_source_index length must match M (matching targets_xyz)");
    }
    if (eps2 < 0.0) {
        throw py::value_error("eps2 must be non-negative");
    }
}

template <typename W>
void validate_inputs_parametric_cpp(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& sources_xyz,
    const py::array_t<W, py::array::c_style | py::array::forcecast>& weights,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& targets_xyz,
    double eps2,
    const py::array_t<int64_t, py::array::c_style>& collision_source_index
) {
    if (sources_xyz.ndim() != 2 || sources_xyz.shape(1) != 3) {
        throw py::value_error("sources_xyz must have shape (N, 3)");
    }
    if (targets_xyz.ndim() != 2 || targets_xyz.shape(1) != 3) {
        throw py::value_error("targets_xyz must have shape (M, 3)");
    }
    if (weights.ndim() != 2 || weights.shape(0) != sources_xyz.shape(0)) {
        throw py::value_error("weights must have shape (N, P_flat) with N matching sources_xyz");
    }
    if (collision_source_index.ndim() != 1 || collision_source_index.shape(0) != targets_xyz.shape(0)) {
        throw py::value_error("collision_source_index length must match M (matching targets_xyz)");
    }
    if (eps2 < 0.0) {
        throw py::value_error("eps2 must be non-negative");
    }
}

template <typename W>
void validate_inputs_dipole_cpp(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& sources_xyz,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& mass,
    const py::array_t<W, py::array::c_style | py::array::forcecast>& disp,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& targets_xyz,
    double eps2,
    const py::array_t<int64_t, py::array::c_style>& collision_source_index
) {
    if (sources_xyz.ndim() != 2 || sources_xyz.shape(1) != 3) {
        throw py::value_error("sources_xyz must have shape (N, 3)");
    }
    if (targets_xyz.ndim() != 2 || targets_xyz.shape(1) != 3) {
        throw py::value_error("targets_xyz must have shape (M, 3)");
    }
    if (mass.ndim() != 1 || mass.shape(0) != sources_xyz.shape(0)) {
        throw py::value_error("mass length must match N (matching sources_xyz)");
    }
    if (disp.ndim() != 2 || disp.shape(0) != sources_xyz.shape(0) || disp.shape(1) != 3) {
        throw py::value_error("disp must have shape (N, 3) with N matching sources_xyz");
    }
    if (collision_source_index.ndim() != 1 || collision_source_index.shape(0) != targets_xyz.shape(0)) {
        throw py::value_error("collision_source_index length must match M (matching targets_xyz)");
    }
    if (eps2 < 0.0) {
        throw py::value_error("eps2 must be non-negative");
    }
}

template <typename W>
py::array_t<W> potential_contract_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> sources_xyz,
    py::array_t<W, py::array::c_style | py::array::forcecast> weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> targets_xyz,
    double eps2,
    py::array_t<int64_t, py::array::c_style> collision_source_index
) {
    validate_inputs_cpp(sources_xyz, weights, targets_xyz, eps2, collision_source_index);

    int64_t n = sources_xyz.shape(0);
    int64_t m = targets_xyz.shape(0);

    py::array_t<W> out(m);
    if (m == 0 || n == 0) {
        std::fill(out.mutable_data(), out.mutable_data() + out.size(), static_cast<W>(0));
        return out;
    }

    auto r_sources = sources_xyz.unchecked<2>();
    auto r_weights = weights.template unchecked<1>();
    auto r_targets = targets_xyz.unchecked<2>();
    auto r_collision = collision_source_index.mutable_unchecked<1>();

    W* out_ptr = out.mutable_data();

    {
        py::gil_scoped_release release;
        potential_contract_impl<W>(
            r_sources.data(0, 0),
            r_weights.data(0),
            r_targets.data(0, 0),
            m,
            n,
            eps2,
            r_collision.mutable_data(0),
            out_ptr
        );
    }

    return out;
}

template <typename W>
py::array_t<W> acceleration_contract_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> sources_xyz,
    py::array_t<W, py::array::c_style | py::array::forcecast> weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> targets_xyz,
    double eps2,
    py::array_t<int64_t, py::array::c_style> collision_source_index
) {
    validate_inputs_cpp(sources_xyz, weights, targets_xyz, eps2, collision_source_index);

    int64_t n = sources_xyz.shape(0);
    int64_t m = targets_xyz.shape(0);

    py::array_t<W> out({m, static_cast<int64_t>(3)});
    if (m == 0 || n == 0) {
        std::fill(out.mutable_data(), out.mutable_data() + out.size(), static_cast<W>(0));
        return out;
    }

    auto r_sources = sources_xyz.unchecked<2>();
    auto r_weights = weights.template unchecked<1>();
    auto r_targets = targets_xyz.unchecked<2>();
    auto r_collision = collision_source_index.mutable_unchecked<1>();

    W* out_ptr = out.mutable_data();

    {
        py::gil_scoped_release release;
        acceleration_contract_impl<W>(
            r_sources.data(0, 0),
            r_weights.data(0),
            r_targets.data(0, 0),
            m,
            n,
            eps2,
            r_collision.mutable_data(0),
            out_ptr
        );
    }

    return out;
}

template <typename W>
py::array_t<W> gradient_contract_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> sources_xyz,
    py::array_t<W, py::array::c_style | py::array::forcecast> weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> targets_xyz,
    double eps2,
    py::array_t<int64_t, py::array::c_style> collision_source_index
) {
    validate_inputs_cpp(sources_xyz, weights, targets_xyz, eps2, collision_source_index);

    auto r_sources_shape = sources_xyz.shape();
    auto r_targets_shape = targets_xyz.shape();
    int64_t n = r_sources_shape[0];
    int64_t m = r_targets_shape[0];

    py::array_t<W> out({m, static_cast<int64_t>(3), static_cast<int64_t>(3)});
    if (m == 0 || n == 0) {
        std::fill(out.mutable_data(), out.mutable_data() + out.size(), static_cast<W>(0));
        return out;
    }

    auto r_sources = sources_xyz.unchecked<2>();
    auto r_weights = weights.template unchecked<1>();
    auto r_targets = targets_xyz.unchecked<2>();
    auto r_collision = collision_source_index.mutable_unchecked<1>();

    W* out_ptr = out.mutable_data();

    {
        py::gil_scoped_release release;
        gradient_contract_impl<W>(
            r_sources.data(0, 0),
            r_weights.data(0),
            r_targets.data(0, 0),
            m,
            n,
            eps2,
            r_collision.mutable_data(0),
            out_ptr
        );
    }

    return out;
}

template <typename W>
py::array_t<W> dipole_contract_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> sources_xyz,
    py::array_t<double, py::array::c_style | py::array::forcecast> mass,
    py::array_t<W, py::array::c_style | py::array::forcecast> disp,
    py::array_t<double, py::array::c_style | py::array::forcecast> targets_xyz,
    double eps2,
    py::array_t<int64_t, py::array::c_style> collision_source_index
) {
    validate_inputs_dipole_cpp(sources_xyz, mass, disp, targets_xyz, eps2, collision_source_index);

    int64_t n = sources_xyz.shape(0);
    int64_t m = targets_xyz.shape(0);

    py::array_t<W> out({m, static_cast<int64_t>(3)});
    if (m == 0 || n == 0) {
        std::fill(out.mutable_data(), out.mutable_data() + out.size(), static_cast<W>(0));
        return out;
    }

    auto r_sources = sources_xyz.unchecked<2>();
    auto r_mass = mass.unchecked<1>();
    auto r_disp = disp.template unchecked<2>();
    auto r_targets = targets_xyz.unchecked<2>();
    auto r_collision = collision_source_index.mutable_unchecked<1>();

    W* out_ptr = out.mutable_data();

    {
        py::gil_scoped_release release;
        dipole_contract_impl<W>(
            r_sources.data(0, 0),
            r_mass.data(0),
            r_disp.data(0, 0),
            r_targets.data(0, 0),
            m,
            n,
            eps2,
            r_collision.mutable_data(0),
            out_ptr
        );
    }

    return out;
}

// ---------------------------------------------------------------------------
// Parametric wrappers: weights arrive as (N, P_flat); outputs are
// (M, P_flat[, 3][, 3, 3]). The output is zero-filled before accumulation.
// ---------------------------------------------------------------------------

template <typename W>
py::array_t<W> potential_contract_parametric_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> sources_xyz,
    py::array_t<W, py::array::c_style | py::array::forcecast> weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> targets_xyz,
    double eps2,
    py::array_t<int64_t, py::array::c_style> collision_source_index
) {
    validate_inputs_parametric_cpp(sources_xyz, weights, targets_xyz, eps2, collision_source_index);

    int64_t n = sources_xyz.shape(0);
    int64_t m = targets_xyz.shape(0);
    int64_t p = weights.shape(1);

    py::array_t<W> out({m, p});
    W* out_ptr = out.mutable_data();
    std::fill(out_ptr, out_ptr + out.size(), static_cast<W>(0));
    if (m == 0 || n == 0 || p == 0) {
        return out;
    }

    auto r_sources = sources_xyz.unchecked<2>();
    auto r_weights = weights.template unchecked<2>();
    auto r_targets = targets_xyz.unchecked<2>();
    auto r_collision = collision_source_index.mutable_unchecked<1>();

    {
        py::gil_scoped_release release;
        potential_contract_parametric_impl<W>(
            r_sources.data(0, 0),
            r_weights.data(0, 0),
            r_targets.data(0, 0),
            m,
            n,
            p,
            eps2,
            r_collision.mutable_data(0),
            out_ptr
        );
    }

    return out;
}

template <typename W>
py::array_t<W> acceleration_contract_parametric_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> sources_xyz,
    py::array_t<W, py::array::c_style | py::array::forcecast> weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> targets_xyz,
    double eps2,
    py::array_t<int64_t, py::array::c_style> collision_source_index
) {
    validate_inputs_parametric_cpp(sources_xyz, weights, targets_xyz, eps2, collision_source_index);

    int64_t n = sources_xyz.shape(0);
    int64_t m = targets_xyz.shape(0);
    int64_t p = weights.shape(1);

    py::array_t<W> out({m, p, static_cast<int64_t>(3)});
    W* out_ptr = out.mutable_data();
    std::fill(out_ptr, out_ptr + out.size(), static_cast<W>(0));
    if (m == 0 || n == 0 || p == 0) {
        return out;
    }

    auto r_sources = sources_xyz.unchecked<2>();
    auto r_weights = weights.template unchecked<2>();
    auto r_targets = targets_xyz.unchecked<2>();
    auto r_collision = collision_source_index.mutable_unchecked<1>();

    {
        py::gil_scoped_release release;
        acceleration_contract_parametric_impl<W>(
            r_sources.data(0, 0),
            r_weights.data(0, 0),
            r_targets.data(0, 0),
            m,
            n,
            p,
            eps2,
            r_collision.mutable_data(0),
            out_ptr
        );
    }

    return out;
}

template <typename W>
py::array_t<W> gradient_contract_parametric_wrapper(
    py::array_t<double, py::array::c_style | py::array::forcecast> sources_xyz,
    py::array_t<W, py::array::c_style | py::array::forcecast> weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> targets_xyz,
    double eps2,
    py::array_t<int64_t, py::array::c_style> collision_source_index
) {
    validate_inputs_parametric_cpp(sources_xyz, weights, targets_xyz, eps2, collision_source_index);

    int64_t n = sources_xyz.shape(0);
    int64_t m = targets_xyz.shape(0);
    int64_t p = weights.shape(1);

    py::array_t<W> out({m, p, static_cast<int64_t>(3), static_cast<int64_t>(3)});
    W* out_ptr = out.mutable_data();
    std::fill(out_ptr, out_ptr + out.size(), static_cast<W>(0));
    if (m == 0 || n == 0 || p == 0) {
        return out;
    }

    auto r_sources = sources_xyz.unchecked<2>();
    auto r_weights = weights.template unchecked<2>();
    auto r_targets = targets_xyz.unchecked<2>();
    auto r_collision = collision_source_index.mutable_unchecked<1>();

    {
        py::gil_scoped_release release;
        gradient_contract_parametric_impl<W>(
            r_sources.data(0, 0),
            r_weights.data(0, 0),
            r_targets.data(0, 0),
            m,
            n,
            p,
            eps2,
            r_collision.mutable_data(0),
            out_ptr
        );
    }

    return out;
}

void validate_inputs_planewave_cpp(
    const py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>& weights,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& positions_xyz,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& directions_xyz,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& wavenumbers
) {
    if (positions_xyz.ndim() != 2 || positions_xyz.shape(1) != 3) {
        throw py::value_error("positions_xyz must have shape (N, 3)");
    }
    if (directions_xyz.ndim() != 2 || directions_xyz.shape(1) != 3) {
        throw py::value_error("directions_xyz must have shape (Ndir, 3)");
    }
    if (weights.ndim() != 1 || weights.shape(0) != positions_xyz.shape(0)) {
        throw py::value_error("weights length must match positions_xyz count");
    }
    if (wavenumbers.ndim() != 1) {
        throw py::value_error("wavenumbers must be a 1D array");
    }
}

py::array_t<std::complex<double>> plane_wave_weight_factor_wrapper(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> weights,
    py::array_t<double, py::array::c_style | py::array::forcecast> positions_xyz,
    py::array_t<double, py::array::c_style | py::array::forcecast> directions_xyz,
    py::array_t<double, py::array::c_style | py::array::forcecast> wavenumbers,
    double phase0,
    int64_t sign
) {
    validate_inputs_planewave_cpp(weights, positions_xyz, directions_xyz, wavenumbers);

    int64_t n = positions_xyz.shape(0);
    int64_t ndir = directions_xyz.shape(0);
    int64_t nf = wavenumbers.shape(0);

    py::array_t<std::complex<double>> out({n, ndir, nf});
    std::complex<double>* out_ptr = out.mutable_data();

    if (n == 0 || ndir == 0 || nf == 0) {
        std::fill(out_ptr, out_ptr + out.size(), std::complex<double>(0.0, 0.0));
        return out;
    }

    auto r_weights = weights.unchecked<1>();
    auto r_positions = positions_xyz.unchecked<2>();
    auto r_directions = directions_xyz.unchecked<2>();
    auto r_wavenumbers = wavenumbers.unchecked<1>();

    {
        py::gil_scoped_release release;
        plane_wave_weight_factor_impl<std::complex<double>>(
            r_weights.data(0),
            r_positions.data(0, 0),
            r_directions.data(0, 0),
            r_wavenumbers.data(0),
            n,
            ndir,
            nf,
            phase0,
            sign,
            out_ptr
        );
    }

    return out;
}

PYBIND11_MODULE(_cpp_backend, m) {
    m.doc() = "gragra C++ gravity kernels backend";

    m.def("potential_contract_real", &potential_contract_wrapper<double>,
          "Compute potential contraction for real weights");
    m.def("potential_contract_complex", &potential_contract_wrapper<std::complex<double>>,
          "Compute potential contraction for complex weights");

    m.def("acceleration_contract_real", &acceleration_contract_wrapper<double>,
          "Compute acceleration contraction for real weights");
    m.def("acceleration_contract_complex", &acceleration_contract_wrapper<std::complex<double>>,
          "Compute acceleration contraction for complex weights");

    m.def("gradient_contract_real", &gradient_contract_wrapper<double>,
          "Compute gravity gradient contraction for real weights");
    m.def("gradient_contract_complex", &gradient_contract_wrapper<std::complex<double>>,
          "Compute gravity gradient contraction for complex weights");

    m.def("dipole_contract_real", &dipole_contract_wrapper<double>,
          "Compute vector-weight (dipole/total) contraction for real displacement");
    m.def("dipole_contract_complex", &dipole_contract_wrapper<std::complex<double>>,
          "Compute vector-weight (dipole/total) contraction for complex displacement");

    m.def("potential_contract_parametric_real",
          &potential_contract_parametric_wrapper<double>,
          "Compute parametric potential contraction for real weights");
    m.def("potential_contract_parametric_complex",
          &potential_contract_parametric_wrapper<std::complex<double>>,
          "Compute parametric potential contraction for complex weights");

    m.def("acceleration_contract_parametric_real",
          &acceleration_contract_parametric_wrapper<double>,
          "Compute parametric acceleration contraction for real weights");
    m.def("acceleration_contract_parametric_complex",
          &acceleration_contract_parametric_wrapper<std::complex<double>>,
          "Compute parametric acceleration contraction for complex weights");

    m.def("gradient_contract_parametric_real",
          &gradient_contract_parametric_wrapper<double>,
          "Compute parametric gravity gradient contraction for real weights");
    m.def("gradient_contract_parametric_complex",
          &gradient_contract_parametric_wrapper<std::complex<double>>,
          "Compute parametric gravity gradient contraction for complex weights");

    m.def("plane_wave_weight_factor", &plane_wave_weight_factor_wrapper,
          "Compute plane wave weight factor");
}
