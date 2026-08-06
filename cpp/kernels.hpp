#ifndef GRAGRA_KERNELS_HPP
#define GRAGRA_KERNELS_HPP

#include <cmath>
#include <complex>
#include <cstdint>

template <typename W>
void potential_contract_impl(
    const double* sources_xyz,
    const W* weights,
    const double* targets_xyz,
    int64_t m,
    int64_t n,
    double eps2,
    int64_t* collision_source_index,
    W* out
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; ++i) {
        double tx = targets_xyz[i * 3 + 0];
        double ty = targets_xyz[i * 3 + 1];
        double tz = targets_xyz[i * 3 + 2];
        W s = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            double dx = sources_xyz[j * 3 + 0] - tx;
            double dy = sources_xyz[j * 3 + 1] - ty;
            double dz = sources_xyz[j * 3 + 2] - tz;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (eps2 == 0.0 && r2 == 0.0) {
                if (collision_source_index[i] == -1) {
                    collision_source_index[i] = j;
                }
                continue;
            }
            double r_eff = std::sqrt(r2 + eps2);
            s += (-1.0 / r_eff) * weights[j];
        }
        out[i] = s;
    }
}

template <typename W>
void acceleration_contract_impl(
    const double* sources_xyz,
    const W* weights,
    const double* targets_xyz,
    int64_t m,
    int64_t n,
    double eps2,
    int64_t* collision_source_index,
    W* out
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; ++i) {
        double tx = targets_xyz[i * 3 + 0];
        double ty = targets_xyz[i * 3 + 1];
        double tz = targets_xyz[i * 3 + 2];
        W ax = 0.0;
        W ay = 0.0;
        W az = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            double dx = sources_xyz[j * 3 + 0] - tx;
            double dy = sources_xyz[j * 3 + 1] - ty;
            double dz = sources_xyz[j * 3 + 2] - tz;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (eps2 == 0.0 && r2 == 0.0) {
                if (collision_source_index[i] == -1) {
                    collision_source_index[i] = j;
                }
                continue;
            }
            double r_eff = std::sqrt(r2 + eps2);
            double r_eff3 = (r2 + eps2) * r_eff;
            W factor = weights[j] / r_eff3;
            ax += dx * factor;
            ay += dy * factor;
            az += dz * factor;
        }
        out[i * 3 + 0] = ax;
        out[i * 3 + 1] = ay;
        out[i * 3 + 2] = az;
    }
}

template <typename W>
void gradient_contract_impl(
    const double* sources_xyz,
    const W* weights,
    const double* targets_xyz,
    int64_t m,
    int64_t n,
    double eps2,
    int64_t* collision_source_index,
    W* out
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; ++i) {
        double tx = targets_xyz[i * 3 + 0];
        double ty = targets_xyz[i * 3 + 1];
        double tz = targets_xyz[i * 3 + 2];
        W g00 = 0.0; W g01 = 0.0; W g02 = 0.0;
        W g10 = 0.0; W g11 = 0.0; W g12 = 0.0;
        W g20 = 0.0; W g21 = 0.0; W g22 = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            double dx = sources_xyz[j * 3 + 0] - tx;
            double dy = sources_xyz[j * 3 + 1] - ty;
            double dz = sources_xyz[j * 3 + 2] - tz;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (eps2 == 0.0 && r2 == 0.0) {
                if (collision_source_index[i] == -1) {
                    collision_source_index[i] = j;
                }
                continue;
            }
            double r_eff = std::sqrt(r2 + eps2);
            double r_eff3 = (r2 + eps2) * r_eff;
            double r_eff5 = r_eff3 * (r2 + eps2);

            W w = weights[j];
            double inv_r3 = 1.0 / r_eff3;
            double inv_r5 = 3.0 / r_eff5;

            W inv_r3_w = inv_r3 * w;
            W inv_r5_w = inv_r5 * w;

            g00 += dx * dx * inv_r5_w - inv_r3_w;
            g01 += dx * dy * inv_r5_w;
            g02 += dx * dz * inv_r5_w;

            g10 += dy * dx * inv_r5_w;
            g11 += dy * dy * inv_r5_w - inv_r3_w;
            g12 += dy * dz * inv_r5_w;

            g20 += dz * dx * inv_r5_w;
            g21 += dz * dy * inv_r5_w;
            g22 += dz * dz * inv_r5_w - inv_r3_w;
        }
        out[i * 9 + 0] = g00; out[i * 9 + 1] = g01; out[i * 9 + 2] = g02;
        out[i * 9 + 3] = g10; out[i * 9 + 4] = g11; out[i * 9 + 5] = g12;
        out[i * 9 + 6] = g20; out[i * 9 + 7] = g21; out[i * 9 + 8] = g22;
    }
}

// Vector-weight (dipole/total) contraction: mass is always real (a physical
// background mass, kernel_contract.md Section 2.1.1), while disp/out carry
// the templated type W (real or complex displacement). K_T = 3*dd/r_eff^5 -
// I/r_eff^3 is symmetric, so only 6 independent components are computed per
// source (unlike gradient_contract_impl's 9) and immediately contracted
// against disp[j] into the 3-component accumulator -- the full 3x3 tensor is
// never needed, only the resulting vector.
template <typename W>
void dipole_contract_impl(
    const double* sources_xyz,
    const double* mass,
    const W* disp,
    const double* targets_xyz,
    int64_t m,
    int64_t n,
    double eps2,
    int64_t* collision_source_index,
    W* out
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; ++i) {
        double tx = targets_xyz[i * 3 + 0];
        double ty = targets_xyz[i * 3 + 1];
        double tz = targets_xyz[i * 3 + 2];
        W ax = 0.0;
        W ay = 0.0;
        W az = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            double dx = sources_xyz[j * 3 + 0] - tx;
            double dy = sources_xyz[j * 3 + 1] - ty;
            double dz = sources_xyz[j * 3 + 2] - tz;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (eps2 == 0.0 && r2 == 0.0) {
                if (collision_source_index[i] == -1) {
                    collision_source_index[i] = j;
                }
                continue;
            }
            double r_eff = std::sqrt(r2 + eps2);
            double r_eff3 = (r2 + eps2) * r_eff;
            double r_eff5 = r_eff3 * (r2 + eps2);
            double inv_r3 = 1.0 / r_eff3;
            double inv_r5 = 3.0 / r_eff5;

            double txx = dx * dx * inv_r5 - inv_r3;
            double txy = dx * dy * inv_r5;
            double txz = dx * dz * inv_r5;
            double tyy = dy * dy * inv_r5 - inv_r3;
            double tyz = dy * dz * inv_r5;
            double tzz = dz * dz * inv_r5 - inv_r3;

            double mj = mass[j];
            W d0 = disp[j * 3 + 0];
            W d1 = disp[j * 3 + 1];
            W d2 = disp[j * 3 + 2];

            ax += -mj * (txx * d0 + txy * d1 + txz * d2);
            ay += -mj * (txy * d0 + tyy * d1 + tyz * d2);
            az += -mj * (txz * d0 + tyz * d1 + tzz * d2);
        }
        out[i * 3 + 0] = ax;
        out[i * 3 + 1] = ay;
        out[i * 3 + 2] = az;
    }
}

// ---------------------------------------------------------------------------
// Parametric contractions: weights are flattened to (N, P_flat); the geometric
// factor for each (target, source) pair is computed once and reused across the
// P_flat axis. Outputs are laid out as (M, P_flat[, 3][, 3, 3]) row-major.
// ---------------------------------------------------------------------------

template <typename W>
void potential_contract_parametric_impl(
    const double* sources_xyz,
    const W* weights,
    const double* targets_xyz,
    int64_t m,
    int64_t n,
    int64_t p,
    double eps2,
    int64_t* collision_source_index,
    W* out
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; ++i) {
        double tx = targets_xyz[i * 3 + 0];
        double ty = targets_xyz[i * 3 + 1];
        double tz = targets_xyz[i * 3 + 2];
        for (int64_t j = 0; j < n; ++j) {
            double dx = sources_xyz[j * 3 + 0] - tx;
            double dy = sources_xyz[j * 3 + 1] - ty;
            double dz = sources_xyz[j * 3 + 2] - tz;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (eps2 == 0.0 && r2 == 0.0) {
                if (collision_source_index[i] == -1) {
                    collision_source_index[i] = j;
                }
                continue;
            }
            double r_eff = std::sqrt(r2 + eps2);
            double k = -1.0 / r_eff;
            for (int64_t q = 0; q < p; ++q) {
                out[i * p + q] += k * weights[j * p + q];
            }
        }
    }
}

template <typename W>
void acceleration_contract_parametric_impl(
    const double* sources_xyz,
    const W* weights,
    const double* targets_xyz,
    int64_t m,
    int64_t n,
    int64_t p,
    double eps2,
    int64_t* collision_source_index,
    W* out
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; ++i) {
        double tx = targets_xyz[i * 3 + 0];
        double ty = targets_xyz[i * 3 + 1];
        double tz = targets_xyz[i * 3 + 2];
        for (int64_t j = 0; j < n; ++j) {
            double dx = sources_xyz[j * 3 + 0] - tx;
            double dy = sources_xyz[j * 3 + 1] - ty;
            double dz = sources_xyz[j * 3 + 2] - tz;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (eps2 == 0.0 && r2 == 0.0) {
                if (collision_source_index[i] == -1) {
                    collision_source_index[i] = j;
                }
                continue;
            }
            double r_eff = std::sqrt(r2 + eps2);
            double r_eff3 = (r2 + eps2) * r_eff;
            double kx = dx / r_eff3;
            double ky = dy / r_eff3;
            double kz = dz / r_eff3;
            for (int64_t q = 0; q < p; ++q) {
                W wq = weights[j * p + q];
                out[(i * p + q) * 3 + 0] += kx * wq;
                out[(i * p + q) * 3 + 1] += ky * wq;
                out[(i * p + q) * 3 + 2] += kz * wq;
            }
        }
    }
}

template <typename W>
void gradient_contract_parametric_impl(
    const double* sources_xyz,
    const W* weights,
    const double* targets_xyz,
    int64_t m,
    int64_t n,
    int64_t p,
    double eps2,
    int64_t* collision_source_index,
    W* out
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < m; ++i) {
        double tx = targets_xyz[i * 3 + 0];
        double ty = targets_xyz[i * 3 + 1];
        double tz = targets_xyz[i * 3 + 2];
        for (int64_t j = 0; j < n; ++j) {
            double dx = sources_xyz[j * 3 + 0] - tx;
            double dy = sources_xyz[j * 3 + 1] - ty;
            double dz = sources_xyz[j * 3 + 2] - tz;
            double r2 = dx * dx + dy * dy + dz * dz;
            if (eps2 == 0.0 && r2 == 0.0) {
                if (collision_source_index[i] == -1) {
                    collision_source_index[i] = j;
                }
                continue;
            }
            double r_eff = std::sqrt(r2 + eps2);
            double r_eff3 = (r2 + eps2) * r_eff;
            double r_eff5 = r_eff3 * (r2 + eps2);
            double inv_r3 = 1.0 / r_eff3;
            double inv_r5 = 3.0 / r_eff5;
            double t00 = dx * dx * inv_r5 - inv_r3;
            double t01 = dx * dy * inv_r5;
            double t02 = dx * dz * inv_r5;
            double t11 = dy * dy * inv_r5 - inv_r3;
            double t12 = dy * dz * inv_r5;
            double t22 = dz * dz * inv_r5 - inv_r3;
            for (int64_t q = 0; q < p; ++q) {
                W wq = weights[j * p + q];
                out[(i * p + q) * 9 + 0] += t00 * wq;
                out[(i * p + q) * 9 + 1] += t01 * wq;
                out[(i * p + q) * 9 + 2] += t02 * wq;
                out[(i * p + q) * 9 + 3] += t01 * wq;
                out[(i * p + q) * 9 + 4] += t11 * wq;
                out[(i * p + q) * 9 + 5] += t12 * wq;
                out[(i * p + q) * 9 + 6] += t02 * wq;
                out[(i * p + q) * 9 + 7] += t12 * wq;
                out[(i * p + q) * 9 + 8] += t22 * wq;
            }
        }
    }
}

template <typename W>
void plane_wave_weight_factor_impl(
    const W* weights,
    const double* positions_xyz,
    const double* directions_xyz,
    const double* wavenumbers,
    int64_t n,
    int64_t ndir,
    int64_t nf,
    double phase0,
    int64_t sign,
    std::complex<double>* out
) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        W w_val = weights[i];
        double px = positions_xyz[i * 3 + 0];
        double py = positions_xyz[i * 3 + 1];
        double pz = positions_xyz[i * 3 + 2];

        for (int64_t j = 0; j < ndir; ++j) {
            double dx = directions_xyz[j * 3 + 0];
            double dy = directions_xyz[j * 3 + 1];
            double dz = directions_xyz[j * 3 + 2];

            double dot = px * dx + py * dy + pz * dz;

            for (int64_t f = 0; f < nf; ++f) {
                double phase = sign * (dot * wavenumbers[f] + phase0);
                std::complex<double> exp_factor(std::cos(phase), std::sin(phase));
                out[(i * ndir + j) * nf + f] = w_val * exp_factor;
            }
        }
    }
}

#endif // GRAGRA_KERNELS_HPP
