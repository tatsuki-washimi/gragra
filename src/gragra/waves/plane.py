"""Harmonic plane wave representations and finite-body evaluation API (SI).

Provides PlaneWave, PlanePWave, and PlaneWaveSources supporting multi-direction,
multi-frequency batches with P/SH/SV/custom polarizations and three observables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from gragra._arrays import as_xyz_table, as_xyz_vector
from gragra.constants import G_SI
from gragra.kernels.parametric import (
    acceleration_contract_parametric,
    gradient_contract_parametric,
)
from gragra.waves.directions import normalize_directions
from gragra.waves.displacement import plane_wave_displacement


@dataclass(frozen=True)
class PlaneWaveSources:
    """Evaluated plane-wave mass and displacement perturbation bundle.

    Carries sampled background masses and multi-direction/multi-frequency
    displacement fields for exact linearized gravitational perturbation evaluation.
    """

    positions_m: np.ndarray
    mass_kg: np.ndarray
    displacement_m: np.ndarray  # shape (N, D, F, 3), complex128
    directions: np.ndarray  # shape (D, 3)
    frequencies_hz: np.ndarray  # shape (F,)
    wavenumbers_rad_m: np.ndarray  # shape (F,)
    phase_origin_m: np.ndarray
    phase0_rad: float
    t_ref_s: float
    coordinate_frame: str = "world"
    unit_system: str = "SI"

    def __post_init__(self):
        # Read-only defensive flags
        for attr in (
            "positions_m",
            "mass_kg",
            "displacement_m",
            "directions",
            "frequencies_hz",
            "wavenumbers_rad_m",
            "phase_origin_m",
        ):
            arr = getattr(self, attr)
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)

    def evaluate(
        self,
        targets_m: list | tuple | np.ndarray,
        *,
        observable: Literal["potential", "acceleration", "gradient"] = "acceleration",
        softening_m: float = 0.0,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        """Calculate complex response for the chosen gravitational observable.

        Parameters
        ----------
        targets_m : array_like of shape (M, 3)
            Evaluation points in meters.
        observable : {"potential", "acceleration", "gradient"}
            Observable quantity to compute.
        softening_m : float, default 0.0
            Plummer softening length.
        chunk_size : int, optional
            Target chunk size for memory control.

        Returns
        -------
        np.ndarray
            - "potential": shape (M, D, F), complex128
            - "acceleration": shape (M, D, F, 3), complex128
            - "gradient": shape (M, D, F, 3, 3), complex128
        """
        targets = as_xyz_table("targets_m", targets_m)
        sources = self.positions_m
        m_bg = self.mass_kg  # (N,)
        xi = self.displacement_m  # (N, D, F, 3)

        n_src, n_dir, n_freq, _ = xi.shape

        if observable == "potential":
            # Linearized mass-displacement potential perturbation:
            # delta Phi = -G * sum_j m_j (r_vec_j . xi_j) / r_j^3
            # In parametric contraction: effective vector weights w_vec = m_j * xi_j
            # Since delta Phi(r) = - grad(Phi_0) . xi, using acceleration kernel:
            # a_gfree has shape (M, D, F, 3) where
            # a_gfree_k = sum_j w_{j, ...} (targets_k - sources_{j, k}) / r^3
            # delta Phi = + G_SI * sum_alpha [ a_gfree_alpha(sources, m_j * xi_alpha) ]
            pot = np.zeros((len(targets), n_dir, n_freq), dtype=np.complex128)
            for alpha in range(3):
                w_alpha = (m_bg[:, None, None] * xi[:, :, :, alpha]).astype(
                    np.complex128
                )
                contract_alpha = acceleration_contract_parametric(
                    sources_xyz=sources,
                    weights=w_alpha,
                    targets_xyz=targets,
                    softening_m=softening_m,
                    chunk_size=chunk_size,
                )  # shape (M, D, F, 3)
                # contraction with the alpha-direction component of the kernel
                pot += G_SI * contract_alpha[:, :, :, alpha]

            pot.setflags(write=False)
            return pot

        if observable == "acceleration":
            # Linearized mass-displacement acceleration perturbation:
            # delta a_i = sum_alpha G_SI * gradient_contract_parametric[..., i, alpha]
            acc = np.zeros((len(targets), n_dir, n_freq, 3), dtype=np.complex128)
            for alpha in range(3):
                w_alpha = (m_bg[:, None, None] * xi[:, :, :, alpha]).astype(
                    np.complex128
                )
                contract_grad = gradient_contract_parametric(
                    sources_xyz=sources,
                    weights=w_alpha,
                    targets_xyz=targets,
                    softening_m=softening_m,
                    chunk_size=chunk_size,
                )  # shape (M, D, F, 3, 3), [targets, D, F, i, alpha]
                acc += G_SI * contract_grad[:, :, :, :, alpha]

            acc.setflags(write=False)
            return acc

        if observable == "gradient":
            # Numerical derivative of acceleration
            # For gradient of dipole field, finite-difference across targets:
            h = 1e-4
            grad = np.zeros((len(targets), n_dir, n_freq, 3, 3), dtype=np.complex128)
            for j in range(3):
                t_p = targets.copy()
                t_m = targets.copy()
                t_p[:, j] += h
                t_m[:, j] -= h
                a_p = self.evaluate(
                    t_p,
                    observable="acceleration",
                    softening_m=softening_m,
                    chunk_size=chunk_size,
                )
                a_m = self.evaluate(
                    t_m,
                    observable="acceleration",
                    softening_m=softening_m,
                    chunk_size=chunk_size,
                )
                grad[:, :, :, :, j] = (a_p - a_m) / (2.0 * h)

            grad.setflags(write=False)
            return grad

        raise ValueError(f"Unsupported observable: '{observable}'")


@dataclass(frozen=True)
class PlaneWave:
    """Harmonic plane wave description."""

    wave_type: Literal["P", "SH", "SV", "custom"]
    directions: np.ndarray  # shape (D, 3)
    frequencies_hz: np.ndarray  # shape (F,)
    wavenumbers_rad_m: np.ndarray  # shape (F,)
    amplitudes_m: np.ndarray  # shape (D, F) or broadcastable
    phase_origin_m: np.ndarray
    phase0_rad: float = 0.0
    t_ref_s: float = 0.0
    vertical_unit: tuple[float, float, float] = (0.0, 0.0, 1.0)
    polarization: np.ndarray | None = None

    def __post_init__(self):
        dirs = normalize_directions(self.directions)
        f = np.asarray(self.frequencies_hz, dtype=np.float64)
        k = np.asarray(self.wavenumbers_rad_m, dtype=np.float64)
        if f.ndim != 1 or k.ndim != 1 or len(f) != len(k):
            raise ValueError(
                "frequencies_hz and wavenumbers_rad_m must be 1D arrays of equal length"
            )
        if np.any(f <= 0.0) or np.any(k <= 0.0):
            raise ValueError("frequencies_hz and wavenumbers_rad_m must be positive")

        p_orig = as_xyz_vector("phase_origin_m", self.phase_origin_m)

        raw_amp = np.asarray(self.amplitudes_m)
        amp = (
            np.broadcast_to(raw_amp, (len(dirs), len(f)))
            .astype(np.complex128 if np.iscomplexobj(raw_amp) else np.float64)
            .copy()
        )

        dirs.setflags(write=False)
        f.setflags(write=False)
        k.setflags(write=False)
        amp.setflags(write=False)

        object.__setattr__(self, "directions", dirs)
        object.__setattr__(self, "frequencies_hz", f)
        object.__setattr__(self, "wavenumbers_rad_m", k)
        object.__setattr__(self, "amplitudes_m", amp)
        object.__setattr__(self, "phase_origin_m", p_orig)

    def evaluate_on(
        self,
        sources,
        *,
        coordinate_frame: str = "world",
    ) -> PlaneWaveSources:
        """Sample plane wave displacement on a point cloud or volume mesh."""
        if hasattr(sources, "positions_m") and hasattr(sources, "mass_kg"):
            pos = sources.positions_m
            mass = sources.mass_kg
        elif hasattr(sources, "positions_m") and hasattr(sources, "masses_kg"):
            pos = sources.positions_m
            mass = sources.masses_kg
        else:
            raise TypeError("sources must provide positions_m and mass_kg/masses_kg")

        # Evaluate displacement field: shape (N, D, F, 3)
        rel_pos = pos - self.phase_origin_m
        xi = plane_wave_displacement(
            positions_m=rel_pos,
            k_hat=self.directions,
            wavenumber_rad_m=self.wavenumbers_rad_m,
            wave_type=self.wave_type,
            amplitude_m=1.0,  # unit amplitude; scale below
            phase0=self.phase0_rad,
            vertical_unit=self.vertical_unit,
            polarization=self.polarization,
        )  # (N, D, F, 3)

        # Scale by amplitudes (D, F):
        xi = xi * self.amplitudes_m[None, :, :, None]

        return PlaneWaveSources(
            positions_m=pos,
            mass_kg=mass,
            displacement_m=xi,
            directions=self.directions,
            frequencies_hz=self.frequencies_hz,
            wavenumbers_rad_m=self.wavenumbers_rad_m,
            phase_origin_m=self.phase_origin_m,
            phase0_rad=self.phase0_rad,
            t_ref_s=self.t_ref_s,
            coordinate_frame=coordinate_frame,
        )


def PlanePWave(  # noqa: N802
    directions,
    frequencies_hz,
    wavenumbers_rad_m,
    *,
    amplitudes_m=1.0,
    phase_origin_m=(0.0, 0.0, 0.0),
    phase0_rad: float = 0.0,
    t_ref_s: float = 0.0,
) -> PlaneWave:
    """Convenience constructor for a compressional (P) plane wave."""
    return PlaneWave(
        wave_type="P",
        directions=directions,
        frequencies_hz=frequencies_hz,
        wavenumbers_rad_m=wavenumbers_rad_m,
        amplitudes_m=amplitudes_m,
        phase_origin_m=phase_origin_m,
        phase0_rad=phase0_rad,
        t_ref_s=t_ref_s,
    )


def directional_power_average(
    field: np.ndarray,
    weights_dir: np.ndarray | None = None,
    axis: int = 1,
) -> np.ndarray:
    """Compute component-wise directional auto-power average.

    Parameters
    ----------
    field : np.ndarray
        Complex field array with direction axis at ``axis`` (e.g. (M, D, F, ...)).
    weights_dir : np.ndarray, optional
        Non-negative direction weights of shape (D,).
        If None, uniform weighting is used.
    axis : int, default 1
        Direction axis index.

    Returns
    -------
    np.ndarray
        Real non-negative power array with direction axis reduced.
    """
    f = np.asarray(field)
    n_dir = f.shape[axis]
    if weights_dir is None:
        w = np.full(n_dir, 1.0 / n_dir, dtype=np.float64)
    else:
        w = np.asarray(weights_dir, dtype=np.float64)
        if w.shape != (n_dir,):
            raise ValueError(f"weights_dir must have shape ({n_dir},), got {w.shape}")
        if np.any(w < 0.0):
            raise ValueError("weights_dir must contain only non-negative values")
        w_sum = np.sum(w)
        if w_sum <= 0.0:
            raise ValueError("weights_dir sum must be positive")
        w = w / w_sum

    # Power is |field|^2
    power = np.abs(f) ** 2
    # Reshape weights for broadcasting
    shape_w = [1] * f.ndim
    shape_w[axis] = n_dir
    w_broadcast = w.reshape(shape_w)

    avg_power = np.sum(power * w_broadcast, axis=axis)
    avg_power.setflags(write=False)
    return avg_power
