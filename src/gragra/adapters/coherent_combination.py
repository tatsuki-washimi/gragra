# ruff: noqa: N806, N803
"""Coherent combination adapter for bulk and wall Newtonian Noise."""

from dataclasses import dataclass

import numpy as np

from gragra._arrays import _check_no_bool, as_xyz_table, as_xyz_vector
from gragra.coherent.weights import plane_wave_mass_weights
from gragra.constants import G_SI
from gragra.kernels.parametric import _validate_chunk_size
from gragra.observables.coherent import coherent_projected_acceleration
from gragra.waves.directions import normalize_directions


def _validate_wnorm(wnorm: list | tuple | np.ndarray, n_dir: int) -> np.ndarray:
    """Validate normalised Lebedev quadrature weights and return them as float64.

    wnorm may be negative: valid Lebedev orders (13/25/27) carry
    negative weights; consistent with nn-sus which has no
    non-negativity guard. As a consequence power()/psd_* are not
    guaranteed non-negative.
    """
    _check_no_bool(wnorm, "wnorm")

    try:
        w = np.asarray(wnorm, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"wnorm must be numeric convertible to float64: {e}") from e

    if w.ndim != 1:
        raise ValueError(f"wnorm must be 1D, got ndim={w.ndim}")
    if len(w) != n_dir:
        raise ValueError(f"wnorm length ({len(w)}) must match khat count ({n_dir})")
    if not np.isfinite(w).all():
        raise ValueError("wnorm must contain only finite numbers")
    if not np.isclose(np.sum(w), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError(f"wnorm sum must be close to 1.0, got {np.sum(w)}")

    return w


@dataclass(frozen=True)
class DirectionalComplexKernel:
    """Per-direction complex transfer amplitude on a Lebedev grid.

    Attributes
    ----------
    khat : np.ndarray
        Direction unit vectors, shape (Ndir, 3).
    wnorm : np.ndarray
        Normalised Lebedev quadrature weights, shape (Ndir,).
    H : np.ndarray
        Complex per-direction transfer amplitude, shape (Ndir,).
    """

    khat: np.ndarray
    wnorm: np.ndarray
    H: np.ndarray

    def __post_init__(self):
        khat_normalized = normalize_directions(self.khat)
        n_dir = khat_normalized.shape[0]

        w = _validate_wnorm(self.wnorm, n_dir)

        _check_no_bool(self.H, "H")

        # Verify H
        try:
            h = np.asarray(self.H, dtype=np.complex128)
        except (ValueError, TypeError) as e:
            raise TypeError(f"H must be numeric convertible to complex128: {e}") from e

        if h.ndim != 1:
            raise ValueError(f"H must be 1D, got ndim={h.ndim}")
        if len(h) != n_dir:
            raise ValueError(f"H length ({len(h)}) must match khat count ({n_dir})")
        if not np.isfinite(h).all():
            raise ValueError("H must contain only finite numbers")

        w = w.copy()
        h = h.copy()
        w.setflags(write=False)
        h.setflags(write=False)

        object.__setattr__(self, "khat", khat_normalized)
        object.__setattr__(self, "wnorm", w)
        object.__setattr__(self, "H", h)

    def power(self) -> float:
        """Direction-averaged power sum(wnorm * |H|^2)."""
        return float(np.sum(self.wnorm * np.abs(self.H) ** 2))


@dataclass(frozen=True)
class DirectionStats:
    """Statistics of H_tot amplitude over directions."""

    max: float
    median: float
    min: float

    def __post_init__(self):
        for name in ("max", "median", "min"):
            val = getattr(self, name)
            if isinstance(val, bool):
                raise TypeError(f"{name} must not be bool")
            if not isinstance(val, (int, float, np.integer, np.floating)):
                raise TypeError(f"{name} must be a float")
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(val))


@dataclass(frozen=True)
class CoherentTotal:
    """Coherently combined bulk and wall transfer amplitudes and power."""

    psd_total: float
    psd_bulk: float
    psd_wall: float
    H_tot: np.ndarray
    H_bulk: np.ndarray
    H_wall: np.ndarray
    direction_stats: DirectionStats

    def __post_init__(self):
        for name in ("H_tot", "H_bulk", "H_wall"):
            arr = getattr(self, name)
            _check_no_bool(arr, name)
            try:
                arr = np.asarray(arr, dtype=np.complex128)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"{name} must be numeric convertible to complex128: {e}"
                ) from e
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1D, got ndim={arr.ndim}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite numbers")
            arr = arr.copy()
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

        if (
            self.H_tot.shape != self.H_bulk.shape
            or self.H_tot.shape != self.H_wall.shape
        ):
            raise ValueError(
                "H arrays must have the same shape: "
                f"H_tot {self.H_tot.shape}, "
                f"H_bulk {self.H_bulk.shape}, "
                f"H_wall {self.H_wall.shape}"
            )

        for name in ("psd_total", "psd_bulk", "psd_wall"):
            val = getattr(self, name)
            if isinstance(val, bool):
                raise TypeError(f"{name} must not be bool")
            if not isinstance(val, (int, float, np.integer, np.floating)):
                raise TypeError(f"{name} must be a float")
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(val))

        if not isinstance(self.direction_stats, DirectionStats):
            raise TypeError("direction_stats must be a DirectionStats instance")

    def as_dict(self) -> dict:
        """Return a dictionary representation compatible with nnsus structure."""
        return {
            "psd_total": self.psd_total,
            "psd_bulk": self.psd_bulk,
            "psd_wall": self.psd_wall,
            "H_tot": self.H_tot,
            "H_bulk": self.H_bulk,
            "H_wall": self.H_wall,
            "direction_stats": {
                "max": self.direction_stats.max,
                "median": self.direction_stats.median,
                "min": self.direction_stats.min,
            },
        }


def volume_transfer_kernel(
    positions_m: list | tuple | np.ndarray,
    volumes_m3: list | tuple | np.ndarray,
    *,
    r0_m: list | tuple | np.ndarray,
    e_beam: list | tuple | np.ndarray,
    khat: list | tuple | np.ndarray,
    wnorm: list | tuple | np.ndarray,
    wavenumber_rad_m: float | int,
    chunk_size: int | None = None,
    backend: str = "numpy",
) -> DirectionalComplexKernel:
    """Calculate the bare bulk volume complex transfer kernel (G-free).

    P-wave bulk only: this reproduces the P-wave branch of nn-sus
    ``VolumeIntegrator.H_complex_one_freq``. SH/SV/custom waves are
    incompressible and contribute zero bulk density perturbation, so they
    are not handled here (they remain a nn-sus-side concern, together with
    wall ``H_wall`` and surface/Lebedev integrators).

    Expects positions_m of shape (N, 3), volumes_m3 of shape (N,).

    ``chunk_size`` is forwarded to the internal ``coherent_projected_acceleration``
    (parametric) call, where it is the VRAM-tiling parameter on the ``"cupy"``
    backend (``docs/architecture/gpu_backend_adr.md`` §6) and a source-axis
    chunk size elsewhere. It has no effect on the internal
    ``plane_wave_mass_weights`` call: that path always materializes its full
    ``(N, Ndir, Nf)`` array regardless of ``chunk_size`` (see
    ``docs/design/cupy_backend.md`` C-3 section).
    """
    if isinstance(wavenumber_rad_m, bool):
        raise TypeError("wavenumber_rad_m must not be bool")
    if not isinstance(wavenumber_rad_m, (int, float, np.integer, np.floating)):
        raise TypeError("wavenumber_rad_m must be a numeric value")
    if not np.isfinite(wavenumber_rad_m):
        raise ValueError("wavenumber_rad_m must be finite")
    if wavenumber_rad_m <= 0.0:
        raise ValueError(f"wavenumber_rad_m must be positive, got {wavenumber_rad_m}")

    pos = as_xyz_table("positions_m", positions_m)

    _check_no_bool(volumes_m3, "volumes_m3")
    try:
        vol = np.asarray(volumes_m3, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise TypeError(f"volumes_m3 must be numeric: {e}") from e
    if vol.ndim != 1:
        raise ValueError(f"volumes_m3 must be 1D, got ndim={vol.ndim}")
    if len(vol) != len(pos):
        raise ValueError(
            f"Length of volumes_m3 ({len(vol)}) "
            f"must match positions_m count ({len(pos)})"
        )
    if not np.isfinite(vol).all():
        raise ValueError("volumes_m3 must contain only finite numbers")
    if (vol <= 0.0).any():
        raise ValueError("volumes_m3 must contain only positive values")

    r0 = as_xyz_vector("r0_m", r0_m)

    eb = as_xyz_vector("e_beam", e_beam)
    eb_len = np.linalg.norm(eb)
    if eb_len == 0.0:
        raise ValueError("e_beam cannot be a zero vector")
    eb_unit = eb / eb_len

    khat_norm = normalize_directions(khat)
    _validate_wnorm(wnorm, khat_norm.shape[0])

    # Fail fast on an invalid chunk_size before any cupy import / VRAM
    # preflight / GPU compute happens below (Codex round-3 review, PR #57
    # Major 2): chunk_size was previously only discovered deep inside the
    # tail-call chain (coherent_projected_acceleration -> parametric kernel
    # validation), by which point GPU work had already started.
    _validate_chunk_size(chunk_size)

    if backend == "cupy":
        try:
            import cupy as cp
        except ImportError as e:

            class CupyImportValueError(ImportError, ValueError):
                pass

            raise CupyImportValueError(
                "cupy is not installed in the current environment. Install the "
                "CuPy wheel matching your CUDA/ROCm runtime (e.g. `pip install "
                "'cupy-cuda12x[ctk]'` for CUDA 12.x without a system CUDA "
                "Toolkit, or plain `cupy-cuda12x` when one is installed) to use "
                "the cupy backend. Note that a CUDA-capable GPU and driver are "
                "required at runtime; see docs/design/cupy_backend.md."
            ) from e

        from gragra.kernels._cupy_inverse_square import _ensure_cuda_device
        from gragra.kernels._cupy_planewave import _estimate_planewave_vram_bytes

        _ensure_cuda_device()

        n_val = int(len(pos))
        n_dir = int(khat_norm.shape[0])
        n_f = 1
        m_val = 1

        extra_bytes = (
            4 * 16 * max(n_val, m_val) * n_dir * n_f + 16 * m_val * n_dir * n_f
        )
        b_total = _estimate_planewave_vram_bytes(
            n_val, n_dir, n_f, 8, extra_bytes=extra_bytes
        )

        free_vram, _ = cp.cuda.Device().mem_info
        if b_total > 0.5 * free_vram:
            raise MemoryError(
                f"Insufficient VRAM for volume transfer kernel: estimated "
                f"{b_total} bytes, free VRAM is {free_vram} bytes."
            )

    # Host-side (N,3) allocation deliberately placed after the cupy VRAM
    # preflight above, so a MemoryError is raised deterministically before
    # any host allocation for the cupy backend (Codex review, PR #57).
    dR = pos - r0[None, :]

    w = plane_wave_mass_weights(
        vol,
        dR,
        khat_norm,
        wavenumber_rad_m,
        sign=1,
        chunk_size=chunk_size,
        backend=backend,
    )

    proj = coherent_projected_acceleration(
        dR, w, [[0.0, 0.0, 0.0]], eb_unit, chunk_size=chunk_size, backend=backend
    )

    H = proj[0, :, 0] / G_SI

    return DirectionalComplexKernel(khat_norm, wnorm, H)


def _assert_same_directions(
    a: DirectionalComplexKernel,
    b: DirectionalComplexKernel,
    *,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    if a.khat.shape != b.khat.shape:
        raise ValueError(
            "Direction grids differ in shape "
            f"({a.khat.shape} vs {b.khat.shape}); coherent sum requires the "
            "same Lebedev order for bulk and wall kernels."
        )
    if not np.allclose(a.khat, b.khat, rtol=rtol, atol=atol):
        raise ValueError(
            "Direction unit vectors (khat) differ between bulk and wall "
            "kernels; coherent sum requires the identical Lebedev grid."
        )
    if not np.allclose(a.wnorm, b.wnorm, rtol=rtol, atol=atol):
        raise ValueError(
            "Lebedev weights (wnorm) differ between bulk and wall kernels; "
            "coherent sum requires the identical Lebedev grid."
        )


def coherent_total(
    h_vol: DirectionalComplexKernel,
    h_wall: DirectionalComplexKernel,
    *,
    Grho: float,
    k: float,
) -> CoherentTotal:
    """Coherently combine the bulk volume kernel and the wall surface kernel."""
    if not isinstance(h_vol, DirectionalComplexKernel):
        raise TypeError("h_vol must be a DirectionalComplexKernel instance")
    if not isinstance(h_wall, DirectionalComplexKernel):
        raise TypeError("h_wall must be a DirectionalComplexKernel instance")

    _assert_same_directions(h_vol, h_wall)

    if isinstance(Grho, bool):
        raise TypeError("Grho must not be bool")
    if not isinstance(Grho, (int, float, np.integer, np.floating)):
        raise TypeError("Grho must be a numeric value")
    if not np.isfinite(Grho):
        raise ValueError("Grho must be finite")

    if isinstance(k, bool):
        raise TypeError("k must not be bool")
    if not isinstance(k, (int, float, np.integer, np.floating)):
        raise TypeError("k must be a numeric value")
    if not np.isfinite(k):
        raise ValueError("k must be finite")
    if k <= 0.0:
        raise ValueError(f"k must be positive, got {k}")

    H_bulk_phys = Grho * (1j * k) * h_vol.H
    H_wall_amp = h_wall.H
    H_tot = H_bulk_phys + H_wall_amp

    wnorm = h_vol.wnorm
    psd_total = float(np.sum(wnorm * np.abs(H_tot) ** 2))
    psd_bulk = float(np.sum(wnorm * np.abs(H_bulk_phys) ** 2))
    psd_wall = float(np.sum(wnorm * np.abs(H_wall_amp) ** 2))

    abs_tot = np.abs(H_tot)
    stats = DirectionStats(
        max=float(np.max(abs_tot)),
        median=float(np.median(abs_tot)),
        min=float(np.min(abs_tot)),
    )

    return CoherentTotal(
        psd_total=psd_total,
        psd_bulk=psd_bulk,
        psd_wall=psd_wall,
        H_tot=H_tot,
        H_bulk=H_bulk_phys,
        H_wall=H_wall_amp,
        direction_stats=stats,
    )
