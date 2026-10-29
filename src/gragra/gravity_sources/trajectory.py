"""Prescribed point and rigid-body motion source descriptions."""

from dataclasses import dataclass

import numpy as np

from gragra.array_types import FloatArray


def _reject_bool_or_complex(value, name: str) -> None:
    """Reject values that a float conversion would silently change."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not contain bool")
    if isinstance(value, (complex, np.complexfloating)):
        raise TypeError(f"{name} must be real-valued; complex values are rejected")

    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_bool_or_complex(item, name)
        return

    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            raise TypeError(f"{name} must not contain bool")
        if value.dtype.kind == "c":
            raise TypeError(f"{name} must be real-valued; complex values are rejected")
        if value.dtype.kind == "O":
            for item in value.flat:
                _reject_bool_or_complex(item, name)
        return

    if isinstance(value, np.generic):
        if value.dtype.kind == "b":
            raise TypeError(f"{name} must not contain bool")
        if value.dtype.kind == "c":
            raise TypeError(f"{name} must be real-valued; complex values are rejected")


def _as_real_array(
    name: str, value, *, ndim: int, tail_shape: tuple[int, ...] = ()
) -> FloatArray:
    """Convert a finite real array to an independent read-only float64 copy."""
    _reject_bool_or_complex(value, name)
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be convertible to float64: {error}") from error
    if arr.ndim != ndim or (tail_shape and arr.shape[-len(tail_shape) :] != tail_shape):
        expected = f"({', '.join(['P'] + [str(n) for n in tail_shape])})"
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite numbers")
    arr = arr.astype(np.float64, copy=True)
    arr.setflags(write=False)
    return arr


def _as_mass(value) -> float:
    """Validate one positive, finite, real scalar mass."""
    _reject_bool_or_complex(value, "mass_kg")
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"mass_kg must be a real scalar: {error}") from error
    if arr.ndim != 0:
        raise ValueError(f"mass_kg must be a scalar, got shape {arr.shape}")
    try:
        mass = float(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError) as error:
        raise TypeError(f"mass_kg must be a real scalar: {error}") from error
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_kg must be positive and finite")
    return mass


def _as_time(value) -> FloatArray:
    """Validate a nonempty, finite, strictly increasing time vector."""
    times = _as_real_array("time_s", value, ndim=1)
    if times.size < 1:
        raise ValueError("time_s must contain at least one frame")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("time_s must be strictly increasing")
    return times


@dataclass(frozen=True)
class PrescribedPointTrajectory:
    """A positive point mass sampled at prescribed times and positions."""

    mass_kg: float
    time_s: FloatArray
    positions_m: FloatArray

    def __post_init__(self) -> None:
        mass = _as_mass(self.mass_kg)
        times = _as_time(self.time_s)
        positions = _as_real_array(
            "positions_m", self.positions_m, ndim=2, tail_shape=(3,)
        )
        if len(positions) != len(times):
            raise ValueError(
                "time_s and positions_m must have the same number of frames: "
                f"{len(times)} != {len(positions)}"
            )
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "positions_m", positions)


@dataclass(frozen=True)
class PrescribedRigidTransform:
    """A sampled rigid transform ``x_world = R(t) x_body + c(t)``."""

    time_s: FloatArray
    translations_m: FloatArray
    rotation_matrices: FloatArray

    def __post_init__(self) -> None:
        times = _as_time(self.time_s)
        translations = _as_real_array(
            "translations_m", self.translations_m, ndim=2, tail_shape=(3,)
        )
        rotations = _as_real_array(
            "rotation_matrices", self.rotation_matrices, ndim=3, tail_shape=(3, 3)
        )
        frame_count = len(times)
        if len(translations) != frame_count or len(rotations) != frame_count:
            raise ValueError(
                "time_s, translations_m, and rotation_matrices must have the same "
                "number of frames"
            )

        identity = np.eye(3, dtype=np.float64)
        for index, rotation in enumerate(rotations):
            if not np.allclose(rotation @ rotation.T, identity, atol=1e-10, rtol=0.0):
                raise ValueError(f"rotation_matrices[{index}] must satisfy R R.T = I")
            determinant = float(np.linalg.det(rotation))
            if determinant <= 0.0 or not np.isclose(
                determinant, 1.0, atol=1e-10, rtol=0.0
            ):
                raise ValueError(f"rotation_matrices[{index}] must have determinant 1")

        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "translations_m", translations)
        object.__setattr__(self, "rotation_matrices", rotations)


def _read_csv_rows(
    filepath_or_buffer, delimiter: str
) -> tuple[list[dict[str, str]], set[str]]:
    import csv
    from pathlib import Path

    if isinstance(filepath_or_buffer, (str, Path)):
        with open(filepath_or_buffer, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = list(reader)
            fieldnames = set(reader.fieldnames or [])
    else:
        reader = csv.DictReader(filepath_or_buffer, delimiter=delimiter)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])
    return rows, fieldnames


def read_prescribed_point_trajectory_csv(
    filepath_or_buffer,
    *,
    mass_kg: float | None = None,
    mass_column: str | None = None,
    time_column: str = "time_s",
    position_columns: tuple[str, str, str] = ("x_m", "y_m", "z_m"),
    delimiter: str = ",",
) -> PrescribedPointTrajectory:
    """Read measured or prescribed point trajectory from a CSV/TSV file.

    Parameters
    ----------
    filepath_or_buffer : str, Path, or file-like object
        Path to CSV/TSV file or open buffer.
    mass_kg : float, optional
        Fixed point mass in kg. Mutually exclusive with ``mass_column``.
    mass_column : str, optional
        Name of column containing mass values (must be identical and
        positive across all rows).
    time_column : str, default "time_s"
        Name of column containing sampling timestamps in seconds.
    position_columns : tuple[str, str, str], default ("x_m", "y_m", "z_m")
        Column names for 3D cartesian coordinates (x, y, z) in meters.
    delimiter : str, default ","
        Field delimiter character.

    Returns
    -------
    PrescribedPointTrajectory
        Validated trajectory with strictly increasing non-uniform time frames.
    """
    rows, fieldnames = _read_csv_rows(filepath_or_buffer, delimiter)

    if time_column not in fieldnames:
        raise ValueError(f"CSV is missing time column: '{time_column}'")

    for col in position_columns:
        if col not in fieldnames:
            raise ValueError(f"CSV is missing position column: '{col}'")

    if mass_kg is not None and mass_column is not None:
        raise ValueError("Cannot specify both mass_kg and mass_column")

    if mass_kg is None and mass_column is None:
        raise ValueError("Must specify either mass_kg or mass_column")

    if mass_column is not None:
        if mass_column not in fieldnames:
            raise ValueError(f"CSV is missing mass column: '{mass_column}'")
        masses = np.array([float(r[mass_column]) for r in rows], dtype=np.float64)
        if len(masses) == 0:
            raise ValueError("CSV contains no data rows")
        if not np.allclose(masses, masses[0], rtol=1e-12, atol=1e-15):
            raise ValueError(
                f"Mass values in '{mass_column}' must be constant across frames"
            )
        resolved_mass = float(masses[0])
    else:
        resolved_mass = float(mass_kg)

    time_s = np.array([float(r[time_column]) for r in rows], dtype=np.float64)
    positions_m = np.array(
        [[float(r[col]) for col in position_columns] for r in rows],
        dtype=np.float64,
    )

    return PrescribedPointTrajectory(
        mass_kg=resolved_mass,
        time_s=time_s,
        positions_m=positions_m,
    )


def read_prescribed_rigid_transform_csv(
    filepath_or_buffer,
    *,
    time_column: str = "time_s",
    translation_columns: tuple[str, str, str] = ("tx_m", "ty_m", "tz_m"),
    rotation_columns: tuple[str, ...] | None = None,
    delimiter: str = ",",
) -> PrescribedRigidTransform:
    """Read measured or prescribed rigid motion transforms from a CSV/TSV file.

    Parameters
    ----------
    filepath_or_buffer : str, Path, or file-like object
        Path to CSV/TSV file or open buffer.
    time_column : str, default "time_s"
        Name of column containing sampling timestamps in seconds.
    translation_columns : tuple[str, str, str], default ("tx_m", "ty_m", "tz_m")
        Column names for translation vector in meters.
    rotation_columns : tuple[str, ...], optional
        Column names for 3x3 rotation matrix entries (9 columns:
        r00, r01, r02, r10, r11, r12, r20, r21, r22) or identity if None.
    delimiter : str, default ","
        Field delimiter character.

    Returns
    -------
    PrescribedRigidTransform
        Validated rigid transform series with strictly increasing time frames.
    """
    rows, fieldnames = _read_csv_rows(filepath_or_buffer, delimiter)

    if time_column not in fieldnames:
        raise ValueError(f"CSV is missing time column: '{time_column}'")

    for col in translation_columns:
        if col not in fieldnames:
            raise ValueError(f"CSV is missing translation column: '{col}'")

    time_s = np.array([float(r[time_column]) for r in rows], dtype=np.float64)
    translations_m = np.array(
        [[float(r[col]) for col in translation_columns] for r in rows],
        dtype=np.float64,
    )

    n_frames = len(rows)
    if rotation_columns is None:
        rotations = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], n_frames, axis=0)
    else:
        if len(rotation_columns) != 9:
            raise ValueError(
                "rotation_columns must contain exactly 9 column names for a 3x3 matrix"
            )
        for col in rotation_columns:
            if col not in fieldnames:
                raise ValueError(f"CSV is missing rotation column: '{col}'")
        rot_flat = np.array(
            [[float(r[col]) for col in rotation_columns] for r in rows],
            dtype=np.float64,
        )
        rotations = rot_flat.reshape(-1, 3, 3)

    return PrescribedRigidTransform(
        time_s=time_s,
        translations_m=translations_m,
        rotation_matrices=rotations,
    )


__all__ = [
    "PrescribedPointTrajectory",
    "PrescribedRigidTransform",
    "read_prescribed_point_trajectory_csv",
    "read_prescribed_rigid_transform_csv",
]
