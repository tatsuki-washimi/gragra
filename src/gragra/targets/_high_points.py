"""Owned high-level target coordinates with stable user IDs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy import units as u

from gragra.geometry._frame import Frame


@dataclass(frozen=True, eq=False, init=False)
class TargetPoints:
    """Target positions as a length Quantity of shape ``(N, 3)``.

    Equal positions may have distinct IDs. Input arrays are copied to immutable
    bytes; a returned NumPy view cannot be made writable.
    """

    ids: tuple[str, ...]
    frame: Frame
    _position_bytes: bytes
    _count: int

    def __init__(
        self, positions: u.Quantity, ids: tuple[str, ...], frame: Frame
    ) -> None:
        if not isinstance(frame, Frame):
            raise TypeError("frame must be a Frame")
        if not isinstance(positions, u.Quantity):
            raise TypeError("positions must be an astropy Quantity with length units")
        raw = np.asarray(positions.to_value(u.m))
        if np.iscomplexobj(raw):
            raise ValueError("positions must contain only real coordinates")
        values = np.asarray(raw, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("positions must contain only finite coordinates")
        if isinstance(ids, (str, bytes)):
            raise TypeError("ids must be a sequence of complete ID strings")
        try:
            stable_ids = tuple(ids)
        except TypeError as exc:
            raise TypeError("ids must be a sequence of strings") from exc
        if (
            len(stable_ids) != len(values)
            or any(not isinstance(item, str) or not item for item in stable_ids)
            or len(set(stable_ids)) != len(stable_ids)
        ):
            raise ValueError(
                "ids must match positions and contain unique nonempty strings"
            )
        object.__setattr__(self, "ids", stable_ids)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "_position_bytes", values.copy().tobytes())
        object.__setattr__(self, "_count", len(values))

    @property
    def positions_m(self) -> np.ndarray:
        return np.frombuffer(self._position_bytes, dtype=np.float64).reshape(
            self._count, 3
        )

    @property
    def positions(self) -> u.Quantity:
        return self.positions_m.copy() * u.m

    def __len__(self) -> int:
        return self._count
