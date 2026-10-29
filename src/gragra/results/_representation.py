"""Static field interpretation alongside the existing spectral descriptors."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class StaticRepresentation:
    """Describe a static real field with ordinary Quantity payloads."""

    kind: ClassVar[str] = "static"


@dataclass(frozen=True, slots=True)
class TimeSeriesRepresentation:
    """Describe real observables sampled on one uniform GPS time axis."""

    kind: ClassVar[str] = "time_series"
