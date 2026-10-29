"""Immutable descriptors for spectral quantity conventions.

The descriptor records interpretation only. Values, units, axes, labels, timing,
and estimator settings belong to the payload or its surrounding result. See
``docs/architecture/spectral_representation_conventions.md`` for the full
conventions.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, TypeAlias


class Sidedness(StrEnum):
    """Frequency storage convention for a Fourier transform.

    As a ``StrEnum``, a member compares equal to its string value. The
    descriptor constructor still requires a ``Sidedness`` member, so pass
    ``Sidedness.ONE_SIDED`` rather than the bare string ``"one_sided"``.
    """

    ONE_SIDED = "one_sided"
    TWO_SIDED = "two_sided"


class FiniteRecordInterpretation(StrEnum):
    """Interpretation of a finite amplitude-spectrum record.

    As a ``StrEnum``, a member compares equal to its string value. The
    descriptor constructor still requires a ``FiniteRecordInterpretation``
    member, so pass ``FiniteRecordInterpretation.PERIODIC`` rather than the
    bare string ``"periodic"``.
    """

    PERIODIC = "periodic"
    TRANSIENT = "transient"


@dataclass(frozen=True, slots=True, kw_only=True)
class FourierTransformRepresentation:
    """Describe a Fourier transform with physical quantity unit Q·s.

    The forward transform uses the negative exponential and the inverse uses
    the positive exponential. ``ONE_SIDED`` omits negative frequencies for real
    input without doubling stored coefficients. For a transformed array
    ``y_j`` with interval ``dt``, length ``M``, DFT coefficients
    ``X_k = Σ_j y_j exp(−2πi k j/M)``, and ``f_k = k/(M dt)``, the finite
    transform convention is
    ``x̃_d(f_k) = dt exp[−2πi f_k(t₀−t_ref)] X_k``. Here ``t₀`` is the time
    of index zero in the array actually transformed, after cropping or padding;
    ``t_ref`` is the caller's absolute phase reference. This descriptor stores
    neither time and performs no transform computation.
    """

    kind: ClassVar[str] = "fourier_transform"
    sidedness: Sidedness = Sidedness.TWO_SIDED

    def __post_init__(self) -> None:
        if not isinstance(self.sidedness, Sidedness):
            raise TypeError("sidedness must be a Sidedness member")


@dataclass(frozen=True, slots=True, kw_only=True)
class AmplitudeSpectrumRepresentation:
    """Describe a one-sided complex amplitude spectrum with unit Q.

    For DFT coefficients ``X_k`` of the array actually transformed, the
    coefficient is ``A_k = (c_k/M) exp[−2πi f_k(t₀−t_ref)] X_k``. Here ``M``
    is the transform length, ``t₀`` is the time of index zero in that array
    after cropping or padding, ``t_ref`` is the caller's absolute phase
    reference, and ``c_k`` is one at DC and even-length Nyquist and two at
    interior bins. Positive-exponential synthesis reconstructs the transformed
    array. PERIODIC means periodic extension of the finite record; TRANSIENT
    means zero extension. A coefficient is a peak amplitude only for an
    unpadded, bin-centered component. The descriptor stores none of these
    record conditions or times.
    """

    kind: ClassVar[str] = "amplitude_spectrum"
    record_interpretation: FiniteRecordInterpretation

    def __post_init__(self) -> None:
        if not isinstance(self.record_interpretation, FiniteRecordInterpretation):
            raise TypeError(
                "record_interpretation must be a FiniteRecordInterpretation member"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class PhasorRepresentation:
    """Describe complex peak phasors P in x(t)=Re[P exp(-2πif(t-t_ref))].

    The physical unit is Q. At an internal Fourier bin, P is the conjugate of
    the amplitude-spectrum coefficient. Phasors generally describe positive
    frequencies; a DC phasor, if used, is real. No reference time is stored.
    """

    kind: ClassVar[str] = "phasor"


@dataclass(frozen=True, slots=True, kw_only=True)
class PSDRepresentation:
    """Describe a nonnegative one-sided power spectral density, unit Q²/Hz.

    This density is for real signals; DC and even-length Nyquist bins are not
    doubled. Its conversion to squared amplitude requires a boxcar record,
    detrending disabled, no zero padding (M=N), and matching frequency grid.
    The descriptor does not define an estimator or store its conditions.
    """

    kind: ClassVar[str] = "psd"


@dataclass(frozen=True, slots=True, kw_only=True)
class ASDRepresentation:
    """Describe the nonnegative square root of a one-sided PSD, unit Q/√Hz.

    It applies to real signals and retains undoubled DC and even-length
    Nyquist endpoints. It represents density, not the square root of a
    spectrum-scaled estimate; estimator settings are not stored here.
    """

    kind: ClassVar[str] = "asd"


@dataclass(frozen=True, slots=True, kw_only=True)
class CSDRepresentation:
    """Describe one-sided cross spectral density, unit Q_x Q_y/Hz.

    For real signals, the Fourier-coefficient convention conjugates the left
    transform: ``conj(FT(x)) · FT(y)``. The corresponding phasor product is
    ``P_x · conj(P_y)``, with unit Q_x Q_y; it is not itself a density. For
    internal bins, density conversion divides that product by ``2Δf``. DC and
    even-length Nyquist bins are not doubled and use
    ``conj(A_x) · A_y / Δf``. Conversion requires boxcar, no detrending, M=N,
    and a common sample grid and phase reference. This descriptor stores no
    signals or estimator conditions.
    """

    kind: ClassVar[str] = "csd"


SpectralRepresentation: TypeAlias = (
    FourierTransformRepresentation
    | AmplitudeSpectrumRepresentation
    | PhasorRepresentation
    | PSDRepresentation
    | ASDRepresentation
    | CSDRepresentation
)


__all__ = [
    "Sidedness",
    "FiniteRecordInterpretation",
    "FourierTransformRepresentation",
    "AmplitudeSpectrumRepresentation",
    "PhasorRepresentation",
    "PSDRepresentation",
    "ASDRepresentation",
    "CSDRepresentation",
    "SpectralRepresentation",
]
