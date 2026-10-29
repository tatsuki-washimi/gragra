"""Independent numerical oracles for the C3 spectral conventions.

These tests define the local transforms needed to exercise the conventions;
they do not add a production spectral calculation API.  SciPy is a required
independent reference for density estimates.  All time-domain phases use
relative times before multiplication, since the fixtures use GPS-scale epochs.
"""

import numpy as np
import pytest
from scipy import signal

from gragra.waves import plane_wave_displacement

_T0 = 1_400_000_000.0
_DT = 1.0 / 256.0


def _fold_factors(transform_length):
    factors = np.full(transform_length // 2 + 1, 2.0)
    factors[0] = 1.0
    if transform_length % 2 == 0:
        factors[-1] = 1.0
    return factors


def _amplitude_spectrum(samples, dt, t0, t_ref, transform_length=None):
    """Return one-sided synthesis coefficients with the C3 phase reference."""
    samples = np.asarray(samples, dtype=np.float64)
    length = len(samples) if transform_length is None else transform_length
    freqs = np.fft.rfftfreq(length, d=dt)
    raw = np.fft.rfft(samples, n=length)
    phase = np.exp(-2j * np.pi * freqs * (t0 - t_ref))
    return freqs, _fold_factors(length) * phase * raw / length


def _synthesize_amplitude(amplitudes, freqs, dt, t0, t_ref, length):
    sample_offsets = np.arange(length, dtype=np.float64) * dt + (t0 - t_ref)
    oscillators = np.exp(2j * np.pi * freqs[:, None] * sample_offsets[None, :])
    return np.real(np.sum(amplitudes[:, None] * oscillators, axis=0))


def _time_from_phasor(phasor, freq, sample_count, dt, t0, t_ref):
    offsets = np.arange(sample_count, dtype=np.float64) * dt + (t0 - t_ref)
    return np.real(phasor * np.exp(-2j * np.pi * freq * offsets))


def _assert_complex_not_close(left, right, *, min_distance=1e-3):
    assert np.max(np.abs(np.asarray(left) - np.asarray(right))) > min_distance


def _scipy_periodogram(samples, fs, nfft):
    return signal.periodogram(
        samples,
        fs=fs,
        window="boxcar",
        detrend=False,
        scaling="density",
        nfft=nfft,
    )


def _scipy_csd(x, y, fs, nperseg, *, noverlap=0):
    return signal.csd(
        x,
        y,
        fs=fs,
        window="boxcar",
        nperseg=nperseg,
        nfft=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        average="mean",
    )


def _scipy_welch(samples, fs, nperseg, noverlap):
    return signal.welch(
        samples,
        fs=fs,
        window="boxcar",
        nperseg=nperseg,
        nfft=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density",
        average="mean",
    )


def test_t1_periodic_bin_tone_has_the_declared_peak_amplitude_and_phase():
    """T1: a bin-centred real tone independently anchors A's sign and phase."""
    sample_count = 256
    dt = 1.0 / 1024.0
    bin_index = 23
    freq = bin_index / (sample_count * dt)
    amplitude = 1.7
    phase0 = 0.413
    t_ref = _T0 + 0.0037
    sample_offsets = np.arange(sample_count) * dt
    samples = amplitude * np.cos(2 * np.pi * freq * sample_offsets + phase0)

    freqs, actual = _amplitude_spectrum(samples, dt, _T0, t_ref)
    expected = amplitude * np.exp(1j * (phase0 + 2 * np.pi * freq * (t_ref - _T0)))
    np.testing.assert_allclose(actual[bin_index], expected, rtol=1e-12, atol=1e-12)
    assert freqs[bin_index] == pytest.approx(freq)
    np.testing.assert_allclose(np.delete(actual, bin_index), 0.0, atol=1e-12)

    # Extra conjugation or a reversed reference-time rotation must fail.
    _assert_complex_not_close(np.conjugate(actual[bin_index]), expected)
    wrong_sign = amplitude * np.exp(1j * (phase0 + 2 * np.pi * freq * (_T0 - t_ref)))
    _assert_complex_not_close(wrong_sign, expected)


@pytest.mark.parametrize(
    ("sample_count", "transform_length", "left_pad"),
    [(7, 7, 0), (8, 8, 0), (7, 12, 2), (8, 13, 3)],
)
def test_t2_real_amplitude_synthesis_round_trips_odd_even_and_zero_padding(
    sample_count, transform_length, left_pad
):
    """T2: inverse synthesis restores the entire transformed record and pads."""
    rng = np.random.default_rng(20_260_926 + sample_count + transform_length)
    source = rng.normal(size=sample_count) + 0.31
    right_pad = transform_length - sample_count - left_pad
    transformed_record = np.pad(source, (left_pad, right_pad))
    t0 = _T0 - left_pad * _DT
    t_ref = _T0 + 0.0029

    freqs, amplitudes = _amplitude_spectrum(
        transformed_record, _DT, t0, t_ref, transform_length
    )
    reconstructed = _synthesize_amplitude(
        amplitudes, freqs, _DT, t0, t_ref, transform_length
    )
    np.testing.assert_allclose(
        reconstructed, transformed_record, rtol=1e-12, atol=2e-12
    )
    np.testing.assert_allclose(reconstructed[:left_pad], 0.0, atol=2e-12)
    if right_pad:
        np.testing.assert_allclose(reconstructed[-right_pad:], 0.0, atol=2e-12)

    if transform_length > sample_count:
        raw = np.fft.rfft(transformed_record, n=transform_length)
        wrong_length_normalization = (
            _fold_factors(transform_length)
            * np.exp(-2j * np.pi * freqs * (t0 - t_ref))
            * raw
            / sample_count
        )
        _assert_complex_not_close(wrong_length_normalization, amplitudes)
        assert np.max(np.abs(wrong_length_normalization - amplitudes)) > 0.1


def test_t3_endpoint_folding_and_rotated_nyquist_still_reconstruct_exactly():
    """T3: only unpaired bins stay single; odd final bins remain doubled."""
    even_record = np.array([1.2 + 0.37 * (-1.0) ** n for n in range(8)])
    even_record += 0.8 * np.cos(2 * np.pi * np.arange(8) / 8 + 0.29)
    even_ref = _T0 + 0.0031  # off the sample grid: Nyquist A is complex
    even_freqs, even_amplitudes = _amplitude_spectrum(even_record, _DT, _T0, even_ref)
    even_raw = np.fft.rfft(even_record)
    np.testing.assert_allclose(even_amplitudes[0], even_raw[0] / 8, atol=1e-13)
    np.testing.assert_allclose(
        even_amplitudes[-1],
        even_raw[-1] / 8 * np.exp(-2j * np.pi * even_freqs[-1] * (_T0 - even_ref)),
        atol=1e-13,
    )
    assert abs(even_amplitudes[-1].imag) > 0.1
    even_reconstructed = _synthesize_amplitude(
        even_amplitudes, even_freqs, _DT, _T0, even_ref, 8
    )
    np.testing.assert_allclose(even_reconstructed, even_record, atol=1e-12)

    doubled_nyquist = even_amplitudes.copy()
    doubled_nyquist[-1] *= 2
    wrong_even = _synthesize_amplitude(
        doubled_nyquist, even_freqs, _DT, _T0, even_ref, 8
    )
    _assert_complex_not_close(wrong_even, even_record, min_distance=0.1)

    odd_record = np.array(
        [0.6 + 0.3 * np.cos(2 * np.pi * 4 * n / 9 + 0.47) for n in range(9)]
    )
    odd_ref = _T0 + 0.0027
    odd_freqs, odd_amplitudes = _amplitude_spectrum(odd_record, _DT, _T0, odd_ref)
    odd_raw = np.fft.rfft(odd_record)
    expected_last = (
        2 * odd_raw[-1] / 9 * np.exp(-2j * np.pi * odd_freqs[-1] * (_T0 - odd_ref))
    )
    np.testing.assert_allclose(odd_amplitudes[-1], expected_last, atol=1e-12)
    odd_reconstructed = _synthesize_amplitude(
        odd_amplitudes, odd_freqs, _DT, _T0, odd_ref, 9
    )
    np.testing.assert_allclose(odd_reconstructed, odd_record, atol=1e-12)

    underweighted_last = odd_amplitudes.copy()
    underweighted_last[-1] *= 0.5
    wrong_odd = _synthesize_amplitude(
        underweighted_last, odd_freqs, _DT, _T0, odd_ref, 9
    )
    _assert_complex_not_close(wrong_odd, odd_record, min_distance=0.1)


def test_t4_phasor_is_the_conjugate_and_matches_plane_wave_time_synthesis():
    """T4: P=A* recovers e^{-i omega t} amplitudes and the wave API output."""
    sample_count = 256
    dt = 1.0 / sample_count
    bin_index = 29
    freq = bin_index / (sample_count * dt)
    t_ref = _T0 + 0.0024
    expected_phasor = 1.3 * np.exp(0.43j)
    samples = _time_from_phasor(expected_phasor, freq, sample_count, dt, _T0, t_ref)
    _, amplitudes = _amplitude_spectrum(samples, dt, _T0, t_ref)
    recovered_phasor = np.conjugate(amplitudes[bin_index])
    np.testing.assert_allclose(
        recovered_phasor, expected_phasor, rtol=1e-12, atol=1e-12
    )
    # Negative control: P=A omits the required conjugation and has the wrong phase.
    _assert_complex_not_close(amplitudes[bin_index], expected_phasor)

    position = np.array([[0.5, -0.3, 0.2]])
    wave = plane_wave_displacement(
        position,
        [[1.0, 0.0, 0.0]],
        [1.4],
        "P",
        amplitude_m=2.1,
        phase0=0.61,
    )[0, 0, 0, 0]
    wave_samples = _time_from_phasor(wave, freq, sample_count, dt, _T0, t_ref)
    _, wave_amplitudes = _amplitude_spectrum(wave_samples, dt, _T0, t_ref)
    np.testing.assert_allclose(
        np.conjugate(wave_amplitudes[bin_index]), wave, rtol=1e-12, atol=1e-12
    )


def test_t5_reference_moves_amplitude_and_phasor_with_opposite_signs():
    """T5: changing t_ref changes representation phase, not the real record."""
    sample_count = 256
    dt = 1.0 / sample_count
    bin_index = 37
    freq = bin_index / (sample_count * dt)
    p0 = 0.91 * np.exp(0.73j)
    ref0 = _T0 + 0.0021
    tau = 0.0017
    ref1 = ref0 + tau
    actual_tau = ref1 - ref0
    samples = _time_from_phasor(p0, freq, sample_count, dt, _T0, ref0)

    _, amplitude0 = _amplitude_spectrum(samples, dt, _T0, ref0)
    _, amplitude1 = _amplitude_spectrum(samples, dt, _T0, ref1)
    expected_amplitude1 = amplitude0[bin_index] * np.exp(2j * np.pi * freq * actual_tau)
    expected_phasor1 = np.conjugate(amplitude0[bin_index]) * np.exp(
        -2j * np.pi * freq * actual_tau
    )
    np.testing.assert_allclose(amplitude1[bin_index], expected_amplitude1, atol=1e-12)
    np.testing.assert_allclose(
        np.conjugate(amplitude1[bin_index]), expected_phasor1, atol=1e-12
    )

    offsets = np.arange(sample_count) * dt + (_T0 - ref0)
    stable_phase = np.exp(-2j * np.pi * freq * offsets)
    absolute_times = _T0 + np.arange(sample_count) * dt
    unstable_phase = np.exp(-2j * np.pi * (freq * absolute_times - freq * ref0))
    # GPS-scale products lose several micro-radians here; this exceeds the
    # float64 oracle tolerances used for the convention transforms by orders.
    assert np.max(np.abs(stable_phase - unstable_phase)) > 1e-6


def test_t6_gaussian_fourier_transform_matches_closed_form_without_one_sided_doubling():
    """T6: Gaussian FT matches its integral and is independent of FFT length."""
    sample_count = 1024
    fs = 1000.0
    dt = 1.0 / fs
    sigma = 0.04
    center = 0.512
    t_ref = _T0 + 0.0023
    relative_times = np.arange(sample_count, dtype=np.float64) * dt
    samples = np.exp(-0.5 * ((relative_times - center) / sigma) ** 2)
    # The finite record boundaries are each 12.8 sigma from the pulse center;
    # the final sample lies 12.775 sigma from it because samples end at N-1.
    assert min(center, sample_count * dt - center) / sigma == pytest.approx(12.8)

    transforms = {}
    amplitudes = {}
    for length in (1024, 2048, 4096):
        freqs = np.fft.rfftfreq(length, d=dt)
        discrete = (
            dt
            * np.exp(-2j * np.pi * freqs * (_T0 - t_ref))
            * np.fft.rfft(samples, n=length)
        )
        transforms[length] = discrete
        amps = _fold_factors(length) * np.fft.rfft(samples, n=length) / length
        amps *= np.exp(-2j * np.pi * freqs * (_T0 - t_ref))
        amplitudes[length] = amps

    shared_freqs = np.fft.rfftfreq(1024, d=dt)
    shared_max_index = np.searchsorted(shared_freqs, 100.0)
    for length in (2048, 4096):
        stride = length // sample_count
        np.testing.assert_allclose(
            transforms[length][::stride][: shared_max_index + 1],
            transforms[1024][: shared_max_index + 1],
            rtol=1e-12,
            atol=1e-14,
        )

    analytic = (
        sigma
        * np.sqrt(2 * np.pi)
        * np.exp(-2 * np.pi**2 * sigma**2 * shared_freqs**2)
        * np.exp(-2j * np.pi * shared_freqs * (center - (t_ref - _T0)))
    )
    # At dt/sigma=0.025, with the 12.8-sigma record support and f<=100.5859375 Hz,
    # float64 error is below rtol=1e-10, atol=1e-12 for this Riemann sum.
    np.testing.assert_allclose(
        transforms[1024][: shared_max_index + 1],
        analytic[: shared_max_index + 1],
        rtol=1e-10,
        atol=1e-12,
    )
    # A one-sided Fourier transform stores the positive-frequency coefficient;
    # unlike a one-sided amplitude spectrum, it does not double it.
    interior = 10
    assert 0 < interior < sample_count // 2
    assert (
        abs(2 * transforms[1024][interior] - analytic[interior])
        > abs(analytic[interior]) * 0.9
    )

    shared_index_2048 = 2 * interior
    np.testing.assert_allclose(
        transforms[1024][interior], transforms[2048][shared_index_2048], atol=1e-14
    )
    ratio = amplitudes[2048][shared_index_2048] / amplitudes[1024][interior]
    assert ratio == pytest.approx(0.5 + 0j, rel=1e-12, abs=1e-12)


def test_t7_boxcar_periodogram_matches_scipy_and_parseval():
    """T7: unpadded density agrees with SciPy and integrates to mean square."""
    sample_count = 16
    fs = 64.0
    dt = 1.0 / fs
    n = np.arange(sample_count)
    samples = 1.4 + 0.8 * np.cos(2 * np.pi * 3 * n / sample_count + 0.38)
    samples += 0.33 * (-1.0) ** n
    t_ref = _T0 + 0.0027

    freqs, amplitudes = _amplitude_spectrum(samples, dt, _T0, t_ref)
    scipy_freqs, scipy_psd = _scipy_periodogram(samples, fs, sample_count)
    raw = np.fft.rfft(samples)
    factors = _fold_factors(sample_count)
    actual_psd = factors * np.abs(raw) ** 2 * dt / sample_count
    np.testing.assert_allclose(freqs, scipy_freqs, atol=0.0)
    np.testing.assert_allclose(actual_psd, scipy_psd, rtol=1e-13, atol=1e-14)
    assert scipy_psd[0] > 0
    assert scipy_psd[-1] > 0

    df = fs / sample_count
    np.testing.assert_allclose(np.sum(actual_psd) * df, np.mean(samples**2), atol=1e-13)
    np.testing.assert_allclose(
        actual_psd * df, np.abs(amplitudes) ** 2 / factors, rtol=1e-12, atol=1e-13
    )
    asd = np.sqrt(actual_psd)
    np.testing.assert_allclose(asd**2, scipy_psd, rtol=1e-13, atol=1e-14)


def test_t7_zero_padded_amplitude_density_conversion_is_not_the_periodogram():
    """T7: naive padded A-to-density conversion is smaller by N/M."""
    samples = np.array([0.7, -0.2, 1.1, 0.4, -0.8, 1.6, 0.3, 0.9])
    sample_count = len(samples)
    transform_length = 16
    fs = 32.0
    dt = 1.0 / fs
    t_ref = _T0 + 0.003
    freqs, amplitudes = _amplitude_spectrum(samples, dt, _T0, t_ref, transform_length)
    scipy_freqs, scipy_psd = _scipy_periodogram(samples, fs, transform_length)
    factors = _fold_factors(transform_length)
    naive_density = np.abs(amplitudes) ** 2 / (factors * (fs / transform_length))
    np.testing.assert_allclose(freqs, scipy_freqs, atol=0.0)
    np.testing.assert_allclose(
        naive_density,
        scipy_psd * (sample_count / transform_length),
        rtol=1e-12,
        atol=1e-13,
    )
    _assert_complex_not_close(naive_density, scipy_psd, min_distance=1e-2)


@pytest.mark.parametrize("sample_count", [8, 9])
def test_t8_csd_matches_scipy_phasor_conversion_and_parseval(sample_count):
    """T8: cross-density keeps SciPy's left-conjugate complex orientation."""
    fs = 72.0
    dt = 1.0 / fs
    n = np.arange(sample_count, dtype=np.float64)
    bin_index = 1
    freq = bin_index * fs / sample_count
    phase_x = 0.37
    delay = 0.007
    phase_y = phase_x - 2 * np.pi * freq * delay
    x = 1.4 + 0.8 * np.cos(2 * np.pi * bin_index * n / sample_count + phase_x)
    y = 0.9 + 1.1 * np.cos(2 * np.pi * bin_index * n / sample_count + phase_y)
    if sample_count % 2 == 0:
        x += 0.3 * (-1.0) ** n
        y += 0.6 * (-1.0) ** n
    t_ref = _T0 + 0.0029

    freqs, ax = _amplitude_spectrum(x, dt, _T0, t_ref)
    _, ay = _amplitude_spectrum(y, dt, _T0, t_ref)
    scipy_freqs, sxy = _scipy_csd(x, y, fs, sample_count)
    _, syx = _scipy_csd(y, x, fs, sample_count)
    raw_x = np.fft.rfft(x)
    raw_y = np.fft.rfft(y)
    factors = _fold_factors(sample_count)
    actual = factors * np.conjugate(raw_x) * raw_y * dt / sample_count
    np.testing.assert_allclose(freqs, scipy_freqs, atol=0.0)
    np.testing.assert_allclose(actual, sxy, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(syx, np.conjugate(sxy), rtol=1e-13, atol=1e-13)
    assert sxy[0].real > 0
    assert abs(sxy[bin_index].imag) > 0.01
    if sample_count % 2 == 0:
        assert sxy[-1].real > 0

    # Density conversion is restricted to the unpadded boxcar case.  Interior
    # bins use P=A*; endpoints use A and have no phasor quadrature contract.
    phasor_x = np.conjugate(ax)
    phasor_y = np.conjugate(ay)
    interior = np.arange(1, len(sxy))
    if sample_count % 2 == 0:
        interior = interior[:-1]
    df = fs / sample_count
    np.testing.assert_allclose(
        phasor_x[interior] * np.conjugate(phasor_y[interior]) / (2 * df),
        sxy[interior],
        rtol=1e-12,
        atol=1e-12,
    )
    endpoints = [0]
    if sample_count % 2 == 0:
        endpoints.append(len(sxy) - 1)
    np.testing.assert_allclose(
        np.conjugate(ax[endpoints]) * ay[endpoints] / df,
        sxy[endpoints],
        rtol=1e-12,
        atol=1e-12,
    )
    _assert_complex_not_close(
        np.conjugate(phasor_x[bin_index]) * phasor_y[bin_index] / (2 * df),
        sxy[bin_index],
    )

    _, pxx = _scipy_periodogram(x, fs, sample_count)
    _, sxx = _scipy_csd(x, x, fs, sample_count)
    np.testing.assert_allclose(sxx, pxx, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.real(np.sum(sxy) * df), np.mean(x * y), atol=1e-13)

    # A real delay has the negative phase for the SciPy left-conjugate order.
    np.testing.assert_allclose(
        np.angle(sxy[bin_index]), -2 * np.pi * freq * delay, atol=1e-12
    )

    # The forbidden 2*Re fold destroys the imaginary cross spectrum.
    two_sided = np.conjugate(np.fft.fft(x)) * np.fft.fft(y) * dt / sample_count
    wrong_fold = two_sided[bin_index] + two_sided[-bin_index]
    _assert_complex_not_close(wrong_fold, sxy[bin_index])


def test_t8_welch_integrates_to_segment_means_not_the_full_record_mean():
    """T8: Welch averages segment products before integrating the CSD."""
    x = np.arange(1.0, 9.0)
    y = np.array([2.0, 1.0, 4.0, 3.0, 7.0, 5.0, 8.0, 6.0])
    fs = 16.0
    nperseg = 4
    noverlap = 2
    freqs, estimated = _scipy_csd(x, y, fs, nperseg, noverlap=noverlap)
    df = freqs[1] - freqs[0]
    segments_x = [x[start : start + nperseg] for start in (0, 2, 4)]
    segments_y = [y[start : start + nperseg] for start in (0, 2, 4)]
    segment_mean_products = np.mean(
        [np.mean(xs * ys) for xs, ys in zip(segments_x, segments_y, strict=True)]
    )
    np.testing.assert_allclose(
        np.real(np.sum(estimated) * df), segment_mean_products, atol=1e-13
    )
    _assert_complex_not_close(np.real(np.sum(estimated) * df), np.mean(x * y))

    psd_freqs, estimated_psd = _scipy_welch(x, fs, nperseg, noverlap)
    segment_mean_squares = np.mean([np.mean(xs**2) for xs in segments_x])
    np.testing.assert_allclose(
        np.real(np.sum(estimated_psd) * (psd_freqs[1] - psd_freqs[0])),
        segment_mean_squares,
        atol=1e-13,
    )
    _assert_complex_not_close(
        np.sum(estimated_psd) * (psd_freqs[1] - psd_freqs[0]), np.mean(x**2)
    )


def test_t9_spatial_phasor_phase_matches_real_time_plane_wave_and_wave_api():
    """T9: P1*conj(P2) measures k dot (r1-r2) for a sampled plane wave."""
    sample_count = 512
    fs = 512.0
    dt = 1.0 / fs
    bin_index = 37
    freq = bin_index * fs / sample_count
    wavenumber = 1.3
    k_hat = np.array([0.6, 0.8, 0.0])
    positions = np.array([[0.4, -0.2, 0.1], [-0.3, 0.5, -0.1]])
    phase0 = 0.57
    t_ref = _T0 + 0.0026
    u = plane_wave_displacement(
        positions,
        [k_hat],
        [wavenumber],
        "P",
        amplitude_m=1.9,
        phase0=phase0,
    )[:, 0, 0, 0]

    recovered = []
    for wave_phasor in u:
        samples = _time_from_phasor(wave_phasor, freq, sample_count, dt, _T0, t_ref)
        _, amplitudes = _amplitude_spectrum(samples, dt, _T0, t_ref)
        recovered.append(np.conjugate(amplitudes[bin_index]))
    recovered = np.asarray(recovered)
    expected_phase = wavenumber * np.dot(k_hat, positions[0] - positions[1])
    observed_phase = np.angle(recovered[0] * np.conjugate(recovered[1]))
    np.testing.assert_allclose(
        np.exp(1j * observed_phase), np.exp(1j * expected_phase), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(recovered, u, rtol=1e-12, atol=1e-12)
    api_phase = np.angle(u[0] * np.conjugate(u[1]))
    np.testing.assert_allclose(
        np.exp(1j * api_phase), np.exp(1j * expected_phase), rtol=1e-12, atol=1e-12
    )
    _assert_complex_not_close(
        np.conjugate(recovered[0]) * recovered[1],
        recovered[0] * np.conjugate(recovered[1]),
    )
