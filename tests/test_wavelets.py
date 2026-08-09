"""Tests for source waveform helpers."""

import math

import pytest
import torch

from nami.wavelets import (
    chirp,
    gaussian,
    gaussian_derivative,
    klauder,
    ormsby,
    ricker,
    sine_burst,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ricker_preserves_previous_formula(dtype):
    freq, length, dt, peak_time = 20.0, 101, 0.001, 0.05
    t = torch.arange(float(length), dtype=dtype) * dt - peak_time
    expected = (1 - 2 * math.pi**2 * freq**2 * t**2) * torch.exp(
        -(math.pi**2) * freq**2 * t**2
    )

    actual = ricker(freq, length, dt, peak_time, dtype)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ricker_is_negative_second_gaussian_derivative(dtype):
    args = (25.0, 121, 0.001, 0.06)

    actual = ricker(*args, dtype=dtype)
    derivative = gaussian_derivative(*args, order=2, dtype=dtype)

    assert torch.equal(actual, -derivative)


def test_gaussian_and_derivatives_have_documented_shapes():
    freq, length, dt, peak_time = 10.0, 201, 0.001, 0.1
    centre = round(peak_time / dt)

    pulse = gaussian(freq, length, dt, peak_time, torch.float64)
    first = gaussian_derivative(
        freq, length, dt, peak_time, order=1, dtype=torch.float64
    )
    second = gaussian_derivative(
        freq, length, dt, peak_time, order=2, dtype=torch.float64
    )

    assert pulse[centre] == 1
    assert first[centre] == 0
    assert second[centre] == -1
    assert torch.allclose(pulse, pulse.flip(0))
    assert torch.allclose(first, -first.flip(0), atol=1e-14)
    assert torch.allclose(second, second.flip(0))


def test_gaussian_derivative_rejects_unsupported_order():
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        gaussian_derivative(10.0, 10, 0.001, 0.005, order=3)


def test_ormsby_is_symmetric_and_unit_at_peak():
    wavelet = ormsby(5.0, 10.0, 30.0, 40.0, 201, 0.001, 0.1, torch.float64)

    assert wavelet[100] == pytest.approx(1.0)
    assert torch.allclose(wavelet, wavelet.flip(0), atol=1e-14)


def test_ormsby_has_trapezoidal_spectrum():
    length, dt = 16385, 0.001
    wavelet = ormsby(
        5.0, 10.0, 30.0, 40.0, length, dt, (length // 2) * dt,
        torch.float64,
    )
    frequencies = torch.fft.rfftfreq(length, dt)
    spectrum = torch.fft.rfft(wavelet).abs()
    spectrum /= spectrum.max()

    def magnitude_at(freq):
        index = torch.argmin((frequencies - freq).abs())
        return spectrum[index].item()

    assert magnitude_at(5.0) < 0.02
    assert magnitude_at(10.0) > 0.95
    assert magnitude_at(30.0) > 0.95
    assert magnitude_at(40.0) < 0.02


@pytest.mark.parametrize(
    "frequencies",
    [
        (-1.0, 10.0, 20.0, 30.0),
        (10.0, 10.0, 20.0, 30.0),
        (10.0, 20.0, 15.0, 30.0),
    ],
)
def test_ormsby_rejects_invalid_corner_frequencies(frequencies):
    with pytest.raises(ValueError, match="frequencies must satisfy"):
        ormsby(*frequencies, 20, 0.001, 0.01)


def test_sine_burst_has_finite_hann_window():
    freq, cycles, dt, start_time = 10.0, 2, 0.005, 0.025
    burst = sine_burst(
        freq,
        cycles,
        60,
        dt,
        start_time=start_time,
        dtype=torch.float64,
    )
    start = round(start_time / dt)
    stop = round((start_time + cycles / freq) / dt)

    assert torch.count_nonzero(burst[: start + 1]) == 0
    assert torch.count_nonzero(burst[stop:]) == 0
    assert torch.count_nonzero(burst[start + 1 : stop]) > 0


def test_sine_burst_rectangular_window_matches_carrier():
    freq, cycles, length, dt = 10.0, 2, 60, 0.005
    burst = sine_burst(
        freq, cycles, length, dt, start_time=0.025, window=None,
        dtype=torch.float64,
    )
    t = torch.arange(length, dtype=torch.float64) * dt - 0.025
    expected = torch.where(
        (t >= 0) & (t <= cycles / freq),
        torch.sin(2 * math.pi * freq * t),
        torch.zeros_like(t),
    )
    assert torch.equal(burst, expected)


@pytest.mark.parametrize(
    ("cycles", "window", "message"),
    [(0, "hann", "cycles"), (1.5, "hann", "cycles"), (1, "hamming", "window")],
)
def test_sine_burst_rejects_invalid_options(cycles, window, message):
    with pytest.raises(ValueError, match=message):
        sine_burst(10.0, cycles, 20, 0.001, window=window)


def test_chirp_matches_linear_phase_and_support():
    start_freq, end_freq = 10.0, 40.0
    length, dt, start_time, duration = 501, 0.001, 0.1, 0.3
    actual = chirp(
        start_freq, end_freq, length, dt, start_time, duration, None,
        torch.float64,
    )
    t = torch.arange(length, dtype=torch.float64) * dt - start_time
    sweep_rate = (end_freq - start_freq) / duration
    expected = torch.sin(
        2 * math.pi * (start_freq * t + 0.5 * sweep_rate * t**2)
    )
    expected = torch.where(
        (t >= 0) & (t <= duration), expected, torch.zeros_like(expected),
    )
    assert torch.equal(actual, expected)


def test_chirp_hann_window_starts_and_ends_at_zero():
    actual = chirp(
        10.0, 40.0, 501, 0.001, start_time=0.1, duration=0.3,
        dtype=torch.float64,
    )
    assert torch.count_nonzero(actual[:101]) == 0
    assert torch.count_nonzero(actual[400:]) == 0
    assert torch.count_nonzero(actual[101:400]) > 0


def test_klauder_is_symmetric_normalized_and_band_limited():
    length, dt, peak_time = 4001, 0.001, 2.0
    wavelet = klauder(
        10.0, 40.0, length, dt, duration=1.0, peak_time=peak_time,
        dtype=torch.float64,
    )
    centre = round(peak_time / dt)
    assert wavelet[centre] == 1
    assert torch.allclose(wavelet, wavelet.flip(0), atol=5e-14)
    assert torch.count_nonzero(wavelet[: centre - 1000]) == 0
    assert torch.count_nonzero(wavelet[centre + 1001 :]) == 0

    frequencies = torch.fft.rfftfreq(length, dt)
    spectrum = torch.fft.rfft(wavelet).abs()
    spectrum /= spectrum.max()
    at_5 = spectrum[torch.argmin((frequencies - 5.0).abs())]
    at_20 = spectrum[torch.argmin((frequencies - 20.0).abs())]
    at_45 = spectrum[torch.argmin((frequencies - 45.0).abs())]
    assert at_5 < 0.03
    assert at_20 > 0.5
    assert at_45 < 0.03


@pytest.mark.parametrize(
    ("fn", "args", "message"),
    (
        (gaussian, (0.0, 10, 0.001, 0.0), "freq"),
        (ricker, (10.0, 0, 0.001, 0.0), "length"),
        (ricker, (10.0, 10, -0.001, 0.0), "dt"),
        (chirp, (10.0, 10.0, 100, 0.001), "must differ"),
        (chirp, (10.0, 20.0, 100, 0.001, 0.0, 0.0), "duration"),
        (klauder, (20.0, 10.0, 100, 0.001, 1.0, 0.5), "greater"),
    ),
)
def test_waveforms_reject_invalid_sampling_or_frequency(fn, args, message):
    with pytest.raises(ValueError, match=message):
        fn(*args)
