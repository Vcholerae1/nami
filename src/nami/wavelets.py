"""Common source waveforms for acoustic, elastic, and EM simulations."""

import math

import torch


def _validate_time_sampling(length: int, dt: float) -> None:
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        raise ValueError("length must be a positive integer.")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite.")


def _validate_frequency(freq: float, name: str = "freq") -> None:
    if not math.isfinite(freq) or freq <= 0:
        raise ValueError(f"{name} must be positive and finite.")


def _time_axis(
    length: int,
    dt: float,
    reference_time: float,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    _validate_time_sampling(length, dt)
    if not math.isfinite(reference_time):
        raise ValueError("reference time must be finite.")
    return torch.arange(float(length), dtype=dtype) * dt - reference_time


def _gaussian_family(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    order: int,
    dtype: torch.dtype | None,
    *,
    invert: bool = False,
) -> torch.Tensor:
    """Return a unit-amplitude Gaussian or normalized derivative shape."""
    _validate_frequency(freq)
    t = _time_axis(length, dt, peak_time, dtype)
    exponent = -(math.pi**2) * freq**2 * t**2
    envelope = torch.exp(exponent)

    if order == 0:
        shape = torch.ones_like(t)
    elif order == 1:
        # The continuous-time extrema are +/-1 after this normalization.
        shape = -math.sqrt(2 * math.e) * math.pi * freq * t
    elif order == 2:
        # Normalized true second derivative: its value at t=peak_time is -1.
        shape = 2 * math.pi**2 * freq**2 * t**2 - 1
    else:
        raise ValueError("order must be 1 or 2.")

    if invert:
        shape = -shape
    return shape * envelope


def gaussian(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a Gaussian pulse with the specified width parameter.

    The pulse is ``exp(-(pi * freq * (t - peak_time))**2)``. ``freq`` is an
    inverse-width parameter rather than the nonzero centre frequency of an
    oscillatory signal.

    Args:
        freq: The inverse-width frequency parameter in Hz.
        length: The number of time samples.
        dt: The time sample spacing in seconds.
        peak_time: The time of the peak amplitude in seconds.
        dtype: The PyTorch datatype to use. Defaults to PyTorch's default
            floating-point dtype.

    Returns:
        A one-dimensional PyTorch tensor containing the pulse.
    """
    return _gaussian_family(freq, length, dt, peak_time, 0, dtype)


def gaussian_derivative(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    order: int = 1,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a normalized first or second derivative of a Gaussian pulse.

    The result is normalized to unit continuous-time peak magnitude. With the
    same arguments, the second derivative has the opposite sign to
    :func:`ricker`.

    Args:
        freq: The inverse-width frequency parameter in Hz.
        length: The number of time samples.
        dt: The time sample spacing in seconds.
        peak_time: The centre time of the Gaussian in seconds.
        order: Derivative order, either ``1`` or ``2``.
        dtype: The PyTorch datatype to use. Defaults to PyTorch's default
            floating-point dtype.

    Returns:
        A one-dimensional PyTorch tensor containing the normalized derivative.
    """
    return _gaussian_family(freq, length, dt, peak_time, order, dtype)


def ricker(
    freq: float,
    length: int,
    dt: float,
    peak_time: float,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a Ricker wavelet with the specified central frequency.

    Args:
        freq: The central frequency.
        length: The number of time samples.
        dt: The time sample spacing.
        peak_time: The time (in secs) of the peak amplitude.
        dtype: The PyTorch datatype to use. Optional, defaults to PyTorch's
            default (float32).

    Returns:
        A PyTorch tensor representing the Ricker wavelet.
    """
    return _gaussian_family(
        freq,
        length,
        dt,
        peak_time,
        2,
        dtype,
        invert=True,
    )


def ormsby(
    f1: float,
    f2: float,
    f3: float,
    f4: float,
    length: int,
    dt: float,
    peak_time: float,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a zero-phase Ormsby wavelet with a trapezoidal spectrum.

    ``f1`` and ``f4`` are the lower and upper stop-band corners; ``f2`` and
    ``f3`` bound the pass band. The wavelet is scaled to one at ``peak_time``.

    Args:
        f1: Lower stop-band corner frequency in Hz.
        f2: Lower pass-band corner frequency in Hz.
        f3: Upper pass-band corner frequency in Hz.
        f4: Upper stop-band corner frequency in Hz.
        length: The number of time samples.
        dt: The time sample spacing in seconds.
        peak_time: The time of the central peak in seconds.
        dtype: The PyTorch datatype to use. Defaults to PyTorch's default
            floating-point dtype.

    Returns:
        A one-dimensional PyTorch tensor containing the wavelet.
    """
    if not all(math.isfinite(freq) for freq in (f1, f2, f3, f4)) or not (
        0 <= f1 < f2 < f3 < f4
    ):
        raise ValueError("frequencies must satisfy 0 <= f1 < f2 < f3 < f4.")

    t = _time_axis(length, dt, peak_time, dtype)

    def corner_term(high: float, low: float) -> torch.Tensor:
        denominator = math.pi * (high - low)
        high_term = (math.pi * high) ** 2 * torch.sinc(high * t) ** 2
        low_term = (math.pi * low) ** 2 * torch.sinc(low * t) ** 2
        return (high_term - low_term) / denominator

    wavelet = corner_term(f4, f3) - corner_term(f2, f1)
    peak_amplitude = math.pi * (f4 + f3 - f2 - f1)
    return wavelet / peak_amplitude


def sine_burst(
    freq: float,
    cycles: int,
    length: int,
    dt: float,
    start_time: float = 0.0,
    window: str | None = "hann",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a finite-duration sinusoidal tone burst.

    Args:
        freq: Carrier frequency in Hz.
        cycles: Positive integer number of carrier cycles.
        length: The number of time samples.
        dt: The time sample spacing in seconds.
        start_time: The start time of the burst in seconds.
        window: ``"hann"`` for a tapered burst or ``None`` for a rectangular
            burst.
        dtype: The PyTorch datatype to use. Defaults to PyTorch's default
            floating-point dtype.

    Returns:
        A one-dimensional PyTorch tensor containing the burst.
    """
    _validate_frequency(freq)
    if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles <= 0:
        raise ValueError("cycles must be a positive integer.")
    if window not in {"hann", None}:
        raise ValueError("window must be 'hann' or None.")

    t = _time_axis(length, dt, start_time, dtype)
    duration = cycles / freq
    inside = (t >= 0) & (t <= duration)
    carrier = torch.sin(2 * math.pi * freq * t)
    if window == "hann":
        envelope = 0.5 - 0.5 * torch.cos(2 * math.pi * t / duration)
        carrier = carrier * envelope
    return torch.where(inside, carrier, torch.zeros_like(carrier))


def chirp(
    start_freq: float,
    end_freq: float,
    length: int,
    dt: float,
    start_time: float = 0.0,
    duration: float | None = None,
    window: str | None = "hann",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a finite linear-frequency sweep.

    Args:
        start_freq: Instantaneous frequency at the start of the sweep in Hz.
        end_freq: Instantaneous frequency at the end of the sweep in Hz.
        length: The number of time samples.
        dt: The time sample spacing in seconds.
        start_time: The start time of the sweep in seconds.
        duration: Sweep duration in seconds. Defaults to the part of the
            sampled record at or after ``start_time``.
        window: ``"hann"`` for a tapered sweep or ``None`` for a rectangular
            sweep.
        dtype: The PyTorch datatype to use. Defaults to PyTorch's default
            floating-point dtype.

    Returns:
        A one-dimensional PyTorch tensor containing the sweep.
    """
    _validate_frequency(start_freq, "start_freq")
    _validate_frequency(end_freq, "end_freq")
    if start_freq == end_freq:
        raise ValueError("start_freq and end_freq must differ.")
    if window not in {"hann", None}:
        raise ValueError("window must be 'hann' or None.")

    t = _time_axis(length, dt, start_time, dtype)
    if duration is None:
        duration = (length - 1) * dt - start_time
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite.")

    sweep_rate = (end_freq - start_freq) / duration
    phase = 2 * math.pi * (start_freq * t + 0.5 * sweep_rate * t**2)
    signal = torch.sin(phase)
    if window == "hann":
        signal = signal * (0.5 - 0.5 * torch.cos(2 * math.pi * t / duration))
    inside = (t >= 0) & (t <= duration)
    return torch.where(inside, signal, torch.zeros_like(signal))


def klauder(
    start_freq: float,
    end_freq: float,
    length: int,
    dt: float,
    duration: float,
    peak_time: float,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return a normalized zero-phase Klauder wavelet.

    The wavelet is the analytic autocorrelation of a real linear sweep from
    ``start_freq`` to ``end_freq`` over ``duration`` seconds. Its support is
    ``peak_time +/- duration`` and its value at ``peak_time`` is one.

    Args:
        start_freq: Lower sweep frequency in Hz.
        end_freq: Upper sweep frequency in Hz.
        length: The number of time samples.
        dt: The time sample spacing in seconds.
        duration: Duration of the underlying linear sweep in seconds.
        peak_time: Centre time of the zero-phase wavelet in seconds.
        dtype: The PyTorch datatype to use. Defaults to PyTorch's default
            floating-point dtype.

    Returns:
        A one-dimensional PyTorch tensor containing the wavelet.
    """
    _validate_frequency(start_freq, "start_freq")
    _validate_frequency(end_freq, "end_freq")
    if end_freq <= start_freq:
        raise ValueError("end_freq must be greater than start_freq.")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive and finite.")

    t = _time_axis(length, dt, peak_time, dtype)
    abs_t = t.abs()
    overlap = duration - abs_t
    sweep_rate = (end_freq - start_freq) / duration
    envelope = overlap / duration * torch.sinc(sweep_rate * t * overlap)
    carrier = torch.cos(math.pi * (start_freq + end_freq) * t)
    wavelet = envelope * carrier
    return torch.where(abs_t <= duration, wavelet, torch.zeros_like(wavelet))


__all__ = [
    "chirp",
    "gaussian",
    "gaussian_derivative",
    "klauder",
    "ormsby",
    "ricker",
    "sine_burst",
]
