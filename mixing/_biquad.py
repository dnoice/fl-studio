"""Biquad filter implementation for the mixing module.

Pure numpy biquad filter for use by EQ, filters, and other effects.
"""

import numpy as np


def biquad_coefficients(
    sr: int, filter_type: str, freq: float, q: float, gain_db: float
) -> tuple[float, float, float, float, float]:
    """Calculate biquad filter coefficients.

    Returns:
        Tuple of (b0, b1, b2, a1, a2) normalized coefficients
    """
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {sr}")
    if q <= 0:
        raise ValueError(f"Q factor must be positive, got {q}")
    # Clamp frequency to valid range (above 0, below Nyquist)
    nyquist = sr / 2
    freq = max(1.0, min(freq, nyquist - 1.0))

    w0 = 2 * np.pi * freq / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * q)

    if filter_type == "lowpass":
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

    elif filter_type == "highpass":
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

    elif filter_type == "bandpass":
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

    elif filter_type == "notch":
        b0 = 1.0
        b1 = -2 * cos_w0
        b2 = 1.0
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

    elif filter_type == "peaking":
        A = 10 ** (gain_db / 40)
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

    elif filter_type == "lowshelf":
        A = 10 ** (gain_db / 40)
        sq = 2 * np.sqrt(A) * alpha
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + sq)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - sq)
        a0 = (A + 1) + (A - 1) * cos_w0 + sq
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - sq

    elif filter_type == "highshelf":
        A = 10 ** (gain_db / 40)
        sq = 2 * np.sqrt(A) * alpha
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + sq)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - sq)
        a0 = (A + 1) - (A - 1) * cos_w0 + sq
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - sq

    else:
        return 1.0, 0.0, 0.0, 0.0, 0.0

    # Normalize
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def biquad_filter(
    audio: np.ndarray, sr: int, filter_type: str, freq: float, q: float, gain_db: float
) -> np.ndarray:
    """Apply a biquad filter to audio data.

    Args:
        audio: Input audio (1D or 2D)
        sr: Sample rate
        filter_type: lowpass, highpass, bandpass, notch, peaking, lowshelf, highshelf
        freq: Center/cutoff frequency
        q: Q factor
        gain_db: Gain in dB (for peaking/shelf types)
    """
    b0, b1, b2, a1, a2 = biquad_coefficients(sr, filter_type, freq, q, gain_db)

    if audio.ndim == 1:
        return _biquad_process_1d(audio, b0, b1, b2, a1, a2)
    else:
        output = np.copy(audio)
        for ch in range(audio.shape[1]):
            output[:, ch] = _biquad_process_1d(audio[:, ch], b0, b1, b2, a1, a2)
        return output


def _biquad_process_1d(data: np.ndarray, b0, b1, b2, a1, a2) -> np.ndarray:
    """Process a 1D array through a biquad filter."""
    output = np.zeros_like(data)
    x1 = x2 = y1 = y2 = 0.0

    for i in range(len(data)):
        x0 = data[i]
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0
        output[i] = y0

    return output
