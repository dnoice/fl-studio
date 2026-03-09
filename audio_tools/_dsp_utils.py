"""Shared DSP utilities for audio_tools.

Common functions used across multiple audio processing modules:
resampling, window functions, and signal helpers.
"""

import numpy as np


def resample(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using scipy polyphase resampling (high quality) with
    numpy linear interpolation fallback.

    Args:
        audio: Input audio (1D mono or 2D multi-channel)
        src_sr: Source sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if src_sr == target_sr:
        return audio

    ratio = target_sr / src_sr
    new_length = int(audio.shape[0] * ratio) if audio.ndim > 1 else int(len(audio) * ratio)

    # Use scipy.signal.resample_poly for higher quality when available
    try:
        from math import gcd

        from scipy.signal import resample_poly

        up = target_sr // gcd(src_sr, target_sr)
        down = src_sr // gcd(src_sr, target_sr)
        result = resample_poly(audio, up, down, axis=0).astype(np.float32)
        return result
    except ImportError:
        pass

    # Fallback: numpy linear interpolation
    if audio.ndim == 1:
        x_old = np.arange(len(audio))
        x_new = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(x_new, x_old, audio).astype(np.float32)
    else:
        result = np.zeros((new_length, audio.shape[1]), dtype=np.float32)
        x_old = np.arange(audio.shape[0])
        x_new = np.linspace(0, audio.shape[0] - 1, new_length)
        for ch in range(audio.shape[1]):
            result[:, ch] = np.interp(x_new, x_old, audio[:, ch])
        return result


def midi_to_freq(midi_note: int, a4: float = 440.0) -> float:
    """Convert MIDI note number to frequency in Hz.

    Args:
        midi_note: MIDI note number (0-127)
        a4: Reference frequency for A4 (MIDI 69)

    Returns:
        Frequency in Hz
    """
    return a4 * 2.0 ** ((midi_note - 69) / 12.0)


def freq_to_midi(freq: float, a4: float = 440.0) -> float:
    """Convert frequency in Hz to MIDI note number.

    Args:
        freq: Frequency in Hz (must be positive)
        a4: Reference frequency for A4 (MIDI 69)

    Returns:
        MIDI note number (fractional)
    """
    if freq <= 0:
        raise ValueError(f"Frequency must be positive, got {freq}")
    return 69 + 12 * np.log2(freq / a4)


def make_window(size: int, window_type: str = "hanning") -> np.ndarray:
    """Create a window function array.

    Args:
        size: Window size in samples
        window_type: Window type (hanning, hamming, blackman, rectangular)

    Returns:
        Window array
    """
    if window_type == "hamming":
        return np.hamming(size)
    elif window_type == "blackman":
        return np.blackman(size)
    elif window_type == "rectangular":
        return np.ones(size)
    else:
        return np.hanning(size)
