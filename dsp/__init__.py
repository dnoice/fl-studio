"""FL Studio Toolkit - DSP Engine

High-performance audio processing via C++ (pybind11) with Python fallbacks.

The C++ module provides ~10-50x speedup for filter processing, distortion,
compression, and reverb. If the C++ module isn't built, pure Python
implementations from the mixing module are used instead.

Quick start::

    from dsp import biquad_filter, has_native_dsp

    if has_native_dsp():
        print("Using C++ DSP engine")
    else:
        print("Using Python fallback")
"""

import numpy as np

# Try importing the native C++ module
try:
    import fl_dsp_py as _native

    _HAS_NATIVE = True
except ImportError:
    _native = None
    _HAS_NATIVE = False


def has_native_dsp() -> bool:
    """Check if the native C++ DSP module is available."""
    return _HAS_NATIVE


# ─── Utility Functions (always available) ───


def db_to_linear(db: float) -> float:
    """Convert dB to linear gain."""
    if _HAS_NATIVE:
        return _native.db_to_linear(db)
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear gain to dB."""
    if _HAS_NATIVE:
        return _native.linear_to_db(linear)
    return 20.0 * np.log10(max(linear, 1e-10))


def midi_to_freq(note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    if _HAS_NATIVE:
        return _native.midi_to_freq(note)
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def freq_to_midi(freq: float) -> float:
    """Convert frequency in Hz to MIDI note number."""
    if _HAS_NATIVE:
        return _native.freq_to_midi(freq)
    return 69.0 + 12.0 * np.log2(freq / 440.0)


# ─── Biquad Filter (C++ or Python fallback) ───


def biquad_filter(
    audio: np.ndarray,
    sr: int,
    filter_type: str,
    freq: float,
    q: float = 0.707,
    gain_db: float = 0.0,
) -> np.ndarray:
    """Apply a biquad filter to audio data.

    Uses the C++ DSP engine if available, otherwise falls back to the
    Python implementation in mixing._biquad.

    Args:
        audio: Audio array (mono float32)
        sr: Sample rate
        filter_type: One of 'lowpass', 'highpass', 'bandpass', 'notch',
                     'peaking', 'lowshelf', 'highshelf', 'allpass'
        freq: Filter frequency in Hz
        q: Q factor (default 0.707 = Butterworth)
        gain_db: Gain in dB (for peaking/shelf filters)

    Returns:
        Filtered audio array
    """
    if _HAS_NATIVE:
        type_map = {
            "lowpass": _native.BiquadType.LowPass,
            "highpass": _native.BiquadType.HighPass,
            "notch": _native.BiquadType.Notch,
            "peaking": _native.BiquadType.Peaking,
            "lowshelf": _native.BiquadType.LowShelf,
            "highshelf": _native.BiquadType.HighShelf,
            "allpass": _native.BiquadType.AllPass,
        }
        native_type = type_map.get(filter_type.lower())
        if native_type is not None:
            filt = _native.BiquadFilter()
            filt.set_params(native_type, freq, q, gain_db, float(sr))
            result = audio.astype(np.float32).copy()
            filt.process_array(result)
            return result

    # Python fallback
    from mixing._biquad import biquad_filter as _py_biquad

    return _py_biquad(audio, sr, filter_type, freq, q, gain_db)


# ─── Compressor (C++ or Python fallback) ───


def compress(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -10.0,
    ratio: float = 4.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    makeup_db: float = 0.0,
) -> np.ndarray:
    """Apply dynamic range compression.

    Uses C++ DSP engine if available.

    Args:
        audio: Mono audio array (float32)
        sr: Sample rate
        threshold_db: Compression threshold in dB
        ratio: Compression ratio (e.g. 4.0 = 4:1)
        attack_ms: Attack time in milliseconds
        release_ms: Release time in milliseconds
        makeup_db: Makeup gain in dB

    Returns:
        Compressed audio array
    """
    if _HAS_NATIVE:
        comp = _native.Compressor()
        comp.set_params(threshold_db, ratio, attack_ms, release_ms, makeup_db, float(sr))
        result = audio.astype(np.float32).copy()
        comp.process_array(result)
        return result

    # Python fallback - simple compressor
    result = audio.copy()
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    makeup_lin = 10.0 ** (makeup_db / 20.0)
    for i in range(len(result)):
        level = abs(result[i])
        if level > threshold_lin:
            gain = threshold_lin + (level - threshold_lin) / ratio
            result[i] = result[i] * (gain / level) * makeup_lin
        else:
            result[i] = result[i] * makeup_lin
    return result
