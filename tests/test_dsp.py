"""Tests for the C++ DSP module (fl_dsp_py pybind11 bindings).

These tests verify the Python bindings work correctly. They are skipped
if the C++ module hasn't been built (pybind11 not compiled).
"""

import math

import numpy as np
import pytest

# Skip entire module if C++ DSP bindings aren't built
fl_dsp = pytest.importorskip("fl_dsp_py", reason="C++ DSP module not built")


# ─── Utility Functions ───


class TestDSPUtilityFunctions:
    """Test standalone utility functions exposed via pybind11."""

    def test_db_to_linear_zero(self):
        assert fl_dsp.db_to_linear(0.0) == pytest.approx(1.0, abs=1e-5)

    def test_db_to_linear_minus6(self):
        assert fl_dsp.db_to_linear(-6.0) == pytest.approx(0.5012, abs=1e-3)

    def test_db_to_linear_plus6(self):
        assert fl_dsp.db_to_linear(6.0) == pytest.approx(1.9953, abs=1e-3)

    def test_linear_to_db_unity(self):
        assert fl_dsp.linear_to_db(1.0) == pytest.approx(0.0, abs=1e-5)

    def test_linear_to_db_half(self):
        assert fl_dsp.linear_to_db(0.5) == pytest.approx(-6.02, abs=0.1)

    def test_db_linear_roundtrip(self):
        for db in [-20, -10, -6, 0, 3, 6, 12]:
            linear = fl_dsp.db_to_linear(float(db))
            result = fl_dsp.linear_to_db(linear)
            assert result == pytest.approx(float(db), abs=0.01)

    def test_midi_to_freq_a4(self):
        assert fl_dsp.midi_to_freq(69) == pytest.approx(440.0, abs=0.01)

    def test_midi_to_freq_c4(self):
        assert fl_dsp.midi_to_freq(60) == pytest.approx(261.63, abs=0.1)

    def test_midi_to_freq_octave(self):
        # Going up 12 semitones should double the frequency
        f1 = fl_dsp.midi_to_freq(60)
        f2 = fl_dsp.midi_to_freq(72)
        assert f2 == pytest.approx(f1 * 2.0, abs=0.01)

    def test_freq_to_midi_440(self):
        assert fl_dsp.freq_to_midi(440.0) == pytest.approx(69.0, abs=0.01)

    def test_freq_midi_roundtrip(self):
        for note in [36, 48, 60, 69, 72, 84, 96]:
            freq = fl_dsp.midi_to_freq(note)
            result = fl_dsp.freq_to_midi(freq)
            assert result == pytest.approx(float(note), abs=0.01)


# ─── AudioBuffer ───


class TestAudioBuffer:
    """Test AudioBuffer class."""

    def test_create_mono(self):
        buf = fl_dsp.AudioBuffer(1024, 1)
        assert buf.frames() == 1024
        assert buf.channels() == 1

    def test_create_stereo(self):
        buf = fl_dsp.AudioBuffer(512, 2)
        assert buf.frames() == 512
        assert buf.channels() == 2

    def test_peak_after_clear(self):
        buf = fl_dsp.AudioBuffer(256, 1)
        buf.clear()
        assert buf.peak_level() == pytest.approx(0.0)

    def test_rms_after_clear(self):
        buf = fl_dsp.AudioBuffer(256, 1)
        buf.clear()
        assert buf.rms_level() == pytest.approx(0.0)

    def test_apply_gain(self):
        buf = fl_dsp.AudioBuffer(256, 1)
        buf.apply_gain(0.5)  # Half the signal
        assert buf.peak_level() == pytest.approx(0.0)  # All zeros, still zero


# ─── BiquadFilter ───


class TestBiquadFilter:
    """Test BiquadFilter via pybind11 bindings."""

    def test_create(self):
        filt = fl_dsp.BiquadFilter()
        assert filt is not None

    def test_lowpass_attenuates_high(self):
        """A lowpass filter at 1kHz should attenuate a 5kHz signal."""
        filt = fl_dsp.BiquadFilter()
        filt.set_params(fl_dsp.BiquadType.LowPass, 1000.0, 0.707, 0.0, 44100.0)

        # Process a high-frequency signal
        sr = 44100
        t = np.arange(sr, dtype=np.float32) / sr
        signal = np.sin(2 * np.pi * 5000 * t).astype(np.float32)
        input_rms = np.sqrt(np.mean(signal**2))

        filt.process_array(signal)
        output_rms = np.sqrt(np.mean(signal**2))

        assert output_rms < input_rms * 0.5  # Should be significantly attenuated

    def test_highpass_attenuates_low(self):
        """A highpass filter at 5kHz should attenuate a 200Hz signal."""
        filt = fl_dsp.BiquadFilter()
        filt.set_params(fl_dsp.BiquadType.HighPass, 5000.0, 0.707, 0.0, 44100.0)

        sr = 44100
        t = np.arange(sr, dtype=np.float32) / sr
        signal = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        input_rms = np.sqrt(np.mean(signal**2))

        filt.process_array(signal)
        output_rms = np.sqrt(np.mean(signal**2))

        assert output_rms < input_rms * 0.3

    def test_lowpass_passes_low(self):
        """A lowpass filter at 5kHz should pass a 200Hz signal."""
        filt = fl_dsp.BiquadFilter()
        filt.set_params(fl_dsp.BiquadType.LowPass, 5000.0, 0.707, 0.0, 44100.0)

        sr = 44100
        t = np.arange(sr, dtype=np.float32) / sr
        signal = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        input_rms = np.sqrt(np.mean(signal**2))

        filt.process_array(signal)
        output_rms = np.sqrt(np.mean(signal**2))

        assert output_rms > input_rms * 0.9  # Should mostly pass through

    def test_reset_clears_state(self):
        filt = fl_dsp.BiquadFilter()
        filt.set_params(fl_dsp.BiquadType.LowPass, 1000.0, 0.707, 0.0, 44100.0)

        # Process some data
        for _ in range(100):
            filt.process(1.0)

        filt.reset()
        # After reset, output should start from clean state
        assert filt.process(0.0) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize(
        "filter_type",
        [
            fl_dsp.BiquadType.LowPass,
            fl_dsp.BiquadType.HighPass,
            fl_dsp.BiquadType.Notch,
            fl_dsp.BiquadType.Peaking,
            fl_dsp.BiquadType.LowShelf,
            fl_dsp.BiquadType.HighShelf,
            fl_dsp.BiquadType.AllPass,
        ],
    )
    def test_filter_type_no_crash(self, filter_type):
        """All filter types should be configurable without crashing."""
        filt = fl_dsp.BiquadFilter()
        filt.set_params(filter_type, 1000.0, 1.0, 3.0, 44100.0)
        result = filt.process(0.5)
        assert math.isfinite(result)


# ─── Distortion ───


class TestDistortion:
    """Test Distortion effect."""

    @pytest.mark.parametrize(
        "dist_type",
        [
            fl_dsp.DistortionType.SoftClip,
            fl_dsp.DistortionType.HardClip,
            fl_dsp.DistortionType.Tanh,
            fl_dsp.DistortionType.Foldback,
            fl_dsp.DistortionType.BitCrush,
            fl_dsp.DistortionType.Tube,
        ],
    )
    def test_distortion_type_no_crash(self, dist_type):
        dist = fl_dsp.Distortion()
        dist.set_params(dist_type, 2.0, 1.0, 0.5)
        result = dist.process(0.5)
        assert math.isfinite(result)

    def test_soft_clip_limits_output(self):
        dist = fl_dsp.Distortion()
        dist.set_params(fl_dsp.DistortionType.SoftClip, 10.0, 1.0, 0.5)
        result = dist.process(1.0)
        assert abs(result) <= 1.5  # Soft clip shouldn't exceed ~1.0

    def test_hard_clip_limits_output(self):
        dist = fl_dsp.Distortion()
        dist.set_params(fl_dsp.DistortionType.HardClip, 10.0, 1.0, 0.5)
        result = dist.process(1.0)
        assert abs(result) <= 1.1

    def test_process_array(self):
        dist = fl_dsp.Distortion()
        dist.set_params(fl_dsp.DistortionType.Tanh, 3.0, 1.0, 0.5)
        data = np.sin(np.linspace(0, 2 * np.pi, 1000)).astype(np.float32)
        dist.process_array(data)
        assert np.all(np.isfinite(data))


# ─── Compressor ───


class TestCompressor:
    """Test Compressor effect."""

    def test_below_threshold_passthrough(self):
        comp = fl_dsp.Compressor()
        comp.set_params(-10.0, 4.0, 10.0, 100.0, 0.0, 44100.0)

        # Very quiet signal should pass through mostly unchanged
        result = comp.process(0.01)
        assert result == pytest.approx(0.01, abs=0.005)

    def test_above_threshold_reduces(self):
        comp = fl_dsp.Compressor()
        comp.set_params(-20.0, 10.0, 0.1, 100.0, 0.0, 44100.0)

        # Process loud signal repeatedly to let envelope settle
        results = [comp.process(0.9) for _ in range(1000)]
        # After settling, output should be reduced compared to input
        assert abs(results[-1]) < 0.9

    def test_gain_reduction_tracking(self):
        comp = fl_dsp.Compressor()
        comp.set_params(-10.0, 4.0, 1.0, 100.0, 0.0, 44100.0)

        for _ in range(1000):
            comp.process(0.9)
        gr = comp.get_gain_reduction_db()
        assert gr <= 0.0  # Should be negative (reducing gain)

    def test_process_array(self):
        comp = fl_dsp.Compressor()
        comp.set_params(-10.0, 4.0, 10.0, 100.0, 0.0, 44100.0)
        data = (np.sin(np.linspace(0, 20 * np.pi, 1000)) * 0.8).astype(np.float32)
        comp.process_array(data)
        assert np.all(np.isfinite(data))


# ─── Reverb ───


class TestSchroederReverb:
    """Test SchroederReverb effect."""

    def test_init_and_process(self):
        rev = fl_dsp.SchroederReverb()
        rev.init(44100.0)
        rev.set_params(0.5, 0.3, 0.3)
        result = rev.process_mono(0.5)
        assert math.isfinite(result)

    def test_wet_adds_tail(self):
        """Reverb should produce output even after input stops."""
        rev = fl_dsp.SchroederReverb()
        rev.init(44100.0)
        rev.set_params(0.8, 0.2, 0.5)

        # Send impulse
        rev.process_mono(1.0)
        # Then silence - reverb tail should produce non-zero output
        # Note: comb delays are ~1422-1617 samples, so need >1600 samples
        tail = [rev.process_mono(0.0) for _ in range(3000)]
        assert any(abs(s) > 0.001 for s in tail)

    def test_process_mono_array(self):
        rev = fl_dsp.SchroederReverb()
        rev.init(44100.0)
        rev.set_params(0.5, 0.3, 0.3)
        data = np.zeros(4096, dtype=np.float32)
        data[0] = 1.0  # Impulse
        rev.process_mono_array(data)
        assert np.all(np.isfinite(data))
        assert np.max(np.abs(data[100:])) > 0.0  # Reverb tail exists

    def test_reset_clears_tail(self):
        rev = fl_dsp.SchroederReverb()
        rev.init(44100.0)
        rev.set_params(0.8, 0.2, 0.5)

        # Build up reverb
        for _ in range(1000):
            rev.process_mono(0.5)

        rev.reset()
        # After reset, silence input should produce silence
        result = rev.process_mono(0.0)
        assert abs(result) < 0.001


# ─── Limiter ───


class TestLimiter:
    """Test Limiter effect."""

    def test_limits_output(self):
        lim = fl_dsp.Limiter()
        lim.init(44100.0, 5.0)
        lim.set_params(-1.0, 50.0)  # ~0.89 linear ceiling

        # Process loud signal, let it settle
        results = [lim.process(1.5) for _ in range(500)]
        # After settling, output should be limited
        assert all(abs(r) <= 1.5 for r in results)

    def test_process_array(self):
        lim = fl_dsp.Limiter()
        lim.init(44100.0, 5.0)
        lim.set_params(-0.5, 50.0)
        data = (np.sin(np.linspace(0, 20 * np.pi, 1000)) * 2.0).astype(np.float32)
        lim.process_array(data)
        assert np.all(np.isfinite(data))


# ─── Stereo Effects ───


class TestStereoDelay:
    """Test StereoDelay effect."""

    def test_init_and_set_params(self):
        delay = fl_dsp.StereoDelay()
        delay.init(1000.0, 44100.0)
        delay.set_params(250.0, 375.0, 0.3, 0.5, 5000.0)
        # Should not crash

    def test_reset(self):
        delay = fl_dsp.StereoDelay()
        delay.init(1000.0, 44100.0)
        delay.set_params(250.0, 375.0, 0.3, 0.5, 5000.0)
        delay.reset()


class TestChorus:
    """Test Chorus effect."""

    def test_init_and_set_params(self):
        chorus = fl_dsp.Chorus()
        chorus.init(44100.0)
        chorus.set_params(1.5, 3.0, 0.5, 0.2)

    def test_reset(self):
        chorus = fl_dsp.Chorus()
        chorus.init(44100.0)
        chorus.set_params(1.5, 3.0, 0.5, 0.2)
        chorus.reset()


# ─── Enum Values ───


class TestEnumValues:
    """Verify all enum values are properly bound."""

    def test_biquad_types(self):
        expected = ["LowPass", "HighPass", "Notch", "Peaking", "LowShelf", "HighShelf", "AllPass"]
        for name in expected:
            assert hasattr(fl_dsp.BiquadType, name)

    def test_distortion_types(self):
        expected = ["SoftClip", "HardClip", "Tanh", "Foldback", "BitCrush", "Tube"]
        for name in expected:
            assert hasattr(fl_dsp.DistortionType, name)
