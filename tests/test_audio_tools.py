"""Comprehensive tests for the audio tools module."""

import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio_tools.batch_processor import BatchProcessor
from audio_tools.bpm_detector import BPMDetector, BPMResult
from audio_tools.format_converter import FormatConverter
from audio_tools.key_detector import KeyDetector, KeyResult
from audio_tools.sample_slicer import SampleSlicer, SliceResult
from audio_tools.spectrum_analyzer import BandAnalysis, SpectrumAnalyzer, SpectrumData


@pytest.fixture
def sample_wav(tmp_path):
    """Generate a test WAV file (1 second, 440 Hz sine)."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = str(tmp_path / "test_440hz.wav")
    sf.write(path, audio, sr)
    return path, audio, sr


@pytest.fixture
def stereo_wav(tmp_path):
    """Generate a stereo test WAV file."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    left = (0.7 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (0.5 * np.sin(2 * np.pi * 554 * t)).astype(np.float32)
    audio = np.column_stack([left, right])
    path = str(tmp_path / "test_stereo.wav")
    sf.write(path, audio, sr)
    return path, audio, sr


@pytest.fixture
def rhythmic_wav(tmp_path):
    """Generate a rhythmic click track at 120 BPM for BPM detection."""
    sr = 44100
    duration = 4.0  # 4 seconds = 8 beats at 120 BPM
    bpm = 120.0
    beat_interval = 60.0 / bpm
    audio = np.zeros(int(sr * duration), dtype=np.float32)

    # Place clicks at beat positions
    for beat in range(int(duration / beat_interval)):
        start = int(beat * beat_interval * sr)
        click_len = min(200, len(audio) - start)
        if click_len > 0:
            click = np.sin(2 * np.pi * 1000 * np.arange(click_len) / sr)
            click *= np.exp(-np.arange(click_len) / 50)  # Exponential decay
            audio[start : start + click_len] += click.astype(np.float32) * 0.9

    path = str(tmp_path / "test_120bpm.wav")
    sf.write(path, audio, sr)
    return path, audio, sr


@pytest.fixture
def silent_wav(tmp_path):
    """Generate a silent WAV file."""
    sr = 44100
    audio = np.zeros(sr, dtype=np.float32)
    path = str(tmp_path / "test_silent.wav")
    sf.write(path, audio, sr)
    return path, audio, sr


# ─── BPM Detector Tests ───


class TestBPMDetector:
    def test_detect_rhythmic(self, rhythmic_wav):
        path, audio, sr = rhythmic_wav
        result = BPMDetector.detect(path, min_bpm=80, max_bpm=160)
        assert isinstance(result, BPMResult)
        assert result.bpm > 0
        # Should be in the ballpark of 120 BPM (allow half/double)
        assert 60 <= result.bpm <= 240

    def test_detect_autocorrelation(self, rhythmic_wav):
        path, _, _ = rhythmic_wav
        result = BPMDetector.detect(path, method="autocorrelation")
        assert result.method == "autocorrelation"
        assert result.bpm > 0

    def test_detect_onset_intervals(self, rhythmic_wav):
        path, _, _ = rhythmic_wav
        result = BPMDetector.detect(path, method="onset_intervals")
        assert result.method == "onset_intervals"

    def test_detect_both_methods(self, rhythmic_wav):
        path, _, _ = rhythmic_wav
        result = BPMDetector.detect(path, method="both")
        assert result.bpm > 0

    def test_detect_silent(self, silent_wav):
        path, _, _ = silent_wav
        result = BPMDetector.detect(path)
        # Should return without crashing
        assert isinstance(result, BPMResult)

    def test_detect_batch(self, rhythmic_wav, sample_wav):
        path1, _, _ = rhythmic_wav
        path2, _, _ = sample_wav
        results = BPMDetector.detect_batch([path1, path2])
        assert len(results) == 2

    def test_bpm_result_repr(self):
        result = BPMResult(120.0, 0.85, [(120.0, 0.85)], "autocorrelation")
        assert "120.0" in repr(result)

    def test_detect_nonexistent_file(self):
        with pytest.raises(IOError):
            BPMDetector.detect("/nonexistent/file.wav")


# ─── Key Detector Tests ───


class TestKeyDetector:
    def test_detect_key(self, sample_wav):
        path, _, _ = sample_wav
        result = KeyDetector.detect(path)
        assert isinstance(result, KeyResult)
        assert result.key
        assert 0 <= result.root <= 11
        assert result.mode in ("major", "minor")

    def test_detect_from_array(self):
        sr = 44100
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        # C major chord: C, E, G
        audio = (
            np.sin(2 * np.pi * 261.63 * t)
            + np.sin(2 * np.pi * 329.63 * t)
            + np.sin(2 * np.pi * 392.00 * t)
        ).astype(np.float32) / 3
        result = KeyDetector.detect_from_array(audio, sr)
        assert isinstance(result, KeyResult)
        assert result.confidence >= 0

    def test_camelot_notation(self):
        result = KeyResult("C Major", 0, "major", 0.9, [], [])
        assert result.camelot == "8B"

    def test_open_key_notation(self):
        result = KeyResult("C Minor", 0, "minor", 0.9, [], [])
        assert result.open_key == "1m"

    def test_detect_duration_limit(self, sample_wav):
        path, _, _ = sample_wav
        result = KeyDetector.detect(path, duration=0.5)
        assert isinstance(result, KeyResult)

    def test_chromagram_length(self, sample_wav):
        path, _, _ = sample_wav
        result = KeyDetector.detect(path)
        assert len(result.chromagram) == 12

    def test_candidates_populated(self, sample_wav):
        path, _, _ = sample_wav
        result = KeyDetector.detect(path)
        assert len(result.candidates) > 0


# ─── Sample Slicer Tests ───


class TestSampleSlicer:
    def test_detect_transients(self, rhythmic_wav):
        _, audio, sr = rhythmic_wav
        positions = SampleSlicer.detect_transients(audio, sr, sensitivity=0.5)
        assert len(positions) > 0

    def test_slice_file(self, rhythmic_wav, tmp_path):
        path, _, _ = rhythmic_wav
        output_dir = str(tmp_path / "slices")
        result = SampleSlicer.slice(path, output_dir=output_dir, sensitivity=0.7)
        assert isinstance(result, SliceResult)
        assert len(result.slices) > 0
        assert result.sample_rate == 44100

    def test_slice_no_output(self, rhythmic_wav):
        """Slice without saving files."""
        path, _, _ = rhythmic_wav
        result = SampleSlicer.slice(path, output_dir=None)
        assert len(result.slices) > 0
        for s in result.slices:
            assert s.filepath is None

    def test_slice_uniform(self, sample_wav, tmp_path):
        path, _, _ = sample_wav
        result = SampleSlicer.slice_uniform(path, slice_duration_ms=250)
        assert len(result.slices) >= 3  # 1 second / 250ms = 4 slices

    def test_slice_result_summary(self, rhythmic_wav):
        path, _, _ = rhythmic_wav
        result = SampleSlicer.slice(path)
        summary = result.summary()
        assert "Source:" in summary
        assert "Slices Found:" in summary


# ─── Batch Processor Tests ───


class TestBatchProcessor:
    def test_normalize(self, sample_wav, tmp_path):
        path, _, _ = sample_wav
        out = str(tmp_path / "normalized.wav")
        proc = BatchProcessor().normalize(target_db=-1.0)
        result = proc.process_file(path, out)
        assert result.success
        assert result.peak_after > 0

    def test_trim_silence(self, tmp_path):
        sr = 44100
        # Audio with silence, then signal, then silence
        silence = np.zeros(sr // 2, dtype=np.float32)
        signal = (0.5 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
        audio = np.concatenate([silence, signal, silence])
        path = str(tmp_path / "with_silence.wav")
        sf.write(path, audio, sr)

        out = str(tmp_path / "trimmed.wav")
        proc = BatchProcessor().trim_silence(threshold_db=-40.0)
        result = proc.process_file(path, out)
        assert result.success
        # Trimmed file should be shorter
        trimmed, _ = sf.read(out)
        assert len(trimmed) < len(audio)

    def test_fade_in_out(self, sample_wav, tmp_path):
        path, _, _ = sample_wav
        out = str(tmp_path / "faded.wav")
        proc = BatchProcessor().fade_in(duration_ms=50).fade_out(duration_ms=50)
        result = proc.process_file(path, out)
        assert result.success

    def test_to_mono(self, stereo_wav, tmp_path):
        path, _, _ = stereo_wav
        out = str(tmp_path / "mono.wav")
        proc = BatchProcessor().to_mono()
        result = proc.process_file(path, out)
        assert result.success
        mono, _ = sf.read(out)
        assert mono.ndim == 1

    def test_to_stereo(self, sample_wav, tmp_path):
        path, _, _ = sample_wav
        out = str(tmp_path / "stereo.wav")
        proc = BatchProcessor().to_stereo()
        result = proc.process_file(path, out)
        assert result.success
        stereo, _ = sf.read(out)
        assert stereo.ndim == 2

    def test_reverse(self, sample_wav, tmp_path):
        path, original, _ = sample_wav
        out = str(tmp_path / "reversed.wav")
        proc = BatchProcessor().reverse()
        result = proc.process_file(path, out)
        assert result.success
        reversed_audio, _ = sf.read(out)
        # First sample of reversed should match last of original
        assert abs(reversed_audio[0] - original[-1]) < 0.01

    def test_chained_operations(self, sample_wav, tmp_path):
        path, _, _ = sample_wav
        out = str(tmp_path / "chained.wav")
        proc = BatchProcessor().remove_dc_offset().normalize().fade_in(10).fade_out(10)
        result = proc.process_file(path, out)
        assert result.success
        assert len(result.operations) == 4

    def test_gain(self, sample_wav, tmp_path):
        path, _, _ = sample_wav
        out = str(tmp_path / "gained.wav")
        proc = BatchProcessor().gain(db=-6.0)
        result = proc.process_file(path, out)
        assert result.success
        assert result.peak_after < result.peak_before

    def test_invalid_file(self, tmp_path):
        proc = BatchProcessor().normalize()
        result = proc.process_file("/nonexistent.wav", str(tmp_path / "out.wav"))
        assert not result.success
        assert result.error is not None


# ─── Spectrum Analyzer Tests ───


class TestSpectrumAnalyzer:
    def test_analyze(self, sample_wav):
        path, _, _ = sample_wav
        result = SpectrumAnalyzer.analyze(path)
        assert isinstance(result, SpectrumData)
        assert result.sample_rate == 44100
        assert len(result.frequencies) > 0
        assert len(result.magnitudes) == len(result.frequencies)

    def test_peak_frequency(self, sample_wav):
        path, _, _ = sample_wav
        result = SpectrumAnalyzer.analyze(path, fft_size=8192)
        # Peak should be near 440 Hz
        assert 400 < result.peak_frequency < 480

    def test_band_analysis(self, sample_wav):
        path, _, _ = sample_wav
        bands = SpectrumAnalyzer.band_analysis(path)
        assert isinstance(bands, BandAnalysis)
        band_dict = bands.as_dict()
        assert len(band_dict) == 7

    def test_band_text_display(self, sample_wav):
        path, _, _ = sample_wav
        bands = SpectrumAnalyzer.band_analysis(path)
        text = bands.text_display()
        assert "Sub Bass" in text
        assert "dB" in text

    def test_analyze_array(self):
        sr = 44100
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        result = SpectrumAnalyzer.analyze_array(audio, sr)
        assert 900 < result.peak_frequency < 1100

    def test_spectrogram(self, sample_wav):
        path, _, _ = sample_wav
        times, freqs, spec = SpectrumAnalyzer.spectrogram(path)
        assert len(times) > 0
        assert len(freqs) > 0
        assert spec.shape == (len(freqs), len(times))

    def test_get_band_energy(self, sample_wav):
        path, _, _ = sample_wav
        result = SpectrumAnalyzer.analyze(path)
        # 440Hz should have energy in the mid band
        mid_energy = result.get_band_energy(400, 500)
        sub_energy = result.get_band_energy(20, 40)
        assert mid_energy > sub_energy


# ─── Format Converter Tests ───


class TestFormatConverter:
    def test_convert_wav_to_flac(self, sample_wav, tmp_path):
        path, _, _ = sample_wav
        out = str(tmp_path / "converted.flac")
        result = FormatConverter.convert(path, out, output_format="flac")
        assert result.success
        assert os.path.exists(out)

    def test_convert_to_mono(self, stereo_wav, tmp_path):
        path, _, _ = stereo_wav
        out = str(tmp_path / "mono.wav")
        result = FormatConverter.convert(path, out, channels=1)
        assert result.success
        assert result.output_channels == 1

    def test_convert_unsupported_format(self, sample_wav, tmp_path):
        path, _, _ = sample_wav
        result = FormatConverter.convert(path, output_format="xyz")
        assert not result.success
        assert result.error is not None and "Unsupported" in result.error

    def test_get_file_info(self, sample_wav):
        path, _, _ = sample_wav
        info = FormatConverter.get_file_info(path)
        assert info["sample_rate"] == 44100
        assert info["channels"] == 1
        assert info["duration_seconds"] > 0

    def test_bit_depth_conversion(self, sample_wav, tmp_path):
        path, _, _ = sample_wav
        out = str(tmp_path / "16bit.wav")
        result = FormatConverter.convert(path, out, bit_depth=16)
        assert result.success

    def test_auto_output_path(self, sample_wav):
        path, _, _ = sample_wav
        result = FormatConverter.convert(path, output_format="flac")
        assert result.success
        assert result.output.endswith(".flac")
