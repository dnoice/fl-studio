"""Hardened tests: parametrized, negative, edge case, and expanded coverage.

Addresses Phase 3 of the roadmap:
- Parametrized tests to eliminate duplication
- Negative tests for invalid inputs
- Edge case tests for boundary values
- Expanded MidiFileUtils and MidiAnalyzer coverage
"""

import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audio_tools._dsp_utils import make_window, resample
from midi_tools._validation import validate_channel, validate_pitch, validate_velocity
from midi_tools.arpeggiator import ArpDirection, Arpeggiator
from midi_tools.chord_engine import ChordEngine
from midi_tools.midi_analyzer import MidiAnalyzer
from midi_tools.midi_file_utils import MidiFileUtils, NoteEvent
from midi_tools.midi_transform import MidiTransform
from midi_tools.scale_library import SCALES, note_name_to_midi
from mixing._biquad import biquad_coefficients, biquad_filter
from mixing.effects_chain import EffectsChain, EQBand
from mixing.gain_staging import GainStaging
from mixing.stereo_tools import StereoTools

# ═══════════════════════════════════════════════════════════════
# Parametrized Tests
# ═══════════════════════════════════════════════════════════════


class TestScalesParametrized:
    @pytest.mark.parametrize(
        "scale_name,expected",
        [
            ("major", [60, 62, 64, 65, 67, 69, 71]),
            ("aeolian", [60, 62, 63, 65, 67, 68, 70]),
            ("dorian", [60, 62, 63, 65, 67, 69, 70]),
            ("mixolydian", [60, 62, 64, 65, 67, 69, 70]),
        ],
    )
    def test_scale_notes(self, scale_name, expected):
        scale = SCALES[scale_name]
        notes = scale.get_notes(root=0, octave_start=4, octaves=1)
        assert notes == expected

    @pytest.mark.parametrize(
        "note_name,octave,expected_midi",
        [
            ("C", 4, 60),
            ("A", 4, 69),
            ("C", 0, 12),
            ("G", 9, 127),
            ("C#", 4, 61),
            ("Bb", 3, 58),
        ],
    )
    def test_note_conversions(self, note_name, octave, expected_midi):
        assert note_name_to_midi(note_name, octave) == expected_midi


class TestFilterTypesParametrized:
    @pytest.mark.parametrize(
        "filter_type",
        [
            "lowpass",
            "highpass",
            "bandpass",
            "notch",
            "peaking",
            "lowshelf",
            "highshelf",
        ],
    )
    def test_biquad_all_types(self, filter_type):
        b0, b1, b2, a1, a2 = biquad_coefficients(44100, filter_type, 1000, 1.0, 3.0)
        assert b0 != 0 or b1 != 0 or b2 != 0, f"{filter_type} produced zero numerator"

    @pytest.mark.parametrize(
        "filter_type",
        [
            "lowpass",
            "highpass",
            "bandpass",
            "notch",
            "peaking",
        ],
    )
    def test_biquad_filter_output_shape(self, filter_type):
        sr = 44100
        audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32) * 0.5
        result = biquad_filter(audio, sr, filter_type, 1000, 1.0, 3.0)
        assert result.shape == audio.shape


class TestArpDirectionsParametrized:
    @pytest.mark.parametrize(
        "direction",
        [
            ArpDirection.UP,
            ArpDirection.DOWN,
            ArpDirection.UP_DOWN,
            ArpDirection.DOWN_UP,
            ArpDirection.RANDOM,
            ArpDirection.AS_PLAYED,
            ArpDirection.CONVERGE,
            ArpDirection.DIVERGE,
            ArpDirection.THUMB,
        ],
    )
    def test_all_directions_produce_output(self, direction):
        arp = Arpeggiator(direction=direction, rate=0.25, octaves=1)
        pattern = arp.generate([60, 64, 67])
        assert len(pattern.notes) > 0, f"{direction.value} produced empty pattern"


class TestEffectPresetsParametrized:
    @pytest.mark.parametrize(
        "factory",
        [
            EffectsChain.vocal_chain,
            EffectsChain.drum_bus_chain,
            EffectsChain.master_chain,
            EffectsChain.bass_chain,
        ],
    )
    def test_preset_chain_produces_output(self, factory):
        sr = 44100
        audio = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32) * 0.5
        chain = factory()
        result = chain.process(audio, sr)
        assert len(result) == len(audio)
        assert not np.any(np.isnan(result)), f"{factory.__name__} produced NaN"


# ═══════════════════════════════════════════════════════════════
# Negative Tests (Invalid Inputs)
# ═══════════════════════════════════════════════════════════════


class TestBiquadNegative:
    def test_zero_sample_rate_raises(self):
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            biquad_coefficients(0, "lowpass", 1000, 1.0, 0)

    def test_negative_sample_rate_raises(self):
        with pytest.raises(ValueError, match="Sample rate must be positive"):
            biquad_coefficients(-44100, "lowpass", 1000, 1.0, 0)

    def test_zero_q_raises(self):
        with pytest.raises(ValueError, match="Q factor must be positive"):
            biquad_coefficients(44100, "lowpass", 1000, 0, 0)

    def test_negative_q_raises(self):
        with pytest.raises(ValueError, match="Q factor must be positive"):
            biquad_coefficients(44100, "lowpass", 1000, -1.0, 0)

    def test_freq_above_nyquist_clamped(self):
        # Should not raise, but clamp to below Nyquist
        b0, b1, b2, a1, a2 = biquad_coefficients(44100, "lowpass", 30000, 1.0, 0)
        assert not np.isnan(b0)


class TestMidiValidationNegative:
    def test_validate_pitch_below_range(self):
        assert validate_pitch(-10) == 0

    def test_validate_pitch_above_range(self):
        assert validate_pitch(200) == 127

    def test_validate_velocity_below_range(self):
        assert validate_velocity(-5) == 0

    def test_validate_velocity_above_range(self):
        assert validate_velocity(300) == 127

    def test_validate_channel_below_range(self):
        assert validate_channel(-1) == 0

    def test_validate_channel_above_range(self):
        assert validate_channel(20) == 15


class TestTransposeNegative:
    def test_transpose_clips_at_boundaries(self):
        notes = [NoteEvent(0, 100, 0, 480), NoteEvent(127, 100, 480, 480)]
        # Transposing down should drop the 0-pitch note
        result = MidiTransform.transpose(notes, -1)
        assert all(0 <= n.pitch <= 127 for n in result)

    def test_transpose_high_removes_out_of_range(self):
        notes = [NoteEvent(120, 100, 0, 480)]
        result = MidiTransform.transpose(notes, 20)
        assert len(result) == 0  # 140 is out of range

    def test_transpose_empty_list(self):
        assert MidiTransform.transpose([], 5) == []


class TestFLPParserNegative:
    def test_parse_nonexistent_file(self):
        from workflow.flp_parser import FLPParser

        with pytest.raises(FileNotFoundError):
            FLPParser.parse("/nonexistent/path/song.flp")

    def test_parse_invalid_header(self, tmp_path):
        from workflow.flp_parser import FLPParser

        bad_file = tmp_path / "bad.flp"
        bad_file.write_bytes(b"NOT_FLP_DATA")
        with pytest.raises(ValueError, match="Not a valid FLP"):
            FLPParser.parse(str(bad_file))

    def test_parse_truncated_file(self, tmp_path):
        from workflow.flp_parser import FLPParser

        truncated = tmp_path / "truncated.flp"
        truncated.write_bytes(b"FLhd\x06\x00\x00\x00")  # Header but no data
        with pytest.raises((ValueError, struct.error)):
            FLPParser.parse(str(truncated))


class TestSpectrumAnalyzerNegative:
    def test_analyze_nonexistent_file(self):
        from audio_tools.spectrum_analyzer import SpectrumAnalyzer

        with pytest.raises(IOError):
            SpectrumAnalyzer.analyze("/nonexistent/audio.wav")


class TestSampleSlicerNegative:
    def test_slice_nonexistent_file(self):
        from audio_tools.sample_slicer import SampleSlicer

        with pytest.raises(IOError):
            SampleSlicer.slice("/nonexistent/audio.wav")


class TestEQBandNegative:
    def test_zero_freq_clamped(self):
        eq = EQBand(freq=0.0, gain_db=3.0, q=1.0)
        assert eq.freq == 1.0

    def test_negative_freq_clamped(self):
        eq = EQBand(freq=-100.0, gain_db=3.0, q=1.0)
        assert eq.freq == 1.0

    def test_zero_q_clamped(self):
        eq = EQBand(freq=1000.0, gain_db=3.0, q=0.0)
        assert eq.q == 0.01


# ═══════════════════════════════════════════════════════════════
# Edge Case Tests (Boundary Values)
# ═══════════════════════════════════════════════════════════════


class TestMidiBoundaryValues:
    @pytest.mark.parametrize("pitch", [0, 1, 63, 64, 126, 127])
    def test_transpose_boundary_pitches(self, pitch):
        notes = [NoteEvent(pitch, 100, 0, 480)]
        result = MidiTransform.transpose(notes, 0)
        assert len(result) == 1
        assert result[0].pitch == pitch

    @pytest.mark.parametrize("velocity", [1, 64, 127])
    def test_velocity_scale_boundary(self, velocity):
        notes = [NoteEvent(60, velocity, 0, 480)]
        result = MidiTransform.velocity_scale(notes, 1.0)
        assert result[0].velocity == velocity

    def test_arpeggiator_single_note(self):
        arp = Arpeggiator(direction=ArpDirection.UP_DOWN, rate=0.25, octaves=1)
        pattern = arp.generate([60])
        assert len(pattern.notes) >= 1

    def test_arpeggiator_two_notes_up_down(self):
        arp = Arpeggiator(direction=ArpDirection.UP_DOWN, rate=0.25, octaves=1)
        pattern = arp.generate([60, 64])
        assert len(pattern.notes) == 2  # Should return [60, 64] without crash

    def test_arpeggiator_empty_input(self):
        arp = Arpeggiator(direction=ArpDirection.UP, rate=0.25)
        pattern = arp.generate([])
        assert len(pattern.notes) == 0

    def test_chord_highest_octave(self):
        # Building a chord at the top of MIDI range
        chord = ChordEngine.build(120, "major")
        assert all(0 <= n <= 127 for n in chord.notes)


class TestAudioEdgeCases:
    def test_gain_staging_silent_audio(self):
        audio = np.zeros(1000, dtype=np.float32)
        gs = GainStaging()
        report = gs.analyze(audio)
        assert report.peak_db < -90
        assert not report.clipping

    def test_gain_staging_dc_only(self):
        audio = np.ones(1000, dtype=np.float32) * 0.5
        dc = GainStaging.dc_offset(audio)
        assert abs(dc - 0.5) < 0.01

    def test_stereo_tools_mono_input_width(self):
        """Width on a mono-duplicated stereo signal should produce mono."""
        mono = np.sin(np.arange(1000) * 0.1)
        stereo = np.column_stack([mono, mono])
        result = StereoTools.width(stereo, 0.0)
        np.testing.assert_array_almost_equal(result[:, 0], result[:, 1])

    def test_biquad_single_sample(self):
        audio = np.array([1.0], dtype=np.float32)
        result = biquad_filter(audio, 44100, "lowpass", 1000, 0.707, 0)
        assert len(result) == 1
        assert not np.isnan(result[0])

    def test_biquad_freq_at_nyquist_boundary(self):
        """Frequency near Nyquist should be clamped, not crash."""
        b0, b1, b2, a1, a2 = biquad_coefficients(44100, "lowpass", 22049, 1.0, 0)
        assert not np.isnan(b0)

    def test_resample_same_rate(self):
        audio = np.ones(100, dtype=np.float32)
        result = resample(audio, 44100, 44100)
        np.testing.assert_array_equal(result, audio)

    def test_resample_upsample(self):
        audio = np.sin(np.arange(100, dtype=np.float32) * 0.1)
        result = resample(audio, 22050, 44100)
        assert len(result) == 200

    def test_resample_stereo(self):
        mono = np.sin(np.arange(100, dtype=np.float32) * 0.1)
        stereo = np.column_stack([mono, mono])
        result = resample(stereo, 22050, 44100)
        assert result.shape == (200, 2)

    def test_make_window_types(self):
        for wtype in ("hanning", "hamming", "blackman", "rectangular"):
            w = make_window(1024, wtype)
            assert len(w) == 1024

    def test_make_window_unknown_defaults_hanning(self):
        w1 = make_window(512, "unknown_type")
        w2 = make_window(512, "hanning")
        np.testing.assert_array_equal(w1, w2)


# ═══════════════════════════════════════════════════════════════
# Expanded MidiFileUtils & MidiAnalyzer Coverage
# ═══════════════════════════════════════════════════════════════


class TestMidiFileUtilsExpanded:
    def test_create_and_extract_notes(self):
        mid = MidiFileUtils.create(480, 120)
        notes = [
            NoteEvent(60, 100, 0, 480),
            NoteEvent(64, 90, 480, 480),
            NoteEvent(67, 80, 960, 480),
        ]
        track = MidiFileUtils.notes_to_track(notes, track_name="Test")
        mid.tracks.append(track)
        extracted = MidiFileUtils.extract_notes(mid)
        assert len(extracted) == 3
        assert extracted[0].pitch == 60
        assert extracted[1].pitch == 64
        assert extracted[2].pitch == 67

    def test_get_tempo(self):
        mid = MidiFileUtils.create(480, 140)
        tempo = MidiFileUtils.get_tempo(mid)
        assert abs(tempo - 140.0) < 0.5

    def test_set_tempo(self):
        mid = MidiFileUtils.create(480, 120)
        MidiFileUtils.set_tempo(mid, 160.0)
        assert abs(MidiFileUtils.get_tempo(mid) - 160.0) < 0.5

    def test_transpose(self):
        mid = MidiFileUtils.create(480, 120)
        notes = [NoteEvent(60, 100, 0, 480)]
        mid.tracks.append(MidiFileUtils.notes_to_track(notes))
        transposed = MidiFileUtils.transpose(mid, 5)
        result_notes = MidiFileUtils.extract_notes(transposed)
        assert result_notes[0].pitch == 65

    def test_quantize(self):
        mid = MidiFileUtils.create(480, 120)
        notes = [NoteEvent(60, 100, 50, 480)]  # Slightly off-grid
        mid.tracks.append(MidiFileUtils.notes_to_track(notes))
        quantized = MidiFileUtils.quantize(mid, grid=0.25)
        result_notes = MidiFileUtils.extract_notes(quantized)
        assert result_notes[0].start_tick % 120 == 0  # Snapped to grid

    def test_split_by_channel(self):
        mid = MidiFileUtils.create(480, 120)
        notes_ch0 = [NoteEvent(60, 100, 0, 480, channel=0)]
        notes_ch1 = [NoteEvent(48, 100, 0, 480, channel=1)]
        track = MidiFileUtils.notes_to_track(notes_ch0 + notes_ch1)
        mid.tracks.append(track)
        splits = MidiFileUtils.split_by_channel(mid)
        assert len(splits) == 2
        assert 0 in splits
        assert 1 in splits

    def test_split_by_note_range(self):
        mid = MidiFileUtils.create(480, 120)
        notes = [
            NoteEvent(40, 100, 0, 480),  # low
            NoteEvent(72, 100, 0, 480),  # high
        ]
        mid.tracks.append(MidiFileUtils.notes_to_track(notes))
        low, high = MidiFileUtils.split_by_note_range(mid, split_note=60)
        low_notes = MidiFileUtils.extract_notes(low)
        high_notes = MidiFileUtils.extract_notes(high)
        assert all(n.pitch < 60 for n in low_notes)
        assert all(n.pitch >= 60 for n in high_notes)

    def test_merge_files(self):
        mid1 = MidiFileUtils.create(480, 120)
        mid1.tracks.append(
            MidiFileUtils.notes_to_track([NoteEvent(60, 100, 0, 480)], track_name="A")
        )
        mid2 = MidiFileUtils.create(480, 120)
        mid2.tracks.append(
            MidiFileUtils.notes_to_track([NoteEvent(72, 100, 0, 480)], track_name="B")
        )
        merged = MidiFileUtils.merge_files([mid1, mid2])
        assert len(merged.tracks) >= 3  # tempo + 2 tracks

    def test_merge_empty(self):
        result = MidiFileUtils.merge_files([])
        assert result is not None

    def test_info(self):
        mid = MidiFileUtils.create(480, 120)
        notes = [NoteEvent(60, 100, 0, 480), NoteEvent(72, 80, 480, 240)]
        mid.tracks.append(MidiFileUtils.notes_to_track(notes))
        info = MidiFileUtils.info(mid)
        assert info["total_notes"] == 2
        assert info["ticks_per_beat"] == 480
        assert info["pitch_range"] == (60, 72)

    def test_get_duration_beats(self):
        mid = MidiFileUtils.create(480, 120)
        notes = [NoteEvent(60, 100, 0, 960)]  # 2 beats
        mid.tracks.append(MidiFileUtils.notes_to_track(notes))
        dur = MidiFileUtils.get_duration_beats(mid)
        assert dur >= 2.0

    def test_notes_to_midi_file(self):
        notes = [NoteEvent(60, 100, 0, 480), NoteEvent(64, 90, 480, 480)]
        mid = MidiFileUtils.notes_to_midi_file(notes, tempo=130)
        assert abs(MidiFileUtils.get_tempo(mid) - 130.0) < 0.5
        extracted = MidiFileUtils.extract_notes(mid)
        assert len(extracted) == 2

    def test_write_and_read_roundtrip(self, tmp_path):
        notes = [NoteEvent(60, 100, 0, 480), NoteEvent(67, 80, 480, 480)]
        mid = MidiFileUtils.notes_to_midi_file(notes, tempo=120)
        path = str(tmp_path / "test.mid")
        MidiFileUtils.write(mid, path)
        loaded = MidiFileUtils.read(path)
        loaded_notes = MidiFileUtils.extract_notes(loaded)
        assert len(loaded_notes) == 2
        assert loaded_notes[0].pitch == 60


class TestMidiAnalyzerExpanded:
    def _make_midi(self, notes):
        return MidiFileUtils.notes_to_midi_file(notes, tempo=120)

    def test_analyze_basic(self):
        notes = [
            NoteEvent(60, 100, 0, 480),
            NoteEvent(64, 90, 480, 480),
            NoteEvent(67, 80, 960, 480),
        ]
        mid = self._make_midi(notes)
        result = MidiAnalyzer.analyze(mid)
        assert result.timing_stats.total_notes == 3
        assert result.velocity_stats.min == 80
        assert result.velocity_stats.max == 100

    def test_analyze_empty_midi(self):
        mid = MidiFileUtils.create(480, 120)
        result = MidiAnalyzer.analyze(mid)
        assert result.timing_stats.total_notes == 0

    def test_analyze_single_note(self):
        notes = [NoteEvent(60, 100, 0, 480)]
        mid = self._make_midi(notes)
        result = MidiAnalyzer.analyze(mid)
        assert result.timing_stats.total_notes == 1

    def test_detect_key(self):
        # C major scale notes
        notes = [
            NoteEvent(p, 100, i * 480, 480) for i, p in enumerate([60, 62, 64, 65, 67, 69, 71])
        ]
        mid = self._make_midi(notes)
        keys = MidiAnalyzer.detect_key(mid)
        assert len(keys) > 0

    def test_compare(self):
        notes_a = [NoteEvent(60, 100, 0, 480), NoteEvent(64, 90, 480, 480)]
        notes_b = [
            NoteEvent(60, 100, 0, 480),
            NoteEvent(64, 90, 480, 480),
            NoteEvent(67, 80, 960, 480),
        ]
        mid_a = self._make_midi(notes_a)
        mid_b = self._make_midi(notes_b)
        diff = MidiAnalyzer.compare(mid_a, mid_b)
        assert diff["note_count_diff"] == 1

    def test_octave_distribution(self):
        notes = [
            NoteEvent(24, 100, 0, 480),  # Octave 1
            NoteEvent(60, 100, 480, 480),  # Octave 4
            NoteEvent(96, 100, 960, 480),  # Octave 7
        ]
        mid = self._make_midi(notes)
        result = MidiAnalyzer.analyze(mid)
        assert len(result.octave_distribution) >= 2

    def test_low_pitch_octave_no_negative_key(self):
        """Pitches 0-11 should map to octave 0, not negative."""
        notes = [NoteEvent(0, 100, 0, 480), NoteEvent(11, 100, 480, 480)]
        mid = self._make_midi(notes)
        result = MidiAnalyzer.analyze(mid)
        assert all(k >= 0 for k in result.octave_distribution)

    def test_interval_distribution(self):
        notes = [
            NoteEvent(60, 100, 0, 480),
            NoteEvent(64, 100, 480, 480),  # +4 semitones
            NoteEvent(67, 100, 960, 480),  # +3 semitones
        ]
        mid = self._make_midi(notes)
        result = MidiAnalyzer.analyze(mid)
        assert len(result.interval_distribution) > 0

    def test_summary(self):
        notes = [NoteEvent(60, 100, 0, 480)]
        mid = self._make_midi(notes)
        result = MidiAnalyzer.analyze(mid)
        summary = result.summary()
        assert "MIDI Analysis" in summary
        assert "Total Notes:" in summary

    def test_note_distribution_histogram(self):
        notes = [NoteEvent(60, 100, i * 480, 480) for i in range(5)]
        mid = self._make_midi(notes)
        result = MidiAnalyzer.analyze(mid)
        hist = result.note_distribution.as_histogram()
        assert "C" in hist

    def test_velocity_histogram(self):
        notes = [
            NoteEvent(60, 20, 0, 480),
            NoteEvent(64, 80, 480, 480),
            NoteEvent(67, 120, 960, 480),
        ]
        mid = self._make_midi(notes)
        result = MidiAnalyzer.analyze(mid)
        assert result.velocity_stats.histogram["pp (1-31)"] == 1
        assert result.velocity_stats.histogram["ff (112-127)"] == 1
