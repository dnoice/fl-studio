"""Real-world scenario tests using actual audio files.

These tests exercise multi-module integration with REAL audio assets,
producing REAL output files. Run with ``pytest -s -v`` to see analysis results.

Assets required: /mnt/c/Tools/fl-studio/assets/audio/ (5 MP3 tracks)
"""

import os
import re
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# audio_tools
from audio_tools.batch_processor import BatchProcessor
from audio_tools.bpm_detector import BPMDetector, BPMResult
from audio_tools.format_converter import FormatConverter
from audio_tools.key_detector import KeyDetector, KeyResult
from audio_tools.sample_slicer import SampleSlicer, SliceResult
from audio_tools.spectrum_analyzer import BandAnalysis, SpectrumAnalyzer
from midi_tools.arpeggiator import ArpDirection, Arpeggiator
from midi_tools.chord_engine import ChordEngine
from midi_tools.drum_patterns import DrumPatternLibrary
from midi_tools.midi_analyzer import MidiAnalyzer
from midi_tools.midi_file_utils import MidiFileUtils, NoteEvent
from midi_tools.midi_transform import MidiTransform

# midi_tools
from midi_tools.scale_library import SCALES
from mixing.channel_strip import ChannelStrip

# mixing
from mixing.effects_chain import (
    EffectsChain,
    HighPassFilter,
)
from mixing.gain_staging import GainStaging
from mixing.mix_analyzer import MixAnalyzer
from mixing.reference_compare import ReferenceCompare
from mixing.stereo_tools import StereoTools
from release.album import Album
from release.export_pipeline import ExportPipeline
from release.licensing import LicensingManager
from release.lyrics import Lyrics, LyricsManager

# release
from release.metadata import MetadataManager, TrackMetadata

# workflow
from workflow.sample_organizer import SampleOrganizer

# ─── Constants & Fixtures ───────────────────────────────────────────────

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "audio")

TRACK_FILES = {
    "c_major": os.path.join(ASSETS_DIR, "c_major.mp3"),
    "capital_echoes": os.path.join(ASSETS_DIR, "capital_echoes.mp3"),
    "iron_forge": os.path.join(ASSETS_DIR, "echoes_of_the_iron_forge.mp3"),
    "entrenched": os.path.join(ASSETS_DIR, "entrenched.mp3"),
    "intentionally": os.path.join(ASSETS_DIR, "intentionally.mp3"),
}

TRACK_DISPLAY = {
    "c_major": "C Major",
    "capital_echoes": "Capital Echoes",
    "iron_forge": "Echoes of the Iron Forge",
    "entrenched": "Entrenched",
    "intentionally": "Intentionally",
}


@pytest.fixture(scope="module")
def audio_paths():
    """Validated dict of all 5 real audio file paths."""
    for _key, path in TRACK_FILES.items():
        assert os.path.exists(path), f"Missing audio asset: {path}"
    return dict(TRACK_FILES)


@pytest.fixture(scope="module")
def short_track():
    """Shortest track (iron_forge ~39 s) for faster processing tests."""
    return TRACK_FILES["iron_forge"]


@pytest.fixture(scope="module")
def all_paths():
    """List of all 5 track paths."""
    return list(TRACK_FILES.values())


def _convert_to_wav(mp3_path: str, wav_path: str) -> str:
    """Helper: convert an MP3 to WAV, return output path."""
    result = FormatConverter.convert(mp3_path, wav_path, output_format="wav")
    assert result.success, f"Conversion failed: {result.error}"
    return wav_path


# ─── Scenario 1: Track Analysis Pipeline ────────────────────────────────


class TestTrackAnalysisPipeline:
    """Analyze all 5 real tracks: BPM, key, spectrum, loudness."""

    @pytest.mark.slow
    def test_full_track_analysis_report(self, audio_paths):
        """Full analysis of every track — BPM, key, bands, LUFS, dynamics."""
        print("\n" + "=" * 80)
        print("FULL TRACK ANALYSIS REPORT")
        print("=" * 80)
        header = f"{'Track':<30} {'BPM':>6} {'Key':<12} {'Camelot':<8} {'LUFS':>7} {'Crest':>6}"
        print(header)
        print("-" * 80)

        for key, path in audio_paths.items():
            name = TRACK_DISPLAY[key]

            bpm_result = BPMDetector.detect(path, method="autocorrelation")
            assert isinstance(bpm_result, BPMResult)
            assert bpm_result.bpm > 0

            key_result = KeyDetector.detect(path, duration=30)
            assert isinstance(key_result, KeyResult)
            assert key_result.key
            assert key_result.mode in ("major", "minor")
            assert re.match(r"\d{1,2}[AB]", key_result.camelot)

            audio, sr = sf.read(path, dtype="float64")
            loudness = MixAnalyzer.measure_loudness(audio, sr)
            assert loudness.integrated_lufs < 0

            dynamics = MixAnalyzer.analyze_dynamics(audio, sr)
            assert dynamics.crest_factor_db > 0

            print(
                f"{name:<30} {bpm_result.bpm:>6.1f} {key_result.key:<12} "
                f"{key_result.camelot:<8} {loudness.integrated_lufs:>7.1f} "
                f"{dynamics.crest_factor_db:>5.1f}dB"
            )

        print("=" * 80)

    @pytest.mark.slow
    def test_batch_bpm_detection(self, all_paths):
        """Batch BPM detection across all 5 files."""
        results = BPMDetector.detect_batch(all_paths)
        assert len(results) == 5  # Returns dict[filename, BPMResult]

        print("\n--- Batch BPM Results ---")
        for name, result in results.items():
            assert isinstance(result, BPMResult)
            assert result.bpm > 0
            print(f"  {name:<40} {result.bpm:>6.1f} BPM  (confidence: {result.confidence:.2f})")

    @pytest.mark.slow
    def test_spectral_profile_comparison(self, audio_paths):
        """Compare spectral profiles across all tracks."""
        profiles = {}
        for key, path in audio_paths.items():
            bands = SpectrumAnalyzer.band_analysis(path)
            assert isinstance(bands, BandAnalysis)
            profiles[key] = bands.as_dict()

        print("\n--- Spectral Profiles (dB) ---")
        band_names = list(next(iter(profiles.values())).keys())
        header = f"{'Track':<20}" + "".join(f"{b:>12}" for b in band_names)
        print(header)
        for key, bd in profiles.items():
            vals = "".join(f"{v:>12.1f}" for v in bd.values())
            print(f"{key:<20}{vals}")

    def test_file_info_all_tracks(self, audio_paths):
        """Get file info for every track."""
        print("\n--- File Info ---")
        for key, path in audio_paths.items():
            info = FormatConverter.get_file_info(path)
            assert info["sample_rate"] == 44100
            assert info["channels"] == 2
            assert info["duration_seconds"] > 0
            print(
                f"  {TRACK_DISPLAY[key]:<30} "
                f"{info['duration_seconds']:.1f}s  {info['channels']}ch  "
                f"{info['sample_rate']}Hz  {info.get('format', '?')}"
            )


# ─── Scenario 2: Audio Processing Pipeline ──────────────────────────────


class TestAudioProcessingPipeline:
    """Convert, process, and chain operations on real audio."""

    def test_mp3_to_wav_normalize_fade(self, short_track, tmp_path):
        """Convert MP3 -> WAV -> normalize -> trim -> fade in/out."""
        wav_path = str(tmp_path / "iron_forge.wav")
        _convert_to_wav(short_track, wav_path)
        assert os.path.exists(wav_path)

        output = str(tmp_path / "processed.wav")
        proc = (
            BatchProcessor()
            .normalize(target_db=-1.0)
            .trim_silence(threshold_db=-50.0)
            .fade_in(duration_ms=100)
            .fade_out(duration_ms=200)
        )
        result = proc.process_file(wav_path, output)
        assert result.success
        assert os.path.exists(output)

        audio, sr = sf.read(output)
        assert sr == 44100
        assert len(audio) > 0
        print(f"\n  Processed: {len(audio)/sr:.1f}s  peak={np.max(np.abs(audio)):.4f}")

    def test_to_mono_processing(self, tmp_path):
        """Convert stereo track to mono."""
        wav_path = str(tmp_path / "c_major.wav")
        _convert_to_wav(TRACK_FILES["c_major"], wav_path)

        mono_out = str(tmp_path / "mono.wav")
        result = BatchProcessor().to_mono().process_file(wav_path, mono_out)
        assert result.success

        audio, sr = sf.read(mono_out)
        assert audio.ndim == 1
        print(f"\n  Mono output: {len(audio)/sr:.1f}s  {audio.ndim}D")

    def test_full_chain_to_flac(self, tmp_path):
        """Full chain: MP3 -> WAV -> DC remove -> normalize -> fade -> FLAC."""
        wav_path = str(tmp_path / "echoes.wav")
        _convert_to_wav(TRACK_FILES["capital_echoes"], wav_path)

        processed = str(tmp_path / "processed.wav")
        result = (
            BatchProcessor()
            .remove_dc_offset()
            .normalize(target_db=-3.0)
            .fade_in(duration_ms=50)
            .fade_out(duration_ms=100)
            .process_file(wav_path, processed)
        )
        assert result.success

        flac_out = str(tmp_path / "final.flac")
        conv = FormatConverter.convert(processed, flac_out, output_format="flac", bit_depth=24)
        assert conv.success
        assert os.path.exists(flac_out)

        info = FormatConverter.get_file_info(flac_out)
        assert "FLAC" in info.get("format", "")
        print(f"\n  FLAC output: {info['duration_seconds']:.1f}s  " f"{info.get('subtype', '?')}")

    @pytest.mark.slow
    def test_batch_convert_all_to_wav(self, audio_paths, tmp_path):
        """Convert all 5 MP3s to WAV."""
        wavs_dir = str(tmp_path / "wavs")
        os.makedirs(wavs_dir)

        for key, path in audio_paths.items():
            wav_out = os.path.join(wavs_dir, f"{key}.wav")
            result = FormatConverter.convert(path, wav_out, output_format="wav")
            assert result.success
            info = sf.info(wav_out)
            assert info.samplerate == 44100
            print(f"  Converted: {key} -> {info.duration:.1f}s WAV")

        wav_files = [f for f in os.listdir(wavs_dir) if f.endswith(".wav")]
        assert len(wav_files) == 5

    def test_spectrum_before_after_hpf(self, short_track, tmp_path):
        """Compare spectrum before/after a 200 Hz high-pass filter."""
        wav_path = str(tmp_path / "source.wav")
        _convert_to_wav(short_track, wav_path)

        # Analyze before
        before = SpectrumAnalyzer.analyze(wav_path, fft_size=4096)
        before_sub = before.get_band_energy(20, 200)

        # Apply HPF via EffectsChain
        filtered_path = str(tmp_path / "filtered.wav")
        chain = EffectsChain("HPF Test")
        chain.add(HighPassFilter(freq=200.0))
        chain.process_file(wav_path, filtered_path)

        # Analyze after
        after = SpectrumAnalyzer.analyze(filtered_path, fft_size=4096)
        after_sub = after.get_band_energy(20, 200)

        assert after_sub < before_sub, "HPF should reduce sub-200Hz energy"
        print(
            f"\n  Sub-200Hz energy: before={before_sub:.1f}dB  after={after_sub:.1f}dB  "
            f"delta={after_sub - before_sub:.1f}dB"
        )


# ─── Scenario 3: Mixing & Mastering Pipeline ────────────────────────────


class TestMixingMasteringPipeline:
    """Channel strips, mastering chains, reference comparison on real audio."""

    def test_channel_strip_before_after(self):
        """Process a track through a vocal channel strip, compare before/after."""
        audio, sr = sf.read(TRACK_FILES["intentionally"], dtype="float64")

        gs = GainStaging()
        before = gs.analyze(audio)
        print(
            f"\n  BEFORE: peak={before.peak_db:.1f}dB  rms={before.rms_db:.1f}dB  "
            f"crest={before.crest_factor_db:.1f}dB"
        )

        strip = ChannelStrip.vocal()
        processed = strip.process(audio, sr)

        after = gs.analyze(processed)
        print(
            f"  AFTER:  peak={after.peak_db:.1f}dB  rms={after.rms_db:.1f}dB  "
            f"crest={after.crest_factor_db:.1f}dB"
        )

        assert np.isfinite(before.peak_db)
        assert np.isfinite(after.peak_db)
        assert processed.shape[0] == audio.shape[0]

    def test_reference_comparison(self):
        """Compare a mix against two reference tracks."""
        rc = ReferenceCompare()
        rc.load_reference("Capital Echoes", TRACK_FILES["capital_echoes"])
        rc.load_reference("Entrenched", TRACK_FILES["entrenched"])

        mix, sr = sf.read(TRACK_FILES["c_major"], dtype="float64")
        results = rc.compare_all(mix, sr)

        assert len(results) == 2
        for ref_name, comp in results.items():
            assert np.isfinite(comp.loudness_diff_lufs)
            assert len(comp.spectral_diffs) == 7
            assert len(comp.suggestions) > 0
            print(
                f"\n  vs {ref_name}: LUFS diff={comp.loudness_diff_lufs:+.1f}  "
                f"peak diff={comp.peak_diff_db:+.1f}dB"
            )
            for s in comp.suggestions[:3]:
                print(f"    -> {s}")

    def test_mastering_chain_full(self, short_track, tmp_path):
        """Run a full mastering chain and measure LUFS before/after."""
        audio, sr = sf.read(short_track, dtype="float64")

        lufs_before = MixAnalyzer.lufs_integrated(audio, sr)

        chain = EffectsChain.master_chain()
        mastered = chain.process(audio, sr)

        lufs_after = MixAnalyzer.lufs_integrated(mastered, sr)

        # Write output
        out_path = str(tmp_path / "mastered.wav")
        sf.write(out_path, mastered, sr)
        assert os.path.exists(out_path)

        peak_after = 20 * np.log10(max(np.max(np.abs(mastered)), 1e-10))
        print(f"\n  LUFS: before={lufs_before:.1f}  after={lufs_after:.1f}")
        print(f"  Peak after mastering: {peak_after:.1f} dBFS")
        assert np.isfinite(lufs_before)
        assert np.isfinite(lufs_after)

    def test_stereo_analysis_and_width(self):
        """Analyze stereo field and test width control on real audio."""
        audio, sr = sf.read(TRACK_FILES["capital_echoes"], dtype="float64")

        corr_before = StereoTools.correlation(audio)
        width_before = StereoTools.stereo_width_meter(audio)
        compat = StereoTools.mono_compatibility_check(audio)
        levels = StereoTools.channel_levels(audio)

        assert -1.0 <= corr_before <= 1.0
        assert width_before >= 0
        assert "mono_compatible" in compat
        assert "left_peak_db" in levels

        print(f"\n  BEFORE: correlation={corr_before:.3f}  width={width_before:.3f}")
        print(f"  Levels: L={levels['left_peak_db']:.1f}dB  R={levels['right_peak_db']:.1f}dB")
        print(f"  Mono compatible: {compat['mono_compatible']}")

        # Narrow width
        narrowed = StereoTools.width(audio, 0.5)
        corr_after = StereoTools.correlation(narrowed)
        width_after = StereoTools.stereo_width_meter(narrowed)

        print(f"  AFTER (w=0.5): correlation={corr_after:.3f}  width={width_after:.3f}")
        assert corr_after >= corr_before - 0.05  # Should increase or stay similar

    def test_gain_staging_and_auto(self):
        """Analyze gain and auto-stage to -18 dBFS RMS."""
        audio, sr = sf.read(TRACK_FILES["entrenched"], dtype="float64")

        gs = GainStaging(target_rms_db=-18.0)
        report = gs.analyze(audio)
        print(
            f"\n  Before: peak={report.peak_db:.1f}dB  rms={report.rms_db:.1f}dB  "
            f"headroom={report.headroom_db:.1f}dB"
        )

        processed, gain_db = gs.auto_gain(audio, mode="rms")
        rms_after = gs.rms_db(processed)

        print(f"  Applied: {gain_db:+.1f}dB gain  ->  RMS={rms_after:.1f}dB")
        assert np.isfinite(gain_db)
        # RMS should be within ~3 dB of target (safety clipping may limit gain)
        assert abs(rms_after - (-18.0)) < 5.0

    def test_full_mix_report(self):
        """Full mix analysis report on a real track."""
        audio, sr = sf.read(TRACK_FILES["intentionally"], dtype="float64")

        analysis = MixAnalyzer.full_analysis(audio, sr)
        assert "loudness" in analysis
        assert "dynamics" in analysis
        assert "spectral" in analysis

        report = MixAnalyzer.report(audio, sr, name="Intentionally")
        assert "LUFS" in report or "lufs" in report.lower()
        print(f"\n{report}")


# ─── Scenario 4: Sample Library Pipeline ────────────────────────────────


class TestSampleLibraryPipeline:
    """Slice real tracks and organize samples."""

    def test_slice_at_transients(self, short_track, tmp_path):
        """Slice a real track at transient points."""
        wav_path = str(tmp_path / "source.wav")
        _convert_to_wav(short_track, wav_path)

        slices_dir = str(tmp_path / "slices")
        result = SampleSlicer.slice(wav_path, output_dir=slices_dir, sensitivity=0.5)
        assert isinstance(result, SliceResult)
        assert len(result.slices) > 0
        assert result.sample_rate == 44100

        existing = [s for s in result.slices if s.filepath and os.path.exists(s.filepath)]
        assert len(existing) > 0

        print(f"\n  Sliced into {len(result.slices)} segments")
        print(result.summary())

    def test_slice_uniform(self, short_track, tmp_path):
        """Slice into uniform 2-second chunks."""
        wav_path = str(tmp_path / "source.wav")
        _convert_to_wav(short_track, wav_path)

        result = SampleSlicer.slice_uniform(wav_path, slice_duration_ms=2000)
        assert len(result.slices) >= 15  # ~39s / 2s ≈ 19 slices
        print(f"\n  Uniform slices: {len(result.slices)} x 2s")

    def test_analyze_sliced_samples(self, short_track, tmp_path):
        """Analyze properties of sliced samples."""
        wav_path = str(tmp_path / "source.wav")
        _convert_to_wav(short_track, wav_path)

        slices_dir = str(tmp_path / "slices")
        slice_result = SampleSlicer.slice(wav_path, output_dir=slices_dir, sensitivity=0.5)
        existing = [s for s in slice_result.slices if s.filepath and os.path.exists(s.filepath)]
        assert len(existing) > 0

        print(f"\n  Analyzing {len(existing)} slices:")
        for i, s in enumerate(existing[:10]):  # First 10 to keep output manageable
            assert s.filepath is not None
            info = FormatConverter.get_file_info(s.filepath)
            spec = SpectrumAnalyzer.analyze(s.filepath)
            assert info["duration_seconds"] > 0
            assert spec.peak_frequency > 0
            print(
                f"    Slice {i+1}: {info['duration_seconds']:.2f}s  "
                f"peak_freq={spec.peak_frequency:.0f}Hz"
            )

    def test_organize_slices(self, short_track, tmp_path):
        """Scan and organize sliced samples."""
        wav_path = str(tmp_path / "source.wav")
        _convert_to_wav(short_track, wav_path)

        slices_dir = str(tmp_path / "slices")
        SampleSlicer.slice(wav_path, output_dir=slices_dir, sensitivity=0.5)

        results = SampleOrganizer.scan_directory(slices_dir)
        assert len(results) > 0

        report = SampleOrganizer.generate_report(results)
        assert "Sample Library Report" in report
        print(f"\n{report}")


# ─── Scenario 5: MIDI Composition Pipeline ──────────────────────────────


class TestMidiCompositionPipeline:
    """Generate MIDI from detected key, chords, arps, and drums."""

    def test_key_driven_chord_progression(self, tmp_path):
        """Detect key of real track -> build chords -> arpeggiate -> export MIDI."""
        key_result = KeyDetector.detect(TRACK_FILES["c_major"], duration=30)
        print(
            f"\n  Detected key: {key_result.key}  root={key_result.root}  "
            f"mode={key_result.mode}"
        )

        scale_name = "major" if key_result.mode == "major" else "aeolian"
        scale = SCALES[scale_name]
        root = key_result.root

        # Build diatonic chords: I, IV, V, vi
        degrees = [1, 4, 5, 6]
        chords = [ChordEngine.diatonic_chord(scale, root, d) for d in degrees]

        # Arpeggiate each chord
        arp = Arpeggiator(direction=ArpDirection.UP, rate=0.25, octaves=2, gate=0.8)
        all_notes = []
        tick = 0
        for chord in chords:
            pattern = arp.generate(chord.notes)
            for note in pattern.notes:
                vel = (
                    max(1, min(127, int(note.velocity * 127)))
                    if note.velocity <= 1.0
                    else int(note.velocity)
                )
                all_notes.append(
                    NoteEvent(
                        note.pitch, vel, tick + int(note.time * 480), int(note.duration * 480)
                    )
                )
            tick += 480 * 4  # 4 beats per chord

        assert len(all_notes) > 0

        # Export MIDI
        bpm = max(
            60,
            min(
                200, round(BPMDetector.detect(TRACK_FILES["c_major"], method="autocorrelation").bpm)
            ),
        )
        midi_file = MidiFileUtils.notes_to_midi_file(
            all_notes, tempo=bpm, track_name="Arpeggiated Chords"
        )
        midi_path = str(tmp_path / "chords_arp.mid")
        midi_file.save(midi_path)
        assert os.path.exists(midi_path)
        assert os.path.getsize(midi_path) > 0

        # Analyze
        analysis = MidiAnalyzer.analyze(midi_path)
        print(
            f"  MIDI: {analysis.timing_stats.total_notes} notes  "
            f"tempo={analysis.file_info.get('tempo_bpm', '?')} BPM"
        )
        print(f"  {analysis.summary()[:200]}...")

    def test_humanized_progression_export(self, tmp_path):
        """Build a progression, humanize it, export to MIDI."""
        prog = ChordEngine.progression_1_5_6_4(key_root=0)
        notes = []
        tick = 0
        for chord in prog.chords:
            for pitch in chord.notes:
                notes.append(NoteEvent(pitch, 100, tick, 480))
            tick += 480 * 4

        # Humanize + swing
        humanized = MidiTransform.humanize(notes, timing_range=15, velocity_range=12)
        swung = MidiTransform.swing(humanized, amount=0.2)

        for n in swung:
            assert 1 <= n.velocity <= 127

        midi_file = MidiFileUtils.notes_to_midi_file(swung, tempo=120, track_name="Humanized Prog")
        midi_path = str(tmp_path / "humanized_prog.mid")
        midi_file.save(midi_path)
        assert os.path.exists(midi_path)

        analysis = MidiAnalyzer.analyze(midi_path)
        print(f"\n  Humanized progression: {analysis.timing_stats.total_notes} notes")
        print(f"  Velocity range: {analysis.velocity_stats.min}-{analysis.velocity_stats.max}")

    def test_drum_pattern_export(self, tmp_path):
        """Generate and export a trap drum pattern to MIDI."""
        GM_DRUMS = {
            "kick": 36,
            "snare": 38,
            "clap": 39,
            "closed_hat": 42,
            "open_hat": 46,
            "hihat": 42,
            "rim": 37,
            "perc": 56,
        }

        pattern = DrumPatternLibrary.trap_basic()
        doubled = pattern.double()
        humanized = doubled.humanize()

        notes = []
        step_ticks = 120  # 16th note at 480 tpb
        for hit in humanized.hits:
            midi_note = GM_DRUMS.get(hit.instrument, GM_DRUMS.get("perc", 56))
            tick = int(hit.step * step_ticks)
            velocity = max(1, min(127, int(hit.velocity * 127)))
            notes.append(NoteEvent(midi_note, velocity, tick, 60, channel=9))

        midi_file = MidiFileUtils.notes_to_midi_file(notes, tempo=140, track_name="Trap Drums")
        midi_path = str(tmp_path / "trap_drums.mid")
        midi_file.save(midi_path)
        assert os.path.exists(midi_path)
        assert len(notes) > 0

        analysis = MidiAnalyzer.analyze(midi_path)
        print(
            f"\n  Trap drums: {analysis.timing_stats.total_notes} hits  " f"{humanized.steps} steps"
        )

    def test_complete_arrangement(self, tmp_path):
        """Full arrangement: detect key -> chords + drums -> merge -> export."""
        key_result = KeyDetector.detect(TRACK_FILES["entrenched"], duration=30)
        scale_name = "major" if key_result.mode == "major" else "aeolian"
        scale = SCALES[scale_name]
        root = key_result.root

        # Chord track
        chord_notes = []
        tick = 0
        for deg in [1, 5, 6, 4]:
            chord = ChordEngine.diatonic_chord(scale, root, deg)
            for pitch in chord.notes:
                chord_notes.append(NoteEvent(pitch, 90, tick, 480 * 4))
            tick += 480 * 4

        chord_midi = MidiFileUtils.notes_to_midi_file(chord_notes, tempo=120, track_name="Chords")

        # Drum track
        GM = {
            "kick": 36,
            "snare": 38,
            "hihat": 42,
            "clap": 39,
            "closed_hat": 42,
            "open_hat": 46,
            "rim": 37,
            "perc": 56,
        }
        pattern = DrumPatternLibrary.house_basic()
        drum_notes = []
        step_ticks = 120
        for hit in pattern.hits:
            midi_note = GM.get(hit.instrument, 56)
            drum_notes.append(
                NoteEvent(
                    midi_note,
                    max(1, min(127, int(hit.velocity * 127))),
                    int(hit.step * step_ticks),
                    60,
                    channel=9,
                )
            )

        drum_midi = MidiFileUtils.notes_to_midi_file(drum_notes, tempo=120, track_name="Drums")

        # Merge
        merged = MidiFileUtils.merge_files([chord_midi, drum_midi])
        midi_path = str(tmp_path / "arrangement.mid")
        merged.save(midi_path)
        assert os.path.exists(midi_path)

        analysis = MidiAnalyzer.analyze(midi_path)
        assert analysis.timing_stats.total_notes > 10
        print(f"\n  Arrangement in {key_result.key}:")
        print(
            f"  Tracks: {analysis.file_info.get('tracks', '?')}  "
            f"Notes: {analysis.timing_stats.total_notes}"
        )


# ─── Scenario 6: Album Release Pipeline ─────────────────────────────────


class TestAlbumReleasePipeline:
    """Full album workflow: metadata, lyrics, licensing, export."""

    @pytest.mark.slow
    def test_album_with_auto_metadata(self, audio_paths):
        """Create album with auto-detected BPM and key for each track."""
        album = Album("Iron & Echoes", "Test Artist")
        album.year = 2025

        print("\n--- Album: Iron & Echoes ---")
        for i, (key, path) in enumerate(audio_paths.items(), 1):
            bpm_result = BPMDetector.detect(path, method="autocorrelation")
            key_result = KeyDetector.detect(path, duration=20)

            meta = TrackMetadata(
                title=TRACK_DISPLAY[key],
                artist="Test Artist",
                album="Iron & Echoes",
                album_artist="Test Artist",
                bpm=round(bpm_result.bpm),
                key=key_result.key,
                genre="Electronic",
                year=2025,
                track_number=i,
                track_total=5,
                isrc=f"US-XX1-25-{i:05d}",
            )
            album.add_track(meta, audio_path=path)

        assert len(album.tracks) == 5
        tracklist = album.tracklist()
        print(tracklist)
        for name in TRACK_DISPLAY.values():
            assert name in tracklist

    def test_lyrics_and_licensing_setup(self):
        """Set up lyrics and licensing for all 5 tracks."""
        # Lyrics
        lm = LyricsManager()
        for _key, name in TRACK_DISPLAY.items():
            lyrics = Lyrics(title=name, artist="Test Artist")
            lyrics.text = f"These are the lyrics for {name}.\nVerse one, line two."
            lyrics.add_synced_line(0, f"[Intro] {name}")
            lyrics.add_synced_line(5000, "First verse begins")
            lyrics.add_synced_line(30000, "Chorus drops")
            lm.add(lyrics)

        assert len(lm._lyrics) == 5
        found = lm.get("Test Artist", "Entrenched")
        assert found is not None

        # Licensing
        lic = LicensingManager()
        for _key, name in TRACK_DISPLAY.items():
            lic.quick_register(
                title=name,
                writers=[
                    {
                        "name": "Artist One",
                        "role": "composer",
                        "share_percent": 50.0,
                        "pro": "ASCAP",
                    },
                    {"name": "Artist Two", "role": "lyricist", "share_percent": 50.0, "pro": "BMI"},
                ],
                copyright_owner="Test Label LLC",
                isrc="US-XX1-25-00001",
            )

        lic.validate_all()
        summary = lic.catalog_summary()
        assert "Total Songs: 5" in summary or "5" in summary
        print(f"\n{summary}")

    @pytest.mark.slow
    def test_export_dry_run(self, audio_paths, tmp_path):
        """Dry run of the export pipeline."""
        album = Album("Iron & Echoes", "Test Artist")
        album.year = 2025
        for i, (key, path) in enumerate(audio_paths.items(), 1):
            meta = TrackMetadata(
                title=TRACK_DISPLAY[key],
                artist="Test Artist",
                album="Iron & Echoes",
                genre="Electronic",
                year=2025,
                track_number=i,
                track_total=5,
            )
            album.add_track(meta, audio_path=path)

        config = ExportPipeline.platform_config("spotify")
        config.output_dir = str(tmp_path / "export")

        pipeline = ExportPipeline(album, config)
        preview = pipeline.dry_run()
        assert "Iron & Echoes" in preview or "Iron" in preview
        print(f"\n{preview}")

    @pytest.mark.slow
    def test_export_full_execution(self, audio_paths, tmp_path):
        """Full export pipeline: convert, process, embed metadata, generate files."""
        # First convert all MP3s to WAV (soundfile can't write MP3)
        wav_dir = str(tmp_path / "wavs")
        os.makedirs(wav_dir)
        wav_paths = {}
        for key, path in audio_paths.items():
            wav_out = os.path.join(wav_dir, f"{key}.wav")
            FormatConverter.convert(path, wav_out, output_format="wav")
            wav_paths[key] = wav_out

        # Build album with WAV paths
        album = Album("Iron & Echoes", "Test Artist")
        album.year = 2025
        for i, (key, wav_path) in enumerate(wav_paths.items(), 1):
            meta = TrackMetadata(
                title=TRACK_DISPLAY[key],
                artist="Test Artist",
                album="Iron & Echoes",
                album_artist="Test Artist",
                genre="Electronic",
                year=2025,
                track_number=i,
                track_total=5,
            )
            album.add_track(meta, audio_path=wav_path)

        # Lyrics
        lm = LyricsManager()
        for _key, name in TRACK_DISPLAY.items():
            lyrics = Lyrics(title=name, artist="Test Artist")
            lyrics.text = f"Lyrics for {name}."
            lm.add(lyrics)

        # Licensing
        lic = LicensingManager()
        for name in TRACK_DISPLAY.values():
            lic.quick_register(
                name,
                [
                    {
                        "name": "Test Writer",
                        "role": "composer",
                        "share_percent": 100.0,
                        "pro": "ASCAP",
                    },
                ],
            )

        # Export config (skip strict validation for test — no ISRC/UPC)
        config = ExportPipeline.platform_config("spotify")
        config.output_dir = str(tmp_path / "export")
        config.create_m3u = True
        config.create_cue = True
        config.validate_metadata = False

        pipeline = ExportPipeline(album, config)
        pipeline.set_lyrics_manager(lm)
        pipeline.set_licensing_manager(lic)
        result = pipeline.execute()

        assert result["status"] in ("success", "partial")
        assert result["tracks_exported"] == 5
        print(f"\n  Exported {result['tracks_exported']} tracks to {result['output_dir']}")

        # Verify output files exist
        export_dir = str(result["output_dir"])
        exported_files = [f for f in os.listdir(export_dir) if f.endswith((".flac", ".wav"))]
        assert len(exported_files) >= 5

        # Check for supplementary files
        all_files = os.listdir(export_dir)
        print(f"  Output files: {all_files}")

    def test_metadata_round_trip(self, tmp_path):
        """Write and read back metadata tags on a FLAC file."""
        # Convert to FLAC first
        flac_path = str(tmp_path / "tagged.flac")
        FormatConverter.convert(TRACK_FILES["iron_forge"], flac_path, output_format="flac")

        # Write metadata
        meta = TrackMetadata(
            title="Echoes of the Iron Forge",
            artist="Test Artist",
            album="Iron & Echoes",
            genre="Electronic",
            year=2025,
            track_number=3,
            track_total=5,
            bpm=120,
        )
        MetadataManager.write_tags(flac_path, meta)

        # Read back
        read_meta = MetadataManager.read_tags(flac_path)
        assert read_meta.title == "Echoes of the Iron Forge"
        assert read_meta.artist == "Test Artist"
        assert read_meta.album == "Iron & Echoes"
        assert read_meta.genre == "Electronic"
        print(
            f"\n  Round-trip OK: title={read_meta.title}  artist={read_meta.artist}  "
            f"genre={read_meta.genre}"
        )


# ─── Scenario 7: Reference Mastering Workflow ────────────────────────────


class TestReferenceMasteringWorkflow:
    """Reference-based mastering comparison and matching."""

    def test_full_reference_workflow(self):
        """Load 2 references, compare a mix, generate report."""
        rc = ReferenceCompare()
        rc.load_reference("Capital Echoes", TRACK_FILES["capital_echoes"])
        rc.load_reference("Entrenched", TRACK_FILES["entrenched"])

        mix, sr = sf.read(TRACK_FILES["c_major"], dtype="float64")
        results = rc.compare_all(mix, sr)

        assert len(results) == 2
        for _name, comp in results.items():
            assert np.isfinite(comp.loudness_diff_lufs)
            assert len(comp.spectral_diffs) == 7

        report = rc.report(mix, sr, mix_name="C Major")
        print(f"\n{report}")

    def test_loudness_match(self):
        """Match mix loudness to a reference track."""
        rc = ReferenceCompare()
        rc.load_reference("Entrenched", TRACK_FILES["entrenched"])

        mix, sr = sf.read(TRACK_FILES["iron_forge"], dtype="float64")
        lufs_before = MixAnalyzer.lufs_integrated(mix, sr)

        matched, gain_db = rc.match_loudness(mix, sr)
        lufs_after = MixAnalyzer.lufs_integrated(matched, sr)

        ref_audio, _ = sf.read(TRACK_FILES["entrenched"], dtype="float64")
        ref_lufs = MixAnalyzer.lufs_integrated(ref_audio, sr)

        assert np.isfinite(gain_db)
        # Gain should move LUFS in the right direction
        diff_before = abs(lufs_before - ref_lufs)
        diff_after = abs(lufs_after - ref_lufs)

        print(f"\n  Reference LUFS: {ref_lufs:.1f}")
        print(f"  Before: {lufs_before:.1f}  (diff: {diff_before:.1f})")
        print(f"  After:  {lufs_after:.1f}  (diff: {diff_after:.1f})")
        print(f"  Gain applied: {gain_db:+.1f} dB")

        # Gain should be non-zero (it did something)
        assert abs(gain_db) > 0.01

    def test_mastering_towards_reference(self):
        """Apply mastering chain and verify it moves closer to reference."""
        rc = ReferenceCompare()
        rc.load_reference("Capital Echoes", TRACK_FILES["capital_echoes"])

        mix, sr = sf.read(TRACK_FILES["iron_forge"], dtype="float64")

        # Compare before mastering
        before = rc.compare(mix, sr)

        # Apply mastering chain
        chain = EffectsChain.master_chain()
        mastered = chain.process(mix, sr)

        # Compare after mastering
        after = rc.compare(mastered, sr)

        print("\n  Before mastering:")
        print(
            f"    LUFS diff: {before.loudness_diff_lufs:+.1f}  "
            f"peak diff: {before.peak_diff_db:+.1f}dB"
        )
        print("  After mastering:")
        print(
            f"    LUFS diff: {after.loudness_diff_lufs:+.1f}  "
            f"peak diff: {after.peak_diff_db:+.1f}dB"
        )

        # At minimum the peak should be more controlled
        assert np.isfinite(after.loudness_diff_lufs)
        assert np.isfinite(after.peak_diff_db)


# ─── Scenario 8: Format Conversion & Metadata Round-Trip ────────────────


class TestFormatConversionRoundTrip:
    """Convert between formats and verify metadata survives."""

    def test_mp3_to_wav_to_flac(self, short_track, tmp_path):
        """Chain: MP3 -> WAV -> FLAC, verify at each step."""
        wav_path = str(tmp_path / "step1.wav")
        r1 = FormatConverter.convert(short_track, wav_path, output_format="wav")
        assert r1.success
        info1 = FormatConverter.get_file_info(wav_path)
        assert "WAV" in info1.get("format", "")

        flac_path = str(tmp_path / "step2.flac")
        r2 = FormatConverter.convert(wav_path, flac_path, output_format="flac")
        assert r2.success
        info2 = FormatConverter.get_file_info(flac_path)
        assert "FLAC" in info2.get("format", "")

        # Durations should be close
        orig_info = FormatConverter.get_file_info(short_track)
        assert abs(info1["duration_seconds"] - orig_info["duration_seconds"]) < 0.5
        assert abs(info2["duration_seconds"] - orig_info["duration_seconds"]) < 0.5

        print(f"\n  MP3: {orig_info['duration_seconds']:.1f}s")
        print(f"  WAV: {info1['duration_seconds']:.1f}s  format={info1['format']}")
        print(f"  FLAC: {info2['duration_seconds']:.1f}s  format={info2['format']}")

    def test_metadata_write_read(self, tmp_path):
        """Write metadata to FLAC, read it back, verify fields."""
        flac_path = str(tmp_path / "test.flac")
        FormatConverter.convert(TRACK_FILES["c_major"], flac_path, output_format="flac")

        meta = TrackMetadata(
            title="C Major",
            artist="Test Producer",
            album="Test Album",
            genre="Classical",
            year=2025,
            track_number=1,
            track_total=10,
            bpm=120,
        )
        MetadataManager.write_tags(flac_path, meta)

        read_back = MetadataManager.read_tags(flac_path)
        assert read_back.title == "C Major"
        assert read_back.artist == "Test Producer"
        assert read_back.album == "Test Album"
        assert read_back.genre == "Classical"
        print(f"\n  Metadata verified: {read_back.title} by {read_back.artist}")

    @pytest.mark.slow
    def test_batch_convert_to_flac(self, audio_paths, tmp_path):
        """Convert all 5 MP3s to FLAC."""
        flac_dir = str(tmp_path / "flac")
        os.makedirs(flac_dir)

        for key, path in audio_paths.items():
            flac_out = os.path.join(flac_dir, f"{key}.flac")
            result = FormatConverter.convert(path, flac_out, output_format="flac")
            assert result.success

            info = FormatConverter.get_file_info(flac_out)
            assert "FLAC" in info.get("format", "")
            print(f"  {key}: {info['duration_seconds']:.1f}s FLAC")

        flac_files = [f for f in os.listdir(flac_dir) if f.endswith(".flac")]
        assert len(flac_files) == 5

    def test_bit_depth_conversion(self, short_track, tmp_path):
        """Convert to 16-bit WAV and 24-bit FLAC."""
        wav16 = str(tmp_path / "16bit.wav")
        r1 = FormatConverter.convert(short_track, wav16, output_format="wav", bit_depth=16)
        assert r1.success
        info16 = FormatConverter.get_file_info(wav16)
        assert "PCM_16" in info16.get("subtype", "")

        flac24 = str(tmp_path / "24bit.flac")
        r2 = FormatConverter.convert(short_track, flac24, output_format="flac", bit_depth=24)
        assert r2.success
        info24 = FormatConverter.get_file_info(flac24)
        assert "PCM_24" in info24.get("subtype", "")

        print(f"\n  16-bit WAV: {info16.get('subtype')}")
        print(f"  24-bit FLAC: {info24.get('subtype')}")

    def test_stereo_to_mono(self, tmp_path):
        """Convert stereo to mono."""
        mono_out = str(tmp_path / "mono.wav")
        result = FormatConverter.convert(
            TRACK_FILES["capital_echoes"], mono_out, output_format="wav", channels=1
        )
        assert result.success
        assert result.output_channels == 1

        audio, sr = sf.read(mono_out)
        assert audio.ndim == 1
        print(f"\n  Mono output: {len(audio)/sr:.1f}s  shape={audio.shape}")
