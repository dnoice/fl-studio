"""Comprehensive tests for the MIDI tools module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from midi_tools.arpeggiator import ArpDirection, Arpeggiator
from midi_tools.chord_engine import ChordEngine
from midi_tools.drum_patterns import DrumPattern, DrumPatternLibrary
from midi_tools.midi_file_utils import NoteEvent
from midi_tools.midi_transform import MidiTransform
from midi_tools.scale_library import SCALES, ScaleLibrary, midi_to_note_name, note_name_to_midi

# ─── Scale Library Tests ───


class TestScaleLibrary:
    def test_note_conversion(self):
        assert note_name_to_midi("C", 4) == 60
        assert note_name_to_midi("A", 4) == 69
        assert midi_to_note_name(60) == "C4"
        assert midi_to_note_name(69) == "A4"

    def test_note_conversion_sharps(self):
        assert note_name_to_midi("C#", 4) == 61
        assert note_name_to_midi("F#", 3) == 54

    def test_scale_notes_major(self):
        major = SCALES["major"]
        notes = major.get_notes(root=0, octave_start=4, octaves=1)
        assert notes == [60, 62, 64, 65, 67, 69, 71]

    def test_scale_notes_minor(self):
        minor = SCALES["aeolian"]  # natural minor = aeolian mode
        notes = minor.get_notes(root=0, octave_start=4, octaves=1)
        assert notes == [60, 62, 63, 65, 67, 68, 70]

    def test_scale_contains(self):
        major = SCALES["major"]
        assert major.contains_note(60, root=0)  # C
        assert major.contains_note(64, root=0)  # E
        assert not major.contains_note(61, root=0)  # C#

    def test_scale_quantize(self):
        major = SCALES["major"]
        assert major.quantize_note(61, root=0) in (60, 62)  # C# -> C or D
        assert major.quantize_note(60, root=0) == 60  # Already in scale

    def test_scale_detection(self):
        c_major_notes = [60, 62, 64, 65, 67, 69, 71]
        results = ScaleLibrary.detect_scale(c_major_notes)
        assert len(results) > 0
        top_name, top_root, top_score = results[0]
        assert top_root == 0 or top_name in ("major", "lydian", "mixolydian")

    def test_all_scales_exist(self):
        assert len(SCALES) >= 22
        for _name, scale in SCALES.items():
            assert len(scale.intervals) >= 2
            assert scale.intervals[0] == 0

    def test_scale_multi_octave(self):
        major = SCALES["major"]
        notes = major.get_notes(root=0, octave_start=4, octaves=2)
        assert len(notes) == 14
        assert notes[-1] == 83  # B5


# ─── Chord Engine Tests ───


class TestChordEngine:
    def test_major_chord(self):
        chord = ChordEngine.build(60, "major")
        assert chord.notes == [60, 64, 67]

    def test_minor_seventh(self):
        chord = ChordEngine.build(60, "min7")
        assert chord.notes == [60, 63, 67, 70]

    def test_chord_inversion(self):
        chord = ChordEngine.build(60, "major", inversion=1)
        assert 72 in chord.notes  # E5 (moved up)

    def test_chord_from_name(self):
        chord = ChordEngine.from_name("Cmaj7")
        assert 60 in chord.notes  # C
        assert 64 in chord.notes  # E
        assert 67 in chord.notes  # G
        assert 71 in chord.notes  # B

    def test_chord_from_name_minor(self):
        chord = ChordEngine.from_name("Am")
        assert 69 in chord.notes  # A
        assert 72 in chord.notes  # C
        assert 76 in chord.notes  # E

    def test_progression(self):
        prog = ChordEngine.progression_1_5_6_4(key_root=0)
        assert len(prog.chords) == 4

    def test_diatonic_chords(self):
        chords = [ChordEngine.diatonic_chord(SCALES["major"], 0, deg) for deg in range(1, 8)]
        assert len(chords) == 7  # 7 diatonic chords
        # First chord should be C major
        assert 60 in chords[0].notes
        assert 64 in chords[0].notes


# ─── Arpeggiator Tests ───


class TestArpeggiator:
    def test_arpeggiator_up(self):
        arp = Arpeggiator(direction=ArpDirection.UP, rate=0.25, octaves=1)
        pattern = arp.generate([60, 64, 67])
        assert len(pattern.notes) == 3
        pitches = [n.pitch for n in pattern.notes]
        assert pitches == [60, 64, 67]

    def test_arpeggiator_down(self):
        arp = Arpeggiator(direction=ArpDirection.DOWN, rate=0.25, octaves=1)
        pattern = arp.generate([60, 64, 67])
        pitches = [n.pitch for n in pattern.notes]
        assert pitches == [67, 64, 60]

    def test_arpeggiator_octaves(self):
        arp = Arpeggiator(direction=ArpDirection.UP, rate=0.25, octaves=2)
        pattern = arp.generate([60, 64, 67])
        assert len(pattern.notes) == 6
        # Second octave should be 12 semitones higher
        pitches = [n.pitch for n in pattern.notes]
        assert pitches[3] == pitches[0] + 12

    def test_arpeggiator_gate(self):
        arp = Arpeggiator(direction=ArpDirection.UP, rate=0.25, octaves=1, gate=0.5)
        pattern = arp.generate([60, 64, 67])
        # Gate 0.5 should make notes shorter
        for note in pattern.notes:
            assert note.duration > 0


# ─── Drum Pattern Tests ───


class TestDrumPatterns:
    def test_drum_patterns_exist(self):
        patterns = DrumPatternLibrary.list_patterns()
        assert len(patterns) >= 15

    def test_drum_pattern_grid(self):
        pattern = DrumPatternLibrary.trap_basic()
        assert pattern.steps == 16
        kick_hits = pattern.get_hits_for("kick")
        assert len(kick_hits) > 0

    def test_drum_pattern_operations(self):
        pattern = DrumPatternLibrary.house_basic()
        doubled = pattern.double()
        assert doubled.steps == 32
        humanized = pattern.humanize()
        assert len(humanized.hits) == len(pattern.hits)

    def test_all_genre_patterns(self):
        """Verify all genre patterns are valid."""
        for name in DrumPatternLibrary.list_patterns():
            method = getattr(DrumPatternLibrary, name, None)
            if callable(method):
                pattern = method()
                assert isinstance(pattern, DrumPattern)
                assert pattern.steps > 0
                assert len(pattern.hits) > 0


# ─── MIDI Transform Tests ───


class TestMidiTransform:
    def _make_notes(self, pitches):
        """Helper to create NoteEvent objects from pitch list."""
        return [NoteEvent(p, 100, i * 480, 240) for i, p in enumerate(pitches)]

    def test_transpose(self):
        notes = self._make_notes([60, 64, 67])
        result = MidiTransform.transpose(notes, 2)
        assert [n.pitch for n in result] == [62, 66, 69]

    def test_transpose_negative(self):
        notes = self._make_notes([60, 64, 67])
        result = MidiTransform.transpose(notes, -12)
        assert [n.pitch for n in result] == [48, 52, 55]

    def test_scale_quantize(self):
        notes = self._make_notes([61, 63, 66])  # C#, D#, F# (out of C major)
        result = MidiTransform.scale_quantize(notes, SCALES["major"], root=0)
        for note in result:
            assert note.pitch % 12 in [0, 2, 4, 5, 7, 9, 11]  # C major scale degrees
