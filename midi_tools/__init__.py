"""FL Studio Toolkit - MIDI Tools

Standalone MIDI utilities for scale/chord generation, arpeggiators,
drum patterns, MIDI file manipulation, analysis, and transformation.

Quick start::

    from midi_tools import ScaleLibrary, ChordEngine, Arpeggiator, MidiFileUtils

    scale = ScaleLibrary.get("C", "major")          # -> Scale
    chord = ChordEngine.build("Cmaj7")              # -> Chord
    arp = Arpeggiator(direction="up").generate(chord) # -> ArpPattern
    mid = MidiFileUtils.create(tempo=120)            # -> mido.MidiFile
"""

from midi_tools._validation import validate_channel, validate_pitch, validate_velocity
from midi_tools.arpeggiator import Arpeggiator, ArpPattern
from midi_tools.chord_engine import Chord, ChordEngine, ChordProgression
from midi_tools.drum_patterns import DrumPattern, DrumPatternLibrary
from midi_tools.midi_analyzer import MidiAnalyzer
from midi_tools.midi_file_utils import MidiFileUtils
from midi_tools.midi_transform import MidiTransform
from midi_tools.scale_library import SCALES, Scale, ScaleLibrary

__all__ = [
    "validate_channel",
    "validate_pitch",
    "validate_velocity",
    "Arpeggiator",
    "ArpPattern",
    "Chord",
    "ChordEngine",
    "ChordProgression",
    "DrumPattern",
    "DrumPatternLibrary",
    "MidiAnalyzer",
    "MidiFileUtils",
    "MidiTransform",
    "SCALES",
    "Scale",
    "ScaleLibrary",
]

__version__ = "0.3.0"
