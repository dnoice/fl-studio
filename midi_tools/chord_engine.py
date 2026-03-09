"""Chord Engine - Comprehensive chord generation, voicings, and progressions.

Supports triads through 13th chords, inversions, voice leading,
slash chords, and common progression templates.
"""

import random
from dataclasses import dataclass, field

from midi_tools.scale_library import (
    NOTE_NAMES,
    Scale,
    ScaleLibrary,
    note_name_to_midi,
)

# ─── Chord Type Definitions (intervals from root in semitones) ───

CHORD_TYPES: dict[str, tuple[int, ...]] = {
    # Triads
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "diminished": (0, 3, 6),
    "augmented": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "power": (0, 7),
    # Seventh chords
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dom7": (0, 4, 7, 10),
    "dim7": (0, 3, 6, 9),
    "half_dim7": (0, 3, 6, 10),
    "min_maj7": (0, 3, 7, 11),
    "aug7": (0, 4, 8, 10),
    "aug_maj7": (0, 4, 8, 11),
    "dom7sus4": (0, 5, 7, 10),
    # Extended chords
    "add9": (0, 4, 7, 14),
    "min_add9": (0, 3, 7, 14),
    "maj9": (0, 4, 7, 11, 14),
    "min9": (0, 3, 7, 10, 14),
    "dom9": (0, 4, 7, 10, 14),
    "maj11": (0, 4, 7, 11, 14, 17),
    "min11": (0, 3, 7, 10, 14, 17),
    "dom11": (0, 4, 7, 10, 14, 17),
    "maj13": (0, 4, 7, 11, 14, 17, 21),
    "min13": (0, 3, 7, 10, 14, 17, 21),
    "dom13": (0, 4, 7, 10, 14, 17, 21),
    # Sixth chords
    "maj6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
    "maj6_9": (0, 4, 7, 9, 14),
    "min6_9": (0, 3, 7, 9, 14),
    # Altered chords
    "dom7_flat5": (0, 4, 6, 10),
    "dom7_sharp5": (0, 4, 8, 10),
    "dom7_flat9": (0, 4, 7, 10, 13),
    "dom7_sharp9": (0, 4, 7, 10, 15),
    "dom7_sharp11": (0, 4, 7, 10, 14, 18),
    "alt": (0, 4, 6, 10, 13),  # altered dominant
}

# Short name aliases
CHORD_ALIASES = {
    "M": "major",
    "m": "minor",
    "dim": "diminished",
    "aug": "augmented",
    "7": "dom7",
    "M7": "maj7",
    "m7": "min7",
    "o7": "dim7",
    "ø7": "half_dim7",
    "mM7": "min_maj7",
    "+7": "aug7",
    "9": "dom9",
    "M9": "maj9",
    "m9": "min9",
    "11": "dom11",
    "M11": "maj11",
    "m11": "min11",
    "13": "dom13",
    "M13": "maj13",
    "m13": "min13",
    "6": "maj6",
    "m6": "min6",
    "5": "power",
}

# Diatonic chord qualities for major scale degrees (1-7)
MAJOR_DIATONIC_CHORDS = {
    1: "major",
    2: "minor",
    3: "minor",
    4: "major",
    5: "major",
    6: "minor",
    7: "diminished",
}

MAJOR_DIATONIC_7TH = {
    1: "maj7",
    2: "min7",
    3: "min7",
    4: "maj7",
    5: "dom7",
    6: "min7",
    7: "half_dim7",
}

MINOR_DIATONIC_CHORDS = {
    1: "minor",
    2: "diminished",
    3: "major",
    4: "minor",
    5: "minor",
    6: "major",
    7: "major",
}

MINOR_DIATONIC_7TH = {
    1: "min7",
    2: "half_dim7",
    3: "maj7",
    4: "min7",
    5: "min7",
    6: "maj7",
    7: "dom7",
}

# Roman numeral labels
ROMAN_NUMERALS = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}


@dataclass
class Chord:
    """A musical chord with root, type, and specific voicing."""

    root: int  # MIDI note number of root
    chord_type: str
    notes: list[int] = field(default_factory=list)
    inversion: int = 0
    bass_note: int | None = None  # For slash chords

    def __post_init__(self):
        if not self.notes:
            self.notes = self._build_notes()

    def _build_notes(self) -> list[int]:
        """Build MIDI notes from root and chord type."""
        type_key = CHORD_ALIASES.get(self.chord_type, self.chord_type)
        if type_key not in CHORD_TYPES:
            raise ValueError(
                f"Unknown chord type: {self.chord_type}. " f"Available: {list(CHORD_TYPES.keys())}"
            )
        intervals = CHORD_TYPES[type_key]
        notes = [self.root + i for i in intervals]

        # Apply inversion
        for i in range(self.inversion):
            if i < len(notes):
                notes[i] += 12

        notes.sort()

        # Add bass note for slash chords
        if self.bass_note is not None:
            bass = self.bass_note
            while bass >= notes[0]:
                bass -= 12
            notes.insert(0, bass)

        return [n for n in notes if 0 <= n <= 127]

    @property
    def name(self) -> str:
        root_name = NOTE_NAMES[self.root % 12]
        CHORD_ALIASES.get(self.chord_type, self.chord_type)
        suffix = self.chord_type
        inv = f" (inv {self.inversion})" if self.inversion else ""
        bass = f"/{NOTE_NAMES[self.bass_note % 12]}" if self.bass_note is not None else ""
        return f"{root_name}{suffix}{inv}{bass}"

    def transpose(self, semitones: int) -> "Chord":
        """Return a new chord transposed by the given semitones."""
        new_root = self.root + semitones
        new_bass = self.bass_note + semitones if self.bass_note is not None else None
        return Chord(new_root, self.chord_type, inversion=self.inversion, bass_note=new_bass)

    def with_inversion(self, inversion: int) -> "Chord":
        """Return a new chord with the specified inversion."""
        return Chord(self.root, self.chord_type, inversion=inversion, bass_note=self.bass_note)

    def spread(self, octaves: int = 2) -> "Chord":
        """Spread chord tones across multiple octaves for open voicing."""
        type_key = CHORD_ALIASES.get(self.chord_type, self.chord_type)
        intervals = CHORD_TYPES[type_key]
        notes = []
        for i, interval in enumerate(intervals):
            octave_offset = (i % octaves) * 12
            notes.append(self.root + interval + octave_offset)
        notes.sort()
        return Chord(self.root, self.chord_type, notes=[n for n in notes if 0 <= n <= 127])

    def drop2(self) -> "Chord":
        """Apply drop-2 voicing (move 2nd highest note down an octave)."""
        if len(self.notes) < 4:
            return self
        notes = list(self.notes)
        notes.sort()
        notes[-2] -= 12
        notes.sort()
        return Chord(self.root, self.chord_type, notes=[n for n in notes if 0 <= n <= 127])

    def drop3(self) -> "Chord":
        """Apply drop-3 voicing (move 3rd highest note down an octave)."""
        if len(self.notes) < 4:
            return self
        notes = list(self.notes)
        notes.sort()
        notes[-3] -= 12
        notes.sort()
        return Chord(self.root, self.chord_type, notes=[n for n in notes if 0 <= n <= 127])

    def __repr__(self) -> str:
        return f"Chord('{self.name}', notes={self.notes})"


@dataclass
class ChordProgression:
    """A sequence of chords with timing and key context."""

    chords: list[Chord]
    key_root: int = 0  # Pitch class
    scale_name: str = "major"
    beats_per_chord: float = 4.0

    @property
    def name(self) -> str:
        root = NOTE_NAMES[self.key_root]
        return f"{root} {self.scale_name} progression ({len(self.chords)} chords)"

    def transpose(self, semitones: int) -> "ChordProgression":
        """Transpose the entire progression."""
        return ChordProgression(
            chords=[c.transpose(semitones) for c in self.chords],
            key_root=(self.key_root + semitones) % 12,
            scale_name=self.scale_name,
            beats_per_chord=self.beats_per_chord,
        )

    def __repr__(self) -> str:
        chord_names = [c.name for c in self.chords]
        return f"ChordProgression({chord_names})"


class ChordEngine:
    """High-level chord and progression generation engine."""

    @staticmethod
    def build(
        root: int, chord_type: str, inversion: int = 0, bass_note: int | None = None
    ) -> Chord:
        """Build a chord from root note and type."""
        return Chord(root, chord_type, inversion=inversion, bass_note=bass_note)

    @staticmethod
    def from_name(name: str, octave: int = 4) -> Chord:
        """Parse a chord from string notation like 'Cmaj7', 'F#m', 'Bb7'.

        Supports: note name + chord type suffix.
        """
        # Extract root note
        if len(name) >= 2 and name[1] in ("#", "b"):
            root_name = name[:2]
            suffix = name[2:]
        else:
            root_name = name[0]
            suffix = name[1:]

        root = note_name_to_midi(root_name, octave)

        # Map suffix to chord type
        suffix_map = {
            "": "major",
            "m": "minor",
            "min": "minor",
            "dim": "diminished",
            "aug": "augmented",
            "sus2": "sus2",
            "sus4": "sus4",
            "maj7": "maj7",
            "M7": "maj7",
            "m7": "min7",
            "min7": "min7",
            "7": "dom7",
            "dom7": "dom7",
            "dim7": "dim7",
            "o7": "dim7",
            "m7b5": "half_dim7",
            "mM7": "min_maj7",
            "minmaj7": "min_maj7",
            "aug7": "aug7",
            "+7": "aug7",
            "9": "dom9",
            "M9": "maj9",
            "maj9": "maj9",
            "m9": "min9",
            "min9": "min9",
            "11": "dom11",
            "M11": "maj11",
            "m11": "min11",
            "13": "dom13",
            "M13": "maj13",
            "m13": "min13",
            "6": "maj6",
            "m6": "min6",
            "add9": "add9",
            "madd9": "min_add9",
            "5": "power",
        }

        chord_type = suffix_map.get(suffix)
        if chord_type is None:
            raise ValueError(f"Unknown chord suffix: '{suffix}' in '{name}'")

        return Chord(root, chord_type)

    @staticmethod
    def diatonic_chord(
        scale: Scale, root_pitch_class: int, degree: int, octave: int = 4, seventh: bool = False
    ) -> Chord:
        """Build a diatonic chord from a scale degree.

        Args:
            scale: Scale to derive chord quality from
            root_pitch_class: Root note of the key (0-11)
            degree: Scale degree (1-7)
            octave: Octave for the chord root
            seventh: Include 7th or just triad
        """
        intervals = scale.intervals
        n = len(intervals)
        idx = degree - 1

        # Get chord tones by stacking thirds within the scale
        chord_intervals: list[int] = []
        for step in (0, 2, 4) if not seventh else (0, 2, 4, 6):
            tone_idx = (idx + step) % n
            interval = intervals[tone_idx]
            # Handle octave wrapping
            if step > 0 and interval <= intervals[idx]:
                interval += 12
            if step > 2 and interval <= chord_intervals[-1] - root_pitch_class:
                interval += 12
            chord_intervals.append(interval)

        # Calculate semitone distances to determine chord quality
        root_interval = chord_intervals[0]
        relative = [i - root_interval for i in chord_intervals]

        # Determine chord quality from intervals
        third = relative[1] if len(relative) > 1 else 0
        fifth = relative[2] if len(relative) > 2 else 0

        if third == 4 and fifth == 7:
            quality = "major"
        elif third == 3 and fifth == 7:
            quality = "minor"
        elif third == 3 and fifth == 6:
            quality = "diminished"
        elif third == 4 and fifth == 8:
            quality = "augmented"
        else:
            quality = "major"

        if seventh and len(relative) > 3:
            sev = relative[3]
            if quality == "major" and sev == 11:
                quality = "maj7"
            elif quality == "major" and sev == 10:
                quality = "dom7"
            elif quality == "minor" and sev == 10:
                quality = "min7"
            elif quality == "diminished" and sev == 9:
                quality = "dim7"
            elif quality == "diminished" and sev == 10:
                quality = "half_dim7"
            elif quality == "minor" and sev == 11:
                quality = "min_maj7"

        midi_root = note_name_to_midi(NOTE_NAMES[(root_pitch_class + root_interval) % 12], octave)
        return Chord(midi_root, quality)

    @staticmethod
    def progression_from_numerals(
        numerals: list[str],
        key_root: int = 0,
        scale_name: str = "major",
        octave: int = 4,
        seventh: bool = False,
    ) -> ChordProgression:
        """Build a progression from Roman numeral notation.

        Args:
            numerals: List like ['I', 'IV', 'V', 'I'] or ['i', 'iv', 'V', 'i']
                      Lowercase = minor, uppercase = major override.
            key_root: Root pitch class (0=C)
            scale_name: Scale name
            octave: Chord octave
            seventh: Add 7ths to all chords
        """
        scale = ScaleLibrary.get(scale_name)
        numeral_to_degree = {
            "i": 1,
            "ii": 2,
            "iii": 3,
            "iv": 4,
            "v": 5,
            "vi": 6,
            "vii": 7,
        }

        chords = []
        for numeral in numerals:
            clean = numeral.lower().rstrip("o+7")
            degree = numeral_to_degree.get(clean)
            if degree is None:
                raise ValueError(f"Unknown Roman numeral: {numeral}")
            chord = ChordEngine.diatonic_chord(scale, key_root, degree, octave, seventh)
            chords.append(chord)

        return ChordProgression(chords, key_root, scale_name)

    # ─── Common Progression Templates ───

    @staticmethod
    def progression_1_4_5_1(
        key_root: int = 0, octave: int = 4, seventh: bool = False
    ) -> ChordProgression:
        """Classic I-IV-V-I progression."""
        return ChordEngine.progression_from_numerals(
            ["I", "IV", "V", "I"], key_root, "major", octave, seventh
        )

    @staticmethod
    def progression_1_5_6_4(
        key_root: int = 0, octave: int = 4, seventh: bool = False
    ) -> ChordProgression:
        """Pop progression: I-V-vi-IV."""
        return ChordEngine.progression_from_numerals(
            ["I", "V", "vi", "IV"], key_root, "major", octave, seventh
        )

    @staticmethod
    def progression_2_5_1(key_root: int = 0, octave: int = 4) -> ChordProgression:
        """Jazz ii-V-I progression with 7ths."""
        return ChordEngine.progression_from_numerals(
            ["ii", "V", "I"], key_root, "major", octave, seventh=True
        )

    @staticmethod
    def progression_1_6_4_5(
        key_root: int = 0, octave: int = 4, seventh: bool = False
    ) -> ChordProgression:
        """50s/doo-wop: I-vi-IV-V."""
        return ChordEngine.progression_from_numerals(
            ["I", "vi", "IV", "V"], key_root, "major", octave, seventh
        )

    @staticmethod
    def progression_6_4_1_5(
        key_root: int = 0, octave: int = 4, seventh: bool = False
    ) -> ChordProgression:
        """Sad/emotional: vi-IV-I-V."""
        return ChordEngine.progression_from_numerals(
            ["vi", "IV", "I", "V"], key_root, "major", octave, seventh
        )

    @staticmethod
    def progression_1_4_6_5(
        key_root: int = 0, octave: int = 4, seventh: bool = False
    ) -> ChordProgression:
        """Country/folk: I-IV-vi-V."""
        return ChordEngine.progression_from_numerals(
            ["I", "IV", "vi", "V"], key_root, "major", octave, seventh
        )

    @staticmethod
    def progression_minor_1_4_5(
        key_root: int = 0, octave: int = 4, seventh: bool = False
    ) -> ChordProgression:
        """Minor i-iv-v."""
        return ChordEngine.progression_from_numerals(
            ["i", "iv", "v"], key_root, "aeolian", octave, seventh
        )

    @staticmethod
    def progression_minor_1_6_3_7(
        key_root: int = 0, octave: int = 4, seventh: bool = False
    ) -> ChordProgression:
        """Andalusian cadence: i-bVII-bVI-V (in minor)."""
        return ChordEngine.progression_from_numerals(
            ["i", "vii", "vi", "v"], key_root, "aeolian", octave, seventh
        )

    @staticmethod
    def apply_voice_leading(
        progression: ChordProgression, max_movement: int = 4
    ) -> ChordProgression:
        """Minimize voice movement between consecutive chords.

        Adjusts inversions and octaves so each voice moves minimally.

        Args:
            progression: Input progression
            max_movement: Maximum semitone movement per voice
        """
        if len(progression.chords) < 2:
            return progression

        result = [progression.chords[0]]

        for i in range(1, len(progression.chords)):
            prev_notes = sorted(result[-1].notes)
            curr = progression.chords[i]
            type_key = CHORD_ALIASES.get(curr.chord_type, curr.chord_type)
            intervals = CHORD_TYPES[type_key]

            # Try all inversions and find the one with minimum total movement
            best_notes = curr.notes
            best_movement = float("inf")

            for inv in range(len(intervals)):
                test_chord = curr.with_inversion(inv)
                test_notes = sorted(test_chord.notes)

                # Adjust octave to be near previous chord
                while test_notes and prev_notes and test_notes[0] > prev_notes[0] + 6:
                    test_notes = [n - 12 for n in test_notes]
                while test_notes and prev_notes and test_notes[0] < prev_notes[0] - 6:
                    test_notes = [n + 12 for n in test_notes]

                # Calculate total movement
                movement = 0
                for j in range(min(len(prev_notes), len(test_notes))):
                    movement += abs(test_notes[j] - prev_notes[j])

                if movement < best_movement:
                    best_movement = movement
                    best_notes = [n for n in test_notes if 0 <= n <= 127]

            result.append(Chord(curr.root, curr.chord_type, notes=best_notes))

        return ChordProgression(
            result, progression.key_root, progression.scale_name, progression.beats_per_chord
        )

    @staticmethod
    def random_progression(
        key_root: int = 0,
        scale_name: str = "major",
        length: int = 4,
        octave: int = 4,
        seventh: bool = False,
    ) -> ChordProgression:
        """Generate a random diatonic progression."""
        scale = ScaleLibrary.get(scale_name)
        degrees = random.choices(range(1, scale.degree_count + 1), k=length)
        chords = [ChordEngine.diatonic_chord(scale, key_root, d, octave, seventh) for d in degrees]
        return ChordProgression(chords, key_root, scale_name)

    @staticmethod
    def list_chord_types() -> list[str]:
        """List all available chord types."""
        return list(CHORD_TYPES.keys())
