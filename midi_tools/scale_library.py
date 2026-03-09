"""Scale Library - Comprehensive musical scale and mode definitions.

Provides all 22+ scales matching FL Studio's built-in scale definitions,
plus interval math, note name utilities, and scale-aware operations.
"""

from dataclasses import dataclass

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NOTE_NAMES_FLAT = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")

# Map note name -> semitone offset from C
_NOTE_MAP = {}
for _i, _n in enumerate(NOTE_NAMES):
    _NOTE_MAP[_n] = _i
for _i, _n in enumerate(NOTE_NAMES_FLAT):
    _NOTE_MAP[_n] = _i
_NOTE_MAP.update(
    {"Db": 1, "Eb": 3, "Fb": 4, "Gb": 6, "Ab": 8, "Bb": 10, "E#": 5, "B#": 0, "Cb": 11}
)


def note_name_to_midi(name: str, octave: int = 4) -> int:
    """Convert note name + octave to MIDI number. C4 = 60."""
    return _NOTE_MAP[name] + (octave + 1) * 12


def midi_to_note_name(midi_num: int, use_flats: bool = False) -> str:
    """Convert MIDI number to note name with octave."""
    names = NOTE_NAMES_FLAT if use_flats else NOTE_NAMES
    return f"{names[midi_num % 12]}{(midi_num // 12) - 1}"


def midi_to_pitch_class(midi_num: int) -> int:
    """Get pitch class (0-11) from MIDI number."""
    return midi_num % 12


@dataclass
class Scale:
    """A musical scale defined by its name, intervals, and optional metadata."""

    name: str
    intervals: tuple[int, ...]  # Semitone intervals from root (e.g., (0, 2, 4, 5, 7, 9, 11))
    category: str = "other"
    alt_names: tuple[str, ...] = ()

    @property
    def degree_count(self) -> int:
        return len(self.intervals)

    def get_notes(self, root: int = 0, octave_start: int = 4, octaves: int = 1) -> list[int]:
        """Get MIDI note numbers for this scale from a given root.

        Args:
            root: Root note as pitch class (0=C, 1=C#, ..., 11=B) or MIDI number
            octave_start: Starting octave (used if root < 12)
            octaves: Number of octaves to generate
        """
        base = note_name_to_midi(NOTE_NAMES[root], octave_start) if root < 12 else root

        notes = []
        for oct in range(octaves):
            for interval in self.intervals:
                note = base + interval + (oct * 12)
                if note <= 127:
                    notes.append(note)
        return notes

    def contains_note(self, note: int, root: int = 0) -> bool:
        """Check if a MIDI note belongs to this scale with the given root."""
        relative = (note - root) % 12
        return relative in self.intervals

    def get_degree(self, note: int, root: int = 0) -> int | None:
        """Get the scale degree (1-based) of a note, or None if not in scale."""
        relative = (note - root) % 12
        if relative in self.intervals:
            return self.intervals.index(relative) + 1
        return None

    def quantize_note(self, note: int, root: int = 0, direction: str = "nearest") -> int:
        """Snap a note to the nearest scale degree.

        Args:
            note: MIDI note number
            root: Root note pitch class
            direction: "nearest", "up", or "down"
        """
        if self.contains_note(note, root):
            return note

        pitch_class = (note - root) % 12
        octave_base = note - pitch_class

        if direction == "up":
            for i in range(1, 12):
                candidate = (pitch_class + i) % 12
                if candidate in self.intervals:
                    result = octave_base + candidate
                    if candidate < pitch_class:
                        result += 12
                    return min(result, 127)

        elif direction == "down":
            for i in range(1, 12):
                candidate = (pitch_class - i) % 12
                if candidate in self.intervals:
                    result = octave_base + candidate
                    if candidate > pitch_class:
                        result -= 12
                    return max(result, 0)

        else:  # nearest
            best = note
            best_dist = 13
            for interval in self.intervals:
                for offset in (-12, 0, 12):
                    candidate = octave_base + root + interval + offset
                    dist = abs(candidate - note)
                    if dist < best_dist:
                        best_dist = dist
                        best = candidate
            return max(0, min(127, best))

        return note  # Fallback: return original note unchanged

    def interval_pattern(self) -> list[int]:
        """Get the step pattern between consecutive degrees (e.g., [2, 2, 1, 2, 2, 2, 1] for major)."""
        steps = []
        for i in range(len(self.intervals) - 1):
            steps.append(self.intervals[i + 1] - self.intervals[i])
        steps.append(12 - self.intervals[-1] + self.intervals[0])
        return steps

    def relative_minor(self, root: int = 0) -> tuple["Scale", int]:
        """Get the relative minor scale and its root (if this is a major-type scale)."""
        if self.degree_count >= 6:
            minor_root = (root + self.intervals[5]) % 12  # 6th degree
            return SCALES["aeolian"], minor_root
        return self, root

    def parallel_scale(self, target_name: str) -> "Scale":
        """Get a parallel scale (same root, different mode)."""
        return SCALES[target_name]

    def __repr__(self) -> str:
        return f"Scale('{self.name}', intervals={self.intervals})"


# ─── Scale Definitions (matching FL Studio's midi.py HARMONICSCALE_* constants) ───

SCALES: dict[str, Scale] = {
    # Major modes
    "major": Scale("Major (Ionian)", (0, 2, 4, 5, 7, 9, 11), "diatonic", ("ionian",)),
    "dorian": Scale("Dorian", (0, 2, 3, 5, 7, 9, 10), "diatonic"),
    "phrygian": Scale("Phrygian", (0, 1, 3, 5, 7, 8, 10), "diatonic"),
    "lydian": Scale("Lydian", (0, 2, 4, 6, 7, 9, 11), "diatonic"),
    "mixolydian": Scale("Mixolydian", (0, 2, 4, 5, 7, 9, 10), "diatonic"),
    "aeolian": Scale(
        "Aeolian (Natural Minor)", (0, 2, 3, 5, 7, 8, 10), "diatonic", ("natural_minor", "minor")
    ),
    "locrian": Scale("Locrian", (0, 1, 3, 5, 6, 8, 10), "diatonic"),
    # Minor variants
    "harmonic_minor": Scale("Harmonic Minor", (0, 2, 3, 5, 7, 8, 11), "minor"),
    "melodic_minor": Scale("Melodic Minor (Ascending)", (0, 2, 3, 5, 7, 9, 11), "minor"),
    # Pentatonic
    "major_pentatonic": Scale("Major Pentatonic", (0, 2, 4, 7, 9), "pentatonic"),
    "minor_pentatonic": Scale("Minor Pentatonic", (0, 3, 5, 7, 10), "pentatonic"),
    # Blues
    "blues": Scale("Blues", (0, 3, 5, 6, 7, 10), "blues"),
    # Symmetric
    "whole_tone": Scale("Whole Tone", (0, 2, 4, 6, 8, 10), "symmetric"),
    "diminished": Scale(
        "Diminished (Half-Whole)", (0, 1, 3, 4, 6, 7, 9, 10), "symmetric", ("octatonic",)
    ),
    # Bebop
    "major_bebop": Scale("Major Bebop", (0, 2, 4, 5, 7, 8, 9, 11), "bebop"),
    "dominant_bebop": Scale("Dominant Bebop", (0, 2, 4, 5, 7, 9, 10, 11), "bebop"),
    # World / Exotic
    "japanese_insen": Scale("Japanese Insen", (0, 1, 5, 7, 10), "world"),
    "arabic": Scale("Arabic", (0, 1, 4, 5, 7, 8, 11), "world", ("double_harmonic",)),
    "enigmatic": Scale("Enigmatic", (0, 1, 4, 6, 8, 10, 11), "exotic"),
    "neapolitan": Scale("Neapolitan Major", (0, 1, 3, 5, 7, 9, 11), "exotic"),
    "neapolitan_minor": Scale("Neapolitan Minor", (0, 1, 3, 5, 7, 8, 11), "exotic"),
    "hungarian_minor": Scale("Hungarian Minor", (0, 2, 3, 6, 7, 8, 11), "exotic", ("gypsy_minor",)),
    # Chromatic
    "chromatic": Scale("Chromatic", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), "chromatic"),
}

# FL Studio constant -> scale key mapping
FL_SCALE_MAP = {
    0: "major",
    1: "harmonic_minor",
    2: "melodic_minor",
    3: "whole_tone",
    4: "diminished",
    5: "major_pentatonic",
    6: "minor_pentatonic",
    7: "japanese_insen",
    8: "major_bebop",
    9: "dominant_bebop",
    10: "blues",
    11: "arabic",
    12: "enigmatic",
    13: "neapolitan",
    14: "neapolitan_minor",
    15: "hungarian_minor",
    16: "dorian",
    17: "phrygian",
    18: "lydian",
    19: "mixolydian",
    20: "aeolian",
    21: "locrian",
    22: "chromatic",
}


class ScaleLibrary:
    """High-level interface for working with scales."""

    @staticmethod
    def get(name: str) -> Scale:
        """Get a scale by name (case-insensitive, supports aliases)."""
        key = name.lower().replace(" ", "_").replace("-", "_")
        if key in SCALES:
            return SCALES[key]
        # Search by alt names
        for scale in SCALES.values():
            if key in (a.lower() for a in scale.alt_names):
                return scale
        raise KeyError(f"Unknown scale: {name}. Available: {list(SCALES.keys())}")

    @staticmethod
    def get_by_fl_index(index: int) -> Scale:
        """Get a scale using FL Studio's HARMONICSCALE_* constant value."""
        if index not in FL_SCALE_MAP:
            raise KeyError(f"Invalid FL Studio scale index: {index} (valid: 0-22)")
        return SCALES[FL_SCALE_MAP[index]]

    @staticmethod
    def list_scales(category: str | None = None) -> list[Scale]:
        """List all available scales, optionally filtered by category."""
        if category:
            return [s for s in SCALES.values() if s.category == category]
        return list(SCALES.values())

    @staticmethod
    def list_categories() -> list[str]:
        """List all scale categories."""
        return sorted(set(s.category for s in SCALES.values()))

    @staticmethod
    def detect_scale(notes: list[int], root: int | None = None) -> list[tuple[str, int, float]]:
        """Detect the most likely scale(s) from a set of MIDI notes.

        Args:
            notes: List of MIDI note numbers
            root: Known root note (pitch class). If None, tries all roots.

        Returns:
            List of (scale_name, root_pitch_class, match_percentage) sorted by match.
        """
        if not notes:
            return []

        pitch_classes = set(n % 12 for n in notes)
        roots = [root] if root is not None else range(12)
        results = []

        for r in roots:
            for name, scale in SCALES.items():
                if name == "chromatic":
                    continue
                scale_pcs = set((interval + r) % 12 for interval in scale.intervals)
                if not pitch_classes:
                    continue
                matches = len(pitch_classes & scale_pcs)
                coverage = matches / len(pitch_classes)
                # Bonus for scales that don't have many extra notes
                efficiency = matches / len(scale_pcs) if scale_pcs else 0
                score = coverage * 0.7 + efficiency * 0.3
                if coverage >= 0.5:
                    results.append((name, r, round(score * 100, 1)))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:10]

    @staticmethod
    def get_chord_scale_tones(scale: Scale, root: int, degree: int) -> list[int]:
        """Get chord tones (1, 3, 5, 7) for a given scale degree.

        Args:
            scale: The scale to use
            root: Root pitch class
            degree: 1-based scale degree

        Returns:
            List of pitch classes for the chord tones.
        """
        intervals = scale.intervals
        n = len(intervals)
        idx = degree - 1
        tones = []
        for step in (0, 2, 4, 6):  # root, 3rd, 5th, 7th
            tone_idx = (idx + step) % n
            octave_offset = (idx + step) // n
            tones.append((root + intervals[tone_idx] + octave_offset * 12) % 12)
        return tones
