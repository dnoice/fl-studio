"""MIDI Analyzer - Statistical analysis and visualization of MIDI files.

Analyzes note distribution, velocity patterns, timing, rhythm density,
and provides key/scale detection from MIDI data.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import mido

from midi_tools.midi_file_utils import MidiFileUtils, NoteEvent
from midi_tools.scale_library import (
    NOTE_NAMES,
    ScaleLibrary,
    midi_to_pitch_class,
)


@dataclass
class NoteDistribution:
    """Pitch class distribution analysis."""

    counts: dict[int, int] = field(default_factory=dict)  # pitch_class -> count
    total: int = 0

    def get_percentages(self) -> dict[str, float]:
        """Get note name -> percentage mapping."""
        if self.total == 0:
            return {}
        return {
            NOTE_NAMES[pc]: round(count / self.total * 100, 1)
            for pc, count in sorted(self.counts.items())
        }

    def top_notes(self, n: int = 5) -> list[tuple[str, int]]:
        """Get the N most common pitch classes."""
        sorted_pcs = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        return [(NOTE_NAMES[pc], count) for pc, count in sorted_pcs[:n]]

    def as_histogram(self, width: int = 40) -> str:
        """Text-based histogram of pitch class distribution."""
        if not self.counts:
            return "No notes"
        max_count = max(self.counts.values())
        lines = []
        for pc in range(12):
            count = self.counts.get(pc, 0)
            bar_len = int(count / max_count * width) if max_count > 0 else 0
            bar = "#" * bar_len
            lines.append(f"{NOTE_NAMES[pc]:>2} | {bar} ({count})")
        return "\n".join(lines)


@dataclass
class VelocityStats:
    """Velocity distribution statistics."""

    min: int = 0
    max: int = 0
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    histogram: dict[str, int] = field(default_factory=dict)  # range_label -> count


@dataclass
class TimingStats:
    """Timing and rhythm analysis."""

    total_beats: float = 0.0
    total_notes: int = 0
    notes_per_beat: float = 0.0
    avg_note_duration_beats: float = 0.0
    shortest_note_beats: float = 0.0
    longest_note_beats: float = 0.0
    density_per_bar: list[float] = field(default_factory=list)


@dataclass
class MidiAnalysis:
    """Complete analysis results for a MIDI file."""

    file_info: dict = field(default_factory=dict)
    note_distribution: NoteDistribution = field(default_factory=NoteDistribution)
    velocity_stats: VelocityStats = field(default_factory=VelocityStats)
    timing_stats: TimingStats = field(default_factory=TimingStats)
    detected_key: list[tuple[str, int, float]] = field(default_factory=list)
    octave_distribution: dict[int, int] = field(default_factory=dict)
    interval_distribution: dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a text summary of the analysis."""
        lines = [
            "=== MIDI Analysis ===",
            f"Tracks: {self.file_info.get('tracks', '?')}",
            f"Tempo: {self.file_info.get('tempo_bpm', '?')} BPM",
            f"Duration: {self.timing_stats.total_beats:.1f} beats",
            f"Total Notes: {self.timing_stats.total_notes}",
            f"Notes/Beat: {self.timing_stats.notes_per_beat:.2f}",
            "",
            "--- Pitch Distribution ---",
            self.note_distribution.as_histogram(),
            "",
            "--- Velocity ---",
            f"Range: {self.velocity_stats.min} - {self.velocity_stats.max}",
            f"Mean: {self.velocity_stats.mean:.1f}",
            f"Std Dev: {self.velocity_stats.std_dev:.1f}",
            "",
            "--- Detected Key ---",
        ]

        for name, root, score in self.detected_key[:3]:
            root_name = NOTE_NAMES[root]
            lines.append(f"  {root_name} {name}: {score}%")

        return "\n".join(lines)


class MidiAnalyzer:
    """Analyze MIDI files for musical content and statistics."""

    @staticmethod
    def analyze(source: str | Path | mido.MidiFile) -> MidiAnalysis:
        """Perform complete analysis of a MIDI file.

        Args:
            source: File path or mido.MidiFile object
        """
        midi_file = MidiFileUtils.read(source) if isinstance(source, str | Path) else source

        notes = MidiFileUtils.extract_notes(midi_file)
        info = MidiFileUtils.info(midi_file)
        tpb = midi_file.ticks_per_beat

        analysis = MidiAnalysis(file_info=info)

        if not notes:
            return analysis

        # Note distribution
        analysis.note_distribution = MidiAnalyzer._analyze_pitch_distribution(notes)

        # Velocity stats
        analysis.velocity_stats = MidiAnalyzer._analyze_velocity(notes)

        # Timing stats
        tempo_bpm = info.get("tempo_bpm", 120)
        analysis.timing_stats = MidiAnalyzer._analyze_timing(
            notes,
            tpb,
            float(tempo_bpm),  # type: ignore[arg-type]
        )

        # Key detection
        pitch_classes = [n.pitch for n in notes]
        analysis.detected_key = ScaleLibrary.detect_scale(pitch_classes)

        # Octave distribution
        octave_counts: dict[int, int] = Counter()
        for note in notes:
            octave = max(0, note.pitch // 12 - 1)
            octave_counts[octave] += 1
        analysis.octave_distribution = dict(sorted(octave_counts.items()))

        # Interval distribution (consecutive note intervals)
        analysis.interval_distribution = MidiAnalyzer._analyze_intervals(notes)

        return analysis

    @staticmethod
    def _analyze_pitch_distribution(notes: list[NoteEvent]) -> NoteDistribution:
        """Analyze pitch class distribution."""
        counts: dict[int, int] = Counter()
        for note in notes:
            counts[midi_to_pitch_class(note.pitch)] += 1
        return NoteDistribution(counts=dict(counts), total=len(notes))

    @staticmethod
    def _analyze_velocity(notes: list[NoteEvent]) -> VelocityStats:
        """Analyze velocity distribution."""
        velocities = [n.velocity for n in notes]
        if not velocities:
            return VelocityStats()

        sorted_vel = sorted(velocities)
        n = len(sorted_vel)
        mean = sum(velocities) / n
        median = sorted_vel[n // 2]
        variance = sum((v - mean) ** 2 for v in velocities) / n
        std_dev = variance**0.5

        # Histogram by ranges
        ranges = {
            "pp (1-31)": 0,
            "p (32-63)": 0,
            "mp (64-79)": 0,
            "mf (80-95)": 0,
            "f (96-111)": 0,
            "ff (112-127)": 0,
        }
        for v in velocities:
            if v <= 31:
                ranges["pp (1-31)"] += 1
            elif v <= 63:
                ranges["p (32-63)"] += 1
            elif v <= 79:
                ranges["mp (64-79)"] += 1
            elif v <= 95:
                ranges["mf (80-95)"] += 1
            elif v <= 111:
                ranges["f (96-111)"] += 1
            else:
                ranges["ff (112-127)"] += 1

        return VelocityStats(
            min=min(velocities),
            max=max(velocities),
            mean=mean,
            median=median,
            std_dev=std_dev,
            histogram=ranges,
        )

    @staticmethod
    def _analyze_timing(notes: list[NoteEvent], tpb: int, tempo: float) -> TimingStats:
        """Analyze timing and rhythm."""
        if not notes:
            return TimingStats()

        max_tick = max(n.start_tick + n.duration_ticks for n in notes)
        total_beats = max_tick / tpb
        durations_beats = [n.duration_ticks / tpb for n in notes]

        # Notes per bar (assuming 4/4)
        bar_ticks = tpb * 4
        max_bar = max_tick // bar_ticks + 1
        density = [0.0] * max_bar
        for note in notes:
            bar_idx = note.start_tick // bar_ticks
            if bar_idx < max_bar:
                density[bar_idx] += 1

        return TimingStats(
            total_beats=total_beats,
            total_notes=len(notes),
            notes_per_beat=len(notes) / total_beats if total_beats > 0 else 0,
            avg_note_duration_beats=sum(durations_beats) / len(durations_beats)
            if durations_beats
            else 0.0,
            shortest_note_beats=min(durations_beats) if durations_beats else 0.0,
            longest_note_beats=max(durations_beats) if durations_beats else 0.0,
            density_per_bar=density,
        )

    @staticmethod
    def _analyze_intervals(notes: list[NoteEvent]) -> dict[int, int]:
        """Analyze melodic intervals between consecutive notes."""
        if len(notes) < 2:
            return {}

        sorted_notes = sorted(notes, key=lambda n: (n.start_tick, n.pitch))
        intervals: dict[int, int] = Counter()

        for i in range(1, len(sorted_notes)):
            # Only measure intervals for notes on the same channel
            if sorted_notes[i].channel == sorted_notes[i - 1].channel:
                interval = sorted_notes[i].pitch - sorted_notes[i - 1].pitch
                intervals[interval] += 1

        return dict(sorted(intervals.items()))

    @staticmethod
    def detect_key(source: str | Path | mido.MidiFile) -> list[tuple[str, int, float]]:
        """Quick key detection from a MIDI file.

        Returns:
            Top key candidates as (scale_name, root_pitch_class, confidence_score)
        """
        analysis = MidiAnalyzer.analyze(source)
        return analysis.detected_key

    @staticmethod
    def compare(
        file_a: str | Path | mido.MidiFile, file_b: str | Path | mido.MidiFile
    ) -> dict[str, object]:
        """Compare two MIDI files and return differences."""
        a = MidiAnalyzer.analyze(file_a)
        b = MidiAnalyzer.analyze(file_b)

        return {
            "note_count_diff": b.timing_stats.total_notes - a.timing_stats.total_notes,
            "duration_diff_beats": b.timing_stats.total_beats - a.timing_stats.total_beats,
            "avg_velocity_diff": b.velocity_stats.mean - a.velocity_stats.mean,
            "density_diff": b.timing_stats.notes_per_beat - a.timing_stats.notes_per_beat,
            "key_a": a.detected_key[:1],
            "key_b": b.detected_key[:1],
            "common_pitch_classes": set(a.note_distribution.counts.keys())
            & set(b.note_distribution.counts.keys()),
        }
