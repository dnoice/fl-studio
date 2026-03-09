"""Arpeggiator - Pattern-based arpeggiator engine.

Generates arpeggiated note sequences from chords with configurable
patterns, octave ranges, gate lengths, swing, and velocity shaping.
"""

import random
from dataclasses import dataclass, field
from enum import Enum


class ArpDirection(Enum):
    UP = "up"
    DOWN = "down"
    UP_DOWN = "up_down"
    DOWN_UP = "down_up"
    RANDOM = "random"
    AS_PLAYED = "as_played"
    CONVERGE = "converge"  # Outside notes move inward
    DIVERGE = "diverge"  # Inside notes move outward
    THUMB = "thumb"  # Lowest note repeats between each


@dataclass
class ArpNote:
    """A single note in an arpeggio sequence."""

    pitch: int  # MIDI note number
    time: float  # Time in beats from start
    duration: float  # Duration in beats
    velocity: float  # 0.0 - 1.0


@dataclass
class ArpPattern:
    """A complete arpeggio pattern with all parameters."""

    notes: list[ArpNote] = field(default_factory=list)
    total_beats: float = 0.0

    def transpose(self, semitones: int) -> "ArpPattern":
        """Return transposed pattern."""
        return ArpPattern(
            notes=[
                ArpNote(n.pitch + semitones, n.time, n.duration, n.velocity)
                for n in self.notes
                if 0 <= n.pitch + semitones <= 127
            ],
            total_beats=self.total_beats,
        )

    def scale_time(self, factor: float) -> "ArpPattern":
        """Scale all timing by a factor."""
        return ArpPattern(
            notes=[
                ArpNote(n.pitch, n.time * factor, n.duration * factor, n.velocity)
                for n in self.notes
            ],
            total_beats=self.total_beats * factor,
        )


class Arpeggiator:
    """Pattern-based arpeggiator engine."""

    def __init__(
        self,
        direction: ArpDirection = ArpDirection.UP,
        octaves: int = 1,
        rate: float = 0.25,  # Note rate in beats (0.25 = 16th notes)
        gate: float = 0.8,  # Gate length as fraction of rate
        swing: float = 0.0,  # Swing amount 0.0-1.0
        velocity_start: float = 0.8,
        velocity_end: float = 0.8,
        velocity_curve: str = "linear",  # linear, exponential, sine, random
        repeats: int = 1,  # Pattern repetitions
        steps: int | None = None,  # Total steps (overrides chord length * octaves)
        tie_repeats: bool = False,
        accent_pattern: list[float] | None = None,  # Velocity multiplier per step
    ):
        self.direction = direction
        self.octaves = octaves
        self.rate = rate
        self.gate = gate
        self.swing = swing
        self.velocity_start = velocity_start
        self.velocity_end = velocity_end
        self.velocity_curve = velocity_curve
        self.repeats = repeats
        self.steps = steps
        self.tie_repeats = tie_repeats
        self.accent_pattern = accent_pattern

    def _build_note_sequence(self, input_notes: list[int]) -> list[int]:
        """Build the ordered pitch sequence based on direction and octaves."""
        if not input_notes:
            return []

        base_sorted = sorted(input_notes)
        list(reversed(base_sorted))

        # Expand across octaves
        expanded_up = []
        for oct in range(self.octaves):
            for note in base_sorted:
                n = note + oct * 12
                if 0 <= n <= 127:
                    expanded_up.append(n)

        expanded_down = list(reversed(expanded_up))

        if self.direction == ArpDirection.UP:
            return expanded_up

        elif self.direction == ArpDirection.DOWN:
            return expanded_down

        elif self.direction == ArpDirection.UP_DOWN:
            if len(expanded_up) <= 2:
                return expanded_up
            return expanded_up + expanded_down[1:-1]

        elif self.direction == ArpDirection.DOWN_UP:
            if len(expanded_down) <= 2:
                return expanded_down
            return expanded_down + expanded_up[1:-1]

        elif self.direction == ArpDirection.RANDOM:
            seq = list(expanded_up)
            random.shuffle(seq)
            return seq

        elif self.direction == ArpDirection.AS_PLAYED:
            # Maintain input order, expand octaves
            expanded = []
            for oct in range(self.octaves):
                for note in input_notes:
                    n = note + oct * 12
                    if 0 <= n <= 127:
                        expanded.append(n)
            return expanded

        elif self.direction == ArpDirection.CONVERGE:
            result = []
            notes = list(expanded_up)
            left, right = 0, len(notes) - 1
            while left <= right:
                result.append(notes[left])
                if left != right:
                    result.append(notes[right])
                left += 1
                right -= 1
            return result

        elif self.direction == ArpDirection.DIVERGE:
            notes = list(expanded_up)
            mid = len(notes) // 2
            result = []
            for i in range(mid + 1):
                if mid - i >= 0:
                    result.append(notes[mid - i])
                if mid + i < len(notes) and i != 0:
                    result.append(notes[mid + i])
            return result

        elif self.direction == ArpDirection.THUMB:
            result = []
            thumb = expanded_up[0] if expanded_up else 0
            for note in expanded_up[1:]:
                result.append(thumb)
                result.append(note)
            return result

        return expanded_up

    def _get_velocity(self, step: int, total_steps: int) -> float:
        """Calculate velocity for a given step."""
        t = 0.0 if total_steps <= 1 else step / (total_steps - 1)

        if self.velocity_curve == "linear":
            vel = self.velocity_start + (self.velocity_end - self.velocity_start) * t
        elif self.velocity_curve == "exponential":
            vel = self.velocity_start + (self.velocity_end - self.velocity_start) * (t**2)
        elif self.velocity_curve == "sine":
            import math

            vel = self.velocity_start + (self.velocity_end - self.velocity_start) * (
                0.5 - 0.5 * math.cos(math.pi * t)
            )
        elif self.velocity_curve == "random":
            vel = random.uniform(
                min(self.velocity_start, self.velocity_end),
                max(self.velocity_start, self.velocity_end),
            )
        else:
            vel = self.velocity_start

        # Apply accent pattern
        if self.accent_pattern:
            accent_idx = step % len(self.accent_pattern)
            vel *= self.accent_pattern[accent_idx]

        return max(0.0, min(1.0, vel))

    def _get_swing_offset(self, step: int) -> float:
        """Calculate swing offset for a step."""
        if self.swing <= 0 or step % 2 == 0:
            return 0.0
        return self.rate * self.swing * 0.5

    def generate(self, input_notes: list[int], start_beat: float = 0.0) -> ArpPattern:
        """Generate an arpeggio pattern from input notes.

        Args:
            input_notes: MIDI note numbers to arpeggiate
            start_beat: Starting position in beats

        Returns:
            ArpPattern with all generated notes.
        """
        if not input_notes:
            return ArpPattern()

        sequence = self._build_note_sequence(input_notes)
        if not sequence:
            return ArpPattern()

        total_steps = self.steps if self.steps else len(sequence)
        result_notes: list[ArpNote] = []
        time = start_beat

        for rep in range(self.repeats):
            for step in range(total_steps):
                seq_idx = step % len(sequence)
                pitch = sequence[seq_idx]
                swing_offset = self._get_swing_offset(step)
                velocity = self._get_velocity(
                    step + rep * total_steps,
                    total_steps * self.repeats,
                )
                duration = self.rate * self.gate

                # For tied repeats, extend previous note instead of creating new
                if (
                    self.tie_repeats
                    and rep > 0
                    and step == 0
                    and result_notes
                    and result_notes[-1].pitch == pitch
                ):
                    result_notes[-1].duration += self.rate
                    time += self.rate + swing_offset
                    continue

                result_notes.append(
                    ArpNote(
                        pitch=pitch,
                        time=round(time + swing_offset, 6),
                        duration=round(duration, 6),
                        velocity=round(velocity, 4),
                    )
                )
                time += self.rate

        return ArpPattern(notes=result_notes, total_beats=round(time - start_beat, 6))

    # ─── Preset Patterns ───

    @classmethod
    def preset_sixteenth_up(cls, octaves: int = 1) -> "Arpeggiator":
        """Classic 16th note upward arpeggio."""
        return cls(ArpDirection.UP, octaves=octaves, rate=0.25)

    @classmethod
    def preset_eighth_up_down(cls, octaves: int = 2) -> "Arpeggiator":
        """8th note up-down bounce."""
        return cls(ArpDirection.UP_DOWN, octaves=octaves, rate=0.5)

    @classmethod
    def preset_triplet(cls, octaves: int = 1) -> "Arpeggiator":
        """Triplet feel arpeggio."""
        return cls(ArpDirection.UP, octaves=octaves, rate=1 / 3)

    @classmethod
    def preset_trance_gate(cls, octaves: int = 2) -> "Arpeggiator":
        """Trance-style gated arpeggio with short gate."""
        return cls(
            ArpDirection.UP,
            octaves=octaves,
            rate=0.25,
            gate=0.5,
            velocity_start=1.0,
            velocity_end=0.6,
            velocity_curve="sine",
            accent_pattern=[1.0, 0.6, 0.8, 0.6],
        )

    @classmethod
    def preset_edm_pluck(cls, octaves: int = 2) -> "Arpeggiator":
        """EDM pluck-style with swing."""
        return cls(
            ArpDirection.UP_DOWN,
            octaves=octaves,
            rate=0.25,
            gate=0.4,
            swing=0.3,
            velocity_start=0.9,
            velocity_end=0.7,
        )

    @classmethod
    def preset_random_ambient(cls, octaves: int = 3) -> "Arpeggiator":
        """Ambient random arpeggio across wide range."""
        return cls(
            ArpDirection.RANDOM, octaves=octaves, rate=0.5, gate=1.5, velocity_curve="random"
        )
