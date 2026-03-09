"""MIDI Transform - Note manipulation and transformation operations.

Provides quantization, humanization, transposition, velocity curves,
time stretching, and other MIDI note transformations.
"""

import math
import random
from collections.abc import Callable

from midi_tools.midi_file_utils import NoteEvent
from midi_tools.scale_library import Scale


class MidiTransform:
    """Collection of MIDI transformation operations.

    All methods accept and return lists of NoteEvent objects,
    making them composable as a processing pipeline.
    """

    # ─── Pitch Operations ───

    @staticmethod
    def transpose(notes: list[NoteEvent], semitones: int) -> list[NoteEvent]:
        """Transpose all notes by semitones."""
        result = []
        for n in notes:
            new_pitch = n.pitch + semitones
            if 0 <= new_pitch <= 127:
                result.append(
                    NoteEvent(
                        new_pitch, n.velocity, n.start_tick, n.duration_ticks, n.channel, n.track
                    )
                )
        return result

    @staticmethod
    def scale_quantize(
        notes: list[NoteEvent], scale: Scale, root: int = 0, direction: str = "nearest"
    ) -> list[NoteEvent]:
        """Snap all notes to the nearest note in a scale.

        Args:
            notes: Input notes
            scale: Target scale
            root: Root pitch class (0-11)
            direction: "nearest", "up", or "down"
        """
        return [
            NoteEvent(
                scale.quantize_note(n.pitch, root, direction),
                n.velocity,
                n.start_tick,
                n.duration_ticks,
                n.channel,
                n.track,
            )
            for n in notes
        ]

    @staticmethod
    def octave_shift(notes: list[NoteEvent], octaves: int) -> list[NoteEvent]:
        """Shift all notes by the given number of octaves."""
        return MidiTransform.transpose(notes, octaves * 12)

    @staticmethod
    def invert(notes: list[NoteEvent], axis: int | None = None) -> list[NoteEvent]:
        """Invert notes around an axis pitch.

        Args:
            axis: MIDI note to invert around. If None, uses the midpoint.
        """
        if not notes:
            return []
        if axis is None:
            pitches = [n.pitch for n in notes]
            axis = (min(pitches) + max(pitches)) // 2

        return [
            NoteEvent(
                max(0, min(127, 2 * axis - n.pitch)),
                n.velocity,
                n.start_tick,
                n.duration_ticks,
                n.channel,
                n.track,
            )
            for n in notes
        ]

    @staticmethod
    def retrograde(notes: list[NoteEvent]) -> list[NoteEvent]:
        """Reverse the order of notes (retrograde)."""
        if not notes:
            return []
        max_end = max(n.start_tick + n.duration_ticks for n in notes)
        return [
            NoteEvent(
                n.pitch,
                n.velocity,
                max_end - n.start_tick - n.duration_ticks,
                n.duration_ticks,
                n.channel,
                n.track,
            )
            for n in notes
        ]

    # ─── Timing Operations ───

    @staticmethod
    def quantize(notes: list[NoteEvent], grid_ticks: int, strength: float = 1.0) -> list[NoteEvent]:
        """Quantize note start times to a grid.

        Args:
            notes: Input notes
            grid_ticks: Grid size in ticks
            strength: Quantize strength 0.0-1.0 (1.0 = full snap, 0.5 = half way)
        """
        result = []
        for n in notes:
            quantized = round(n.start_tick / grid_ticks) * grid_ticks
            new_start = round(n.start_tick + (quantized - n.start_tick) * strength)
            result.append(
                NoteEvent(n.pitch, n.velocity, new_start, n.duration_ticks, n.channel, n.track)
            )
        return result

    @staticmethod
    def quantize_duration(
        notes: list[NoteEvent], grid_ticks: int, strength: float = 1.0
    ) -> list[NoteEvent]:
        """Quantize note durations to a grid."""
        result = []
        for n in notes:
            quantized_dur = max(grid_ticks, round(n.duration_ticks / grid_ticks) * grid_ticks)
            new_dur = round(n.duration_ticks + (quantized_dur - n.duration_ticks) * strength)
            result.append(
                NoteEvent(n.pitch, n.velocity, n.start_tick, max(1, new_dur), n.channel, n.track)
            )
        return result

    @staticmethod
    def humanize(
        notes: list[NoteEvent],
        timing_range: int = 10,
        velocity_range: int = 10,
        duration_range: int = 5,
    ) -> list[NoteEvent]:
        """Add human-like randomization to timing, velocity, and duration.

        Args:
            timing_range: Max timing offset in ticks
            velocity_range: Max velocity variation
            duration_range: Max duration variation in ticks
        """
        result = []
        for n in notes:
            new_start = max(0, n.start_tick + random.randint(-timing_range, timing_range))
            new_vel = max(1, min(127, n.velocity + random.randint(-velocity_range, velocity_range)))
            new_dur = max(1, n.duration_ticks + random.randint(-duration_range, duration_range))
            result.append(NoteEvent(n.pitch, new_vel, new_start, new_dur, n.channel, n.track))
        return result

    @staticmethod
    def swing(
        notes: list[NoteEvent], amount: float = 0.3, grid_ticks: int = 240
    ) -> list[NoteEvent]:
        """Apply swing to notes on off-beat positions.

        Args:
            amount: Swing amount 0.0-1.0
            grid_ticks: Base grid size (half-beat typically)
        """
        result = []
        for n in notes:
            # Check if note is on an off-beat position
            grid_pos = n.start_tick / grid_ticks
            is_offbeat = abs(grid_pos - round(grid_pos)) < 0.1 and round(grid_pos) % 2 == 1
            offset = int(grid_ticks * amount * 0.5) if is_offbeat else 0
            result.append(
                NoteEvent(
                    n.pitch, n.velocity, n.start_tick + offset, n.duration_ticks, n.channel, n.track
                )
            )
        return result

    @staticmethod
    def time_stretch(notes: list[NoteEvent], factor: float) -> list[NoteEvent]:
        """Stretch or compress timing by a factor.

        Args:
            factor: Time multiplier (2.0 = twice as long, 0.5 = half)
        """
        return [
            NoteEvent(
                n.pitch,
                n.velocity,
                round(n.start_tick * factor),
                round(n.duration_ticks * factor),
                n.channel,
                n.track,
            )
            for n in notes
        ]

    @staticmethod
    def shift_time(notes: list[NoteEvent], ticks: int) -> list[NoteEvent]:
        """Shift all notes forward or backward in time."""
        return [
            NoteEvent(
                n.pitch,
                n.velocity,
                max(0, n.start_tick + ticks),
                n.duration_ticks,
                n.channel,
                n.track,
            )
            for n in notes
        ]

    @staticmethod
    def legato(notes: list[NoteEvent]) -> list[NoteEvent]:
        """Extend each note to reach the start of the next note (legato)."""
        if not notes:
            return []
        sorted_notes = sorted(notes, key=lambda n: (n.channel, n.start_tick))
        result = []

        for i, n in enumerate(sorted_notes):
            # Find next note on the same channel
            next_start = None
            for j in range(i + 1, len(sorted_notes)):
                if sorted_notes[j].channel == n.channel:
                    next_start = sorted_notes[j].start_tick
                    break

            if next_start is not None and next_start > n.start_tick:
                new_dur = next_start - n.start_tick
            else:
                new_dur = n.duration_ticks

            result.append(NoteEvent(n.pitch, n.velocity, n.start_tick, new_dur, n.channel, n.track))
        return result

    @staticmethod
    def staccato(notes: list[NoteEvent], fraction: float = 0.25) -> list[NoteEvent]:
        """Shorten all notes to a fraction of their original duration."""
        return [
            NoteEvent(
                n.pitch,
                n.velocity,
                n.start_tick,
                max(1, round(n.duration_ticks * fraction)),
                n.channel,
                n.track,
            )
            for n in notes
        ]

    # ─── Velocity Operations ───

    @staticmethod
    def velocity_scale(notes: list[NoteEvent], factor: float) -> list[NoteEvent]:
        """Scale all velocities by a factor."""
        return [
            NoteEvent(
                n.pitch,
                max(1, min(127, round(n.velocity * factor))),
                n.start_tick,
                n.duration_ticks,
                n.channel,
                n.track,
            )
            for n in notes
        ]

    @staticmethod
    def velocity_curve(
        notes: list[NoteEvent], curve: str = "linear", start: int = 60, end: int = 120
    ) -> list[NoteEvent]:
        """Apply a velocity curve across the time span of notes.

        Args:
            curve: "linear", "exponential", "logarithmic", "sine", "inverse"
            start: Starting velocity
            end: Ending velocity
        """
        if not notes:
            return []

        min_tick = min(n.start_tick for n in notes)
        max_tick = max(n.start_tick for n in notes)
        span = max_tick - min_tick

        curves: dict[str, Callable[[float], float]] = {
            "linear": lambda t: t,
            "exponential": lambda t: t**2,
            "logarithmic": lambda t: math.log(1 + t * 9) / math.log(10),
            "sine": lambda t: 0.5 - 0.5 * math.cos(math.pi * t),
            "inverse": lambda t: 1.0 - t,
        }

        curve_fn = curves.get(curve, curves["linear"])

        result = []
        for n in notes:
            t = (n.start_tick - min_tick) / span if span > 0 else 0
            vel = start + (end - start) * curve_fn(t)
            new_vel = max(1, min(127, round(vel)))
            result.append(
                NoteEvent(n.pitch, new_vel, n.start_tick, n.duration_ticks, n.channel, n.track)
            )
        return result

    @staticmethod
    def velocity_compress(
        notes: list[NoteEvent], target_min: int = 60, target_max: int = 100
    ) -> list[NoteEvent]:
        """Compress velocity range to a target range."""
        if not notes:
            return []

        velocities = [n.velocity for n in notes]
        src_min, src_max = min(velocities), max(velocities)
        src_range = src_max - src_min

        result = []
        for n in notes:
            if src_range > 0:
                normalized = (n.velocity - src_min) / src_range
                new_vel = round(target_min + normalized * (target_max - target_min))
            else:
                new_vel = (target_min + target_max) // 2
            result.append(
                NoteEvent(
                    n.pitch,
                    max(1, min(127, new_vel)),
                    n.start_tick,
                    n.duration_ticks,
                    n.channel,
                    n.track,
                )
            )
        return result

    @staticmethod
    def accent_pattern(
        notes: list[NoteEvent], pattern: list[float], grid_ticks: int = 480
    ) -> list[NoteEvent]:
        """Apply a repeating accent pattern to notes.

        Args:
            pattern: Velocity multipliers per beat (e.g., [1.2, 0.7, 1.0, 0.7])
            grid_ticks: Ticks per pattern step
        """
        result = []
        for n in notes:
            step = (n.start_tick // grid_ticks) % len(pattern)
            new_vel = max(1, min(127, round(n.velocity * pattern[step])))
            result.append(
                NoteEvent(n.pitch, new_vel, n.start_tick, n.duration_ticks, n.channel, n.track)
            )
        return result

    # ─── Filtering ───

    @staticmethod
    def filter_by_velocity(
        notes: list[NoteEvent], min_vel: int = 0, max_vel: int = 127
    ) -> list[NoteEvent]:
        """Keep only notes within a velocity range."""
        return [n for n in notes if min_vel <= n.velocity <= max_vel]

    @staticmethod
    def filter_by_pitch(
        notes: list[NoteEvent], min_pitch: int = 0, max_pitch: int = 127
    ) -> list[NoteEvent]:
        """Keep only notes within a pitch range."""
        return [n for n in notes if min_pitch <= n.pitch <= max_pitch]

    @staticmethod
    def filter_by_duration(
        notes: list[NoteEvent], min_ticks: int = 0, max_ticks: int = 999999
    ) -> list[NoteEvent]:
        """Keep only notes within a duration range."""
        return [n for n in notes if min_ticks <= n.duration_ticks <= max_ticks]

    @staticmethod
    def remove_duplicates(notes: list[NoteEvent], tolerance_ticks: int = 5) -> list[NoteEvent]:
        """Remove duplicate notes (same pitch, close timing)."""
        if not notes:
            return []
        sorted_notes = sorted(notes, key=lambda n: (n.pitch, n.start_tick))
        result = [sorted_notes[0]]
        for n in sorted_notes[1:]:
            prev = result[-1]
            if n.pitch != prev.pitch or abs(n.start_tick - prev.start_tick) > tolerance_ticks:
                result.append(n)
        return result

    # ─── Generation / Mutation ───

    @staticmethod
    def add_octave_doubles(
        notes: list[NoteEvent], octave_offset: int = 1, velocity_factor: float = 0.7
    ) -> list[NoteEvent]:
        """Add octave-doubled notes."""
        result = list(notes)
        for n in notes:
            new_pitch = n.pitch + octave_offset * 12
            if 0 <= new_pitch <= 127:
                new_vel = max(1, min(127, round(n.velocity * velocity_factor)))
                result.append(
                    NoteEvent(
                        new_pitch, new_vel, n.start_tick, n.duration_ticks, n.channel, n.track
                    )
                )
        return result

    @staticmethod
    def create_echo(
        notes: list[NoteEvent], delay_ticks: int = 240, repeats: int = 3, decay: float = 0.6
    ) -> list[NoteEvent]:
        """Create echo/delay effect by repeating notes with decay.

        Args:
            delay_ticks: Delay between echoes
            repeats: Number of echoes
            decay: Velocity decay per echo (0.0-1.0)
        """
        result = list(notes)
        for rep in range(1, repeats + 1):
            for n in notes:
                vel = max(1, min(127, round(n.velocity * (decay**rep))))
                result.append(
                    NoteEvent(
                        n.pitch,
                        vel,
                        n.start_tick + delay_ticks * rep,
                        n.duration_ticks,
                        n.channel,
                        n.track,
                    )
                )
        return result

    # ─── Pipeline Helper ───

    @staticmethod
    def pipeline(
        notes: list[NoteEvent], *transforms: Callable[[list[NoteEvent]], list[NoteEvent]]
    ) -> list[NoteEvent]:
        """Apply a sequence of transforms in order.

        Example:
            result = MidiTransform.pipeline(
                notes,
                lambda n: MidiTransform.transpose(n, 5),
                lambda n: MidiTransform.quantize(n, 480),
                lambda n: MidiTransform.humanize(n, timing_range=5),
            )
        """
        for transform in transforms:
            notes = transform(notes)
        return notes
