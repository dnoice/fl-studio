"""MIDI File Utilities - Read, write, merge, split, and manipulate MIDI files.

Uses the `mido` library for MIDI file I/O. Provides higher-level operations
for common FL Studio production workflows.
"""

from dataclasses import dataclass
from pathlib import Path

import mido

from midi_tools._validation import validate_channel


@dataclass
class NoteEvent:
    """A note event with timing information."""

    pitch: int
    velocity: int
    start_tick: int
    duration_ticks: int
    channel: int = 0
    track: int = 0

    @property
    def end_tick(self) -> int:
        return self.start_tick + self.duration_ticks


class MidiFileUtils:
    """High-level MIDI file operations."""

    @staticmethod
    def read(filepath: str | Path) -> mido.MidiFile:
        """Read a MIDI file."""
        return mido.MidiFile(str(filepath))

    @staticmethod
    def write(midi_file: mido.MidiFile, filepath: str | Path) -> None:
        """Write a MIDI file."""
        midi_file.save(str(filepath))

    @staticmethod
    def create(ticks_per_beat: int = 480, tempo: int = 120) -> mido.MidiFile:
        """Create a new MIDI file with a tempo track.

        Args:
            ticks_per_beat: Resolution (FL Studio uses 96 by default, 480 is common)
            tempo: BPM
        """
        mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        tempo_track = mido.MidiTrack()
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo)))
        tempo_track.append(mido.MetaMessage("end_of_track"))
        mid.tracks.append(tempo_track)
        return mid

    @staticmethod
    def extract_notes(midi_file: mido.MidiFile) -> list[NoteEvent]:
        """Extract all note events from a MIDI file into NoteEvent objects.

        Skips messages with out-of-range pitch (0-127), velocity (0-127),
        or channel (0-15) values rather than crashing.
        """
        notes = []

        for track_idx, track in enumerate(midi_file.tracks):
            active_notes: dict[tuple[int, int], tuple[int, int]] = {}  # (pitch, ch) -> (tick, vel)
            tick = 0

            for msg in track:
                tick += msg.time

                if not hasattr(msg, "type"):
                    continue

                if msg.type == "note_on" and msg.velocity > 0:
                    if not (
                        0 <= msg.note <= 127 and 0 <= msg.velocity <= 127 and 0 <= msg.channel <= 15
                    ):
                        continue
                    active_notes[(msg.note, msg.channel)] = (tick, msg.velocity)

                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    key = (msg.note, msg.channel)
                    if key in active_notes:
                        start_tick, velocity = active_notes.pop(key)
                        duration = tick - start_tick
                        if duration < 0:
                            continue
                        notes.append(
                            NoteEvent(
                                pitch=msg.note,
                                velocity=velocity,
                                start_tick=start_tick,
                                duration_ticks=duration,
                                channel=msg.channel,
                                track=track_idx,
                            )
                        )

        notes.sort(key=lambda n: (n.start_tick, n.pitch))
        return notes

    @staticmethod
    def notes_to_track(
        notes: list[NoteEvent], channel: int = 0, track_name: str | None = None
    ) -> mido.MidiTrack:
        """Convert NoteEvent objects into a MIDI track.

        Args:
            notes: List of NoteEvent objects
            channel: MIDI channel (0-15)
            track_name: Optional track name

        Raises:
            ValueError: If channel is out of range 0-15
        """
        validate_channel(channel)
        track = mido.MidiTrack()

        if track_name:
            track.append(mido.MetaMessage("track_name", name=track_name))

        # Build time-sorted event list
        events = []
        for note in notes:
            pitch = max(0, min(127, note.pitch))
            vel = max(0, min(127, note.velocity))
            ch = note.channel if 0 <= note.channel <= 15 else channel
            events.append((note.start_tick, "note_on", pitch, vel, ch))
            events.append((note.end_tick, "note_off", pitch, 0, ch))

        events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))

        prev_tick = 0
        for tick, msg_type, pitch, velocity, ch in events:
            delta = tick - prev_tick
            track.append(
                mido.Message(msg_type, note=pitch, velocity=velocity, channel=ch, time=delta)
            )
            prev_tick = tick

        track.append(mido.MetaMessage("end_of_track"))
        return track

    @staticmethod
    def merge_tracks(midi_file: mido.MidiFile) -> mido.MidiTrack:
        """Merge all tracks into a single track."""
        return mido.merge_tracks(midi_file.tracks)

    @staticmethod
    def split_by_channel(midi_file: mido.MidiFile) -> dict[int, mido.MidiFile]:
        """Split a MIDI file into separate files, one per MIDI channel.

        Returns:
            Dict mapping channel number -> MidiFile
        """
        notes = MidiFileUtils.extract_notes(midi_file)
        channels = set(n.channel for n in notes)

        result = {}
        for ch in channels:
            ch_notes = [n for n in notes if n.channel == ch]
            new_mid = MidiFileUtils.create(midi_file.ticks_per_beat)
            track = MidiFileUtils.notes_to_track(ch_notes, ch, f"Channel {ch}")
            new_mid.tracks.append(track)
            result[ch] = new_mid

        return result

    @staticmethod
    def split_by_note_range(
        midi_file: mido.MidiFile, split_note: int = 60
    ) -> tuple[mido.MidiFile, mido.MidiFile]:
        """Split a MIDI file into two based on a note number threshold.

        Useful for splitting bass from melody, or left/right hand piano parts.

        Args:
            midi_file: Input MIDI file
            split_note: Notes below this go to the first file, at or above to the second

        Returns:
            Tuple of (low_notes_file, high_notes_file)
        """
        notes = MidiFileUtils.extract_notes(midi_file)
        low = [n for n in notes if n.pitch < split_note]
        high = [n for n in notes if n.pitch >= split_note]

        tpb = midi_file.ticks_per_beat
        low_mid = MidiFileUtils.create(tpb)
        high_mid = MidiFileUtils.create(tpb)

        if low:
            low_mid.tracks.append(MidiFileUtils.notes_to_track(low, track_name="Low"))
        if high:
            high_mid.tracks.append(MidiFileUtils.notes_to_track(high, track_name="High"))

        return low_mid, high_mid

    @staticmethod
    def merge_files(files: list[mido.MidiFile]) -> mido.MidiFile:
        """Merge multiple MIDI files into one (all tracks combined)."""
        if not files:
            return MidiFileUtils.create()

        result = MidiFileUtils.create(files[0].ticks_per_beat)
        for midi_file in files:
            for track in midi_file.tracks[1:]:  # Skip tempo tracks
                result.tracks.append(track)

        return result

    @staticmethod
    def get_tempo(midi_file: mido.MidiFile) -> float:
        """Get the initial tempo in BPM."""
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    return mido.tempo2bpm(msg.tempo)
        return 120.0  # Default

    @staticmethod
    def set_tempo(midi_file: mido.MidiFile, bpm: float) -> None:
        """Set the tempo of the first track."""
        tempo = mido.bpm2tempo(bpm)
        for track in midi_file.tracks:
            for i, msg in enumerate(track):
                if msg.type == "set_tempo":
                    track[i] = mido.MetaMessage("set_tempo", tempo=tempo, time=msg.time)
                    return
        # No tempo message found, add one
        if midi_file.tracks:
            midi_file.tracks[0].insert(0, mido.MetaMessage("set_tempo", tempo=tempo))

    @staticmethod
    def get_duration_beats(midi_file: mido.MidiFile) -> float:
        """Get the total duration in beats."""
        max_tick = 0
        for track in midi_file.tracks:
            tick = 0
            for msg in track:
                tick += msg.time
            max_tick = max(max_tick, tick)
        return max_tick / midi_file.ticks_per_beat

    @staticmethod
    def quantize(midi_file: mido.MidiFile, grid: float = 0.25) -> mido.MidiFile:
        """Quantize note starts to the nearest grid position.

        Args:
            midi_file: Input MIDI file
            grid: Grid size in beats (0.25 = 16th note)
        """
        notes = MidiFileUtils.extract_notes(midi_file)
        grid_ticks = int(grid * midi_file.ticks_per_beat)

        for note in notes:
            note.start_tick = round(note.start_tick / grid_ticks) * grid_ticks

        result = MidiFileUtils.create(midi_file.ticks_per_beat)
        track = MidiFileUtils.notes_to_track(notes)
        result.tracks.append(track)
        return result

    @staticmethod
    def transpose(midi_file: mido.MidiFile, semitones: int) -> mido.MidiFile:
        """Transpose all notes by the given number of semitones."""
        notes = MidiFileUtils.extract_notes(midi_file)
        for note in notes:
            note.pitch = max(0, min(127, note.pitch + semitones))

        result = MidiFileUtils.create(midi_file.ticks_per_beat)
        track = MidiFileUtils.notes_to_track(notes)
        result.tracks.append(track)
        return result

    @staticmethod
    def info(midi_file: mido.MidiFile) -> dict[str, object]:
        """Get summary information about a MIDI file."""
        notes = MidiFileUtils.extract_notes(midi_file)
        return {
            "tracks": len(midi_file.tracks),
            "ticks_per_beat": midi_file.ticks_per_beat,
            "tempo_bpm": MidiFileUtils.get_tempo(midi_file),
            "duration_beats": MidiFileUtils.get_duration_beats(midi_file),
            "total_notes": len(notes),
            "channels_used": sorted(set(n.channel for n in notes)),
            "pitch_range": (min(n.pitch for n in notes), max(n.pitch for n in notes))
            if notes
            else (0, 0),
            "velocity_range": (min(n.velocity for n in notes), max(n.velocity for n in notes))
            if notes
            else (0, 0),
        }

    @staticmethod
    def notes_to_midi_file(
        notes: list[NoteEvent],
        ticks_per_beat: int = 480,
        tempo: int = 120,
        track_name: str = "Track 1",
    ) -> mido.MidiFile:
        """Convenience: create a complete MIDI file from a list of NoteEvents."""
        mid = MidiFileUtils.create(ticks_per_beat, tempo)
        track = MidiFileUtils.notes_to_track(notes, track_name=track_name)
        mid.tracks.append(track)
        return mid
