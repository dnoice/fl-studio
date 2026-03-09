"""FLP Parser - Parse FL Studio project files (.flp).

FL Studio project files use a binary chunk-based format.
This parser extracts metadata, channel information, mixer settings,
tempo, and plugin data from .flp files.
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO


class FLPEvent(IntEnum):
    """FL Studio project file event types."""

    # Byte events (0-63)
    FLP_Byte = 0
    FLP_Enabled = 0
    FLP_NoteOn = 1
    FLP_Vol = 2
    FLP_Pan = 3
    FLP_MIDIChan = 4
    FLP_MIDINote = 5
    FLP_MIDIPatch = 6
    FLP_MIDIBank = 7
    FLP_LoopActive = 9
    FLP_ShowInfo = 10
    FLP_Shuffle = 11
    FLP_MainVol = 12
    FLP_Stretch = 13
    FLP_Pitchable = 14
    FLP_Zipped = 15
    FLP_Delay_Flags = 16
    FLP_PatLength = 17
    FLP_BlockLength = 18
    FLP_UseLoopPoints = 19
    FLP_LoopType = 20
    FLP_ChanType = 21
    FLP_MixSliceNum = 22
    FLP_EffectChannelMuted = 27

    # Word events (64-127)
    FLP_Word = 64
    FLP_NewChan = 64
    FLP_NewPat = 65
    FLP_Tempo = 66
    FLP_CurrentPatNum = 67
    FLP_PatData = 68
    FLP_FX = 69
    FLP_Fade_Stereo = 70
    FLP_CutOff = 71
    FLP_DotVol = 72
    FLP_DotPan = 73
    FLP_PreAmp = 74
    FLP_Decay = 75
    FLP_Attack = 76
    FLP_DotNote = 77
    FLP_DotPitch = 78
    FLP_DotMix = 79
    FLP_MainPitch = 80
    FLP_RandChan = 81
    FLP_MixChan = 82
    FLP_Resonance = 83
    FLP_LoopBar = 84
    FLP_StDel = 85
    FLP_FX3 = 86
    FLP_DotReso = 87
    FLP_DotCutOff = 88
    FLP_ShiftDelay = 89
    FLP_LoopEndBar = 90
    FLP_Dot = 91
    FLP_DotShift = 92
    FLP_LayerChans = 94
    FLP_InsertIcon = 155

    # Int events (128-191)
    FLP_Int = 128
    FLP_Color = 128
    FLP_PlayListItem = 129
    FLP_Echo = 130
    FLP_FXSine = 131
    FLP_CutCutBy = 132
    FLP_WindowH = 133
    FLP_MiddleNote = 135
    FLP_InsertColor = 136
    FLP_DelayReso = 137
    FLP_Reverb = 138
    FLP_IntStretch = 139
    FLP_SSNote = 140
    FLP_FineTune = 141

    # Text events (192+)
    FLP_Text = 192
    FLP_ChanName = 192
    FLP_PatName = 193
    FLP_Title = 194
    FLP_Comment = 195
    FLP_SampleFileName = 196
    FLP_URL = 197
    FLP_CommentRTF = 198
    FLP_Version = 199
    FLP_PluginName = 201
    FLP_MIDICtrls = 204
    FLP_Delay = 205
    FLP_TS404Params = 206
    FLP_DelayLine = 207
    FLP_NewPlugin = 208
    FLP_PluginParams = 209
    FLP_ChanParams = 210
    FLP_CtrlRecChan = 224
    FLP_PLSel = 225
    FLP_Envelope = 226
    FLP_BasicChanParams = 227
    FLP_OldFilterParams = 228
    FLP_AutomationData = 233
    FLP_InsertParams = 234
    FLP_ChanGroupName = 235
    FLP_PlaylistItems = 236
    FLP_InsertRoute = 237
    FLP_InsertFlags = 238
    FLP_PlaylistTrackName = 241


@dataclass
class FLPChannel:
    """A channel in the FL project."""

    index: int
    name: str = ""
    type: int = 0
    plugin_name: str = ""
    sample_filename: str = ""
    volume: int = 100
    pan: int = 64
    color: int = 0
    mixer_track: int = 0


@dataclass
class FLPPattern:
    """A pattern in the FL project."""

    index: int
    name: str = ""
    color: int = 0


@dataclass
class FLPProject:
    """Parsed FL Studio project data."""

    filepath: str = ""
    version: str = ""
    title: str = ""
    comment: str = ""
    url: str = ""
    tempo: float = 140.0
    main_volume: int = 100
    main_pitch: int = 0
    channels: list[FLPChannel] = field(default_factory=list)
    patterns: list[FLPPattern] = field(default_factory=list)
    plugin_names: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== FL Studio Project ===",
            f"File: {self.filepath}",
            f"Version: {self.version}",
            f"Title: {self.title or '(untitled)'}",
            f"Tempo: {self.tempo} BPM",
            f"Channels: {len(self.channels)}",
            f"Patterns: {len(self.patterns)}",
            "",
            "--- Channels ---",
        ]
        for ch in self.channels:
            plugin = ch.plugin_name or ch.sample_filename or "?"
            lines.append(f"  [{ch.index}] {ch.name or '(unnamed)'} - {plugin}")

        if self.patterns:
            lines.append("")
            lines.append("--- Patterns ---")
            for pat in self.patterns:
                lines.append(f"  [{pat.index}] {pat.name or f'Pattern {pat.index}'}")

        if self.plugin_names:
            lines.append("")
            lines.append("--- Plugins Used ---")
            for name in sorted(set(self.plugin_names)):
                lines.append(f"  - {name}")

        return "\n".join(lines)


class FLPParser:
    """Parser for FL Studio .flp project files."""

    FLP_HEADER = b"FLhd"
    FLP_DATA = b"FLdt"

    @staticmethod
    def parse(filepath: str | Path) -> FLPProject:
        """Parse an FL Studio project file.

        Args:
            filepath: Path to .flp file

        Returns:
            FLPProject with extracted data
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"FLP file not found: {filepath}")

        project = FLPProject(filepath=str(filepath))

        try:
            return FLPParser._parse_file(filepath, project)
        except (struct.error, EOFError) as e:
            raise ValueError(f"Malformed FLP file '{filepath}': {e}") from e
        except OSError as e:
            raise OSError(f"Failed to read FLP file '{filepath}': {e}") from e

    @staticmethod
    def _parse_file(filepath: Path, project: FLPProject) -> FLPProject:
        """Internal parsing logic."""
        with open(filepath, "rb") as f:
            # Read header
            header_id = f.read(4)
            if header_id != FLPParser.FLP_HEADER:
                raise ValueError(f"Not a valid FLP file: {filepath}")

            header_length = struct.unpack("<I", f.read(4))[0]
            struct.unpack("<H", f.read(2))[0]
            struct.unpack("<H", f.read(2))[0]
            struct.unpack("<H", f.read(2))[0]

            # Skip remaining header bytes
            if header_length > 6:
                f.read(header_length - 6)

            # Read data chunk
            data_id = f.read(4)
            if data_id != FLPParser.FLP_DATA:
                raise ValueError("Missing FLdt chunk")

            data_length = struct.unpack("<I", f.read(4))[0]

            # Parse events
            current_channel = None
            current_pattern = None

            end_pos = f.tell() + data_length
            while f.tell() < end_pos:
                event_id = FLPParser._read_byte(f)
                if event_id is None:
                    break

                if event_id < 64:  # Byte event
                    value = FLPParser._read_byte(f)
                    if value is None:
                        break
                    FLPParser._handle_byte_event(project, event_id, value, current_channel)

                elif event_id < 128:  # Word event (2 bytes)
                    value = FLPParser._read_word(f)
                    if value is None:
                        break
                    result = FLPParser._handle_word_event(
                        project, event_id, value, current_channel, current_pattern
                    )
                    if result is not None:
                        if isinstance(result, FLPChannel):
                            current_channel = result
                        elif isinstance(result, FLPPattern):
                            current_pattern = result

                elif event_id < 192:  # DWord event (4 bytes)
                    value = FLPParser._read_dword(f)
                    if value is None:
                        break
                    FLPParser._handle_dword_event(project, event_id, value, current_channel)

                else:  # Text/data event (variable length)
                    length = FLPParser._read_varint(f)
                    if length is None:
                        break
                    data = f.read(length)
                    FLPParser._handle_text_event(
                        project, event_id, data, current_channel, current_pattern
                    )

        return project

    @staticmethod
    def _read_byte(f: BinaryIO) -> int | None:
        data = f.read(1)
        return data[0] if data else None

    @staticmethod
    def _read_word(f: BinaryIO) -> int | None:
        data = f.read(2)
        return struct.unpack("<H", data)[0] if len(data) == 2 else None

    @staticmethod
    def _read_dword(f: BinaryIO) -> int | None:
        data = f.read(4)
        return struct.unpack("<I", data)[0] if len(data) == 4 else None

    @staticmethod
    def _read_varint(f: BinaryIO) -> int | None:
        """Read a variable-length integer (FL format)."""
        result = 0
        shift = 0
        while True:
            byte = f.read(1)
            if not byte:
                return None
            b = byte[0]
            result |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                break
        return result

    @staticmethod
    def _handle_byte_event(
        project: FLPProject, event_id: int, value: int, channel: FLPChannel | None
    ) -> None:
        if event_id == FLPEvent.FLP_MainVol:
            project.main_volume = value
        elif event_id == FLPEvent.FLP_ChanType and channel:
            channel.type = value
        elif event_id == FLPEvent.FLP_Vol and channel:
            channel.volume = value
        elif event_id == FLPEvent.FLP_Pan and channel:
            channel.pan = value
        elif event_id == FLPEvent.FLP_MixSliceNum and channel:
            channel.mixer_track = value

    @staticmethod
    def _handle_word_event(
        project: FLPProject,
        event_id: int,
        value: int,
        channel: FLPChannel | None,
        pattern: FLPPattern | None,
    ) -> FLPChannel | FLPPattern | None:
        if event_id == FLPEvent.FLP_NewChan:
            new_channel = FLPChannel(index=value)
            project.channels.append(new_channel)
            return new_channel
        elif event_id == FLPEvent.FLP_NewPat:
            new_pattern = FLPPattern(index=value)
            project.patterns.append(new_pattern)
            return new_pattern
        elif event_id == FLPEvent.FLP_Tempo:
            project.tempo = value
        elif event_id == FLPEvent.FLP_MainPitch:
            project.main_pitch = value
        return None

    @staticmethod
    def _handle_dword_event(
        project: FLPProject, event_id: int, value: int, channel: FLPChannel | None
    ) -> None:
        if event_id == FLPEvent.FLP_Color and channel:
            channel.color = value
        elif event_id == FLPEvent.FLP_FineTune and channel:
            pass  # Store if needed
        elif event_id == FLPEvent.FLP_InsertColor:
            pass  # Mixer insert color

    @staticmethod
    def _handle_text_event(
        project: FLPProject,
        event_id: int,
        data: bytes,
        channel: FLPChannel | None,
        pattern: FLPPattern | None,
    ) -> None:
        try:
            text = data.decode("utf-16-le").rstrip("\x00")
        except (UnicodeDecodeError, ValueError):
            text = data.decode("ascii", errors="replace").rstrip("\x00")

        if event_id == FLPEvent.FLP_ChanName and channel:
            channel.name = text
        elif event_id == FLPEvent.FLP_PatName and pattern:
            pattern.name = text
        elif event_id == FLPEvent.FLP_Title:
            project.title = text
        elif event_id == FLPEvent.FLP_Comment:
            project.comment = text
        elif event_id == FLPEvent.FLP_URL:
            project.url = text
        elif event_id == FLPEvent.FLP_Version:
            project.version = text
        elif event_id == FLPEvent.FLP_PluginName:
            if channel:
                channel.plugin_name = text
            if text:
                project.plugin_names.append(text)
        elif event_id == FLPEvent.FLP_SampleFileName and channel:
            channel.sample_filename = text

    @staticmethod
    def list_plugins(filepath: str | Path) -> list[str]:
        """Get a list of all plugins used in an FLP file."""
        project = FLPParser.parse(filepath)
        return sorted(set(project.plugin_names))

    @staticmethod
    def get_tempo(filepath: str | Path) -> float:
        """Get the project tempo."""
        project = FLPParser.parse(filepath)
        return project.tempo

    @staticmethod
    def batch_info(directory: str | Path, recursive: bool = True) -> list[FLPProject]:
        """Parse all FLP files in a directory."""
        path = Path(directory)
        pattern = "**/*.flp" if recursive else "*.flp"
        results = []
        for flp_file in sorted(path.glob(pattern)):
            try:
                results.append(FLPParser.parse(flp_file))
            except (ValueError, struct.error, OSError):
                continue  # Skip unparseable or inaccessible FLP files
        return results
