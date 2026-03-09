"""Lyrics Manager - Lyrics storage, syncing, and embedding.

Handles plain text and time-synced (LRC format) lyrics, with utilities
for editing, formatting, importing/exporting, and embedding into
audio file metadata.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SyncedLine:
    """A single line of time-synced lyrics."""

    timestamp_ms: int  # Milliseconds from track start
    text: str
    end_ms: int = 0  # Optional end time for karaoke-style

    @property
    def timestamp_str(self) -> str:
        """Format as [mm:ss.xx]."""
        minutes = self.timestamp_ms // 60000
        seconds = (self.timestamp_ms % 60000) / 1000
        return f"[{minutes:02d}:{seconds:05.2f}]"

    @property
    def timestamp_lrc(self) -> str:
        """Standard LRC timestamp format [mm:ss.xx]."""
        minutes = self.timestamp_ms // 60000
        seconds = (self.timestamp_ms % 60000) // 1000
        hundredths = (self.timestamp_ms % 1000) // 10
        return f"[{minutes:02d}:{seconds:02d}.{hundredths:02d}]"


@dataclass
class LyricsSection:
    """A labeled section of lyrics (verse, chorus, etc.)."""

    label: str  # "Verse 1", "Chorus", "Bridge", etc.
    lines: list[str] = field(default_factory=list)


class Lyrics:
    """Complete lyrics for a single track.

    Supports both plain text and synced (LRC) formats, section labeling,
    and multiple language translations.
    """

    def __init__(self, title: str = "", artist: str = ""):
        self.title = title
        self.artist = artist
        self.album: str = ""
        self.language: str = "eng"  # ISO 639-2
        self._plain_text: str = ""
        self._synced_lines: list[SyncedLine] = []
        self._sections: list[LyricsSection] = []
        self._translations: dict[str, str] = {}  # lang_code -> text

    # ─── Plain Text ───

    @property
    def text(self) -> str:
        """Get plain text lyrics."""
        if self._plain_text:
            return self._plain_text
        # Generate from synced lines if available
        if self._synced_lines:
            return "\n".join(line.text for line in self._synced_lines)
        # Generate from sections
        if self._sections:
            parts = []
            for section in self._sections:
                parts.append(f"[{section.label}]")
                parts.extend(section.lines)
                parts.append("")
            return "\n".join(parts).strip()
        return ""

    @text.setter
    def text(self, value: str):
        self._plain_text = value

    # ─── Synced Lyrics ───

    @property
    def synced_lines(self) -> list[SyncedLine]:
        return list(self._synced_lines)

    def add_synced_line(self, timestamp_ms: int, text: str, end_ms: int = 0) -> None:
        """Add a time-synced line."""
        line = SyncedLine(timestamp_ms=timestamp_ms, text=text, end_ms=end_ms)
        self._synced_lines.append(line)
        self._synced_lines.sort(key=lambda x: x.timestamp_ms)

    def clear_sync(self) -> None:
        """Remove all sync timing data."""
        self._synced_lines.clear()

    @property
    def is_synced(self) -> bool:
        return len(self._synced_lines) > 0

    # ─── Sections ───

    def add_section(self, label: str, lines: list[str]) -> None:
        """Add a labeled section."""
        self._sections.append(LyricsSection(label=label, lines=lines))

    @property
    def sections(self) -> list[LyricsSection]:
        return list(self._sections)

    def set_sections_from_text(self, text: str) -> None:
        """Parse sections from text with [Section] labels.

        Format:
            [Verse 1]
            Line one
            Line two

            [Chorus]
            Chorus line
        """
        self._sections.clear()
        current_label = "Intro"
        current_lines: list[str] = []

        for line in text.split("\n"):
            stripped = line.strip()
            match = re.match(r"^\[(.+?)\]$", stripped)
            if match:
                if current_lines:
                    self._sections.append(LyricsSection(label=current_label, lines=current_lines))
                current_label = match.group(1)
                current_lines = []
            elif stripped:
                current_lines.append(stripped)

        if current_lines:
            self._sections.append(LyricsSection(label=current_label, lines=current_lines))

    # ─── Translations ───

    def add_translation(self, language: str, text: str) -> None:
        """Add a translation of the lyrics."""
        self._translations[language] = text

    def get_translation(self, language: str) -> str | None:
        return self._translations.get(language)

    @property
    def translations(self) -> dict[str, str]:
        return dict(self._translations)

    # ─── LRC Format ───

    def to_lrc(self) -> str:
        """Export as LRC format string.

        LRC format:
            [ti:Title]
            [ar:Artist]
            [al:Album]
            [la:eng]
            [00:12.34]First line of lyrics
            [00:15.67]Second line
        """
        lines = []

        # Header tags
        if self.title:
            lines.append(f"[ti:{self.title}]")
        if self.artist:
            lines.append(f"[ar:{self.artist}]")
        if self.album:
            lines.append(f"[al:{self.album}]")
        if self.language:
            lines.append(f"[la:{self.language}]")
        lines.append("")

        # Synced lines
        if self._synced_lines:
            for sl in self._synced_lines:
                lines.append(f"{sl.timestamp_lrc}{sl.text}")
        else:
            # Output plain text without timestamps
            for line in self.text.split("\n"):
                lines.append(line)

        return "\n".join(lines)

    @classmethod
    def from_lrc(cls, lrc_text: str) -> "Lyrics":
        """Parse an LRC file into a Lyrics object."""
        lyrics = cls()

        for line in lrc_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Header tags
            header_match = re.match(r"^\[(\w+):(.+)\]$", line)
            if header_match:
                tag = header_match.group(1).lower()
                value = header_match.group(2).strip()
                if tag == "ti":
                    lyrics.title = value
                elif tag == "ar":
                    lyrics.artist = value
                elif tag == "al":
                    lyrics.album = value
                elif tag == "la":
                    lyrics.language = value
                continue

            # Synced lines: [mm:ss.xx]text
            sync_match = re.match(r"^\[(\d{2}):(\d{2})\.(\d{2})\](.*)$", line)
            if sync_match:
                minutes = int(sync_match.group(1))
                seconds = int(sync_match.group(2))
                hundredths = int(sync_match.group(3))
                text = sync_match.group(4)
                ms = minutes * 60000 + seconds * 1000 + hundredths * 10
                lyrics.add_synced_line(ms, text)
                continue

            # Multi-timestamp lines: [mm:ss.xx][mm:ss.xx]text
            multi_match = re.findall(r"\[(\d{2}):(\d{2})\.(\d{2})\]", line)
            if multi_match:
                text = re.sub(r"\[\d{2}:\d{2}\.\d{2}\]", "", line).strip()
                for m, s, h in multi_match:
                    ms = int(m) * 60000 + int(s) * 1000 + int(h) * 10
                    lyrics.add_synced_line(ms, text)

        return lyrics

    # ─── File I/O ───

    def save_lrc(self, path: str) -> None:
        """Save lyrics as an LRC file."""
        Path(path).write_text(self.to_lrc(), encoding="utf-8")

    @classmethod
    def load_lrc(cls, path: str) -> "Lyrics":
        """Load lyrics from an LRC file."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_lrc(text)

    def save_txt(self, path: str) -> None:
        """Save plain text lyrics."""
        Path(path).write_text(self.text, encoding="utf-8")

    @classmethod
    def load_txt(cls, path: str, title: str = "", artist: str = "") -> "Lyrics":
        """Load plain text lyrics from file."""
        lyrics = cls(title=title, artist=artist)
        lyrics.text = Path(path).read_text(encoding="utf-8")
        return lyrics

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "language": self.language,
            "text": self.text,
            "synced": [
                {"ms": sl.timestamp_ms, "text": sl.text, "end_ms": sl.end_ms}
                for sl in self._synced_lines
            ],
            "sections": [{"label": s.label, "lines": s.lines} for s in self._sections],
            "translations": self._translations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Lyrics":
        """Create from dictionary."""
        lyrics = cls(
            title=data.get("title", ""),
            artist=data.get("artist", ""),
        )
        lyrics.album = data.get("album", "")
        lyrics.language = data.get("language", "eng")
        lyrics.text = data.get("text", "")

        for sl in data.get("synced", []):
            lyrics.add_synced_line(sl["ms"], sl["text"], sl.get("end_ms", 0))

        for sec in data.get("sections", []):
            lyrics.add_section(sec["label"], sec["lines"])

        lyrics._translations = data.get("translations", {})
        return lyrics


class LyricsManager:
    """Manage lyrics for an entire album or catalog.

    Provides bulk operations, search, and lyrics database management.
    """

    def __init__(self):
        self._lyrics: dict[str, Lyrics] = {}  # key: "artist - title"

    def add(self, lyrics: Lyrics) -> str:
        """Add lyrics to the manager.

        Returns:
            Key string used for retrieval.
        """
        key = f"{lyrics.artist} - {lyrics.title}".strip(" -")
        self._lyrics[key] = lyrics
        return key

    def get(self, artist: str = "", title: str = "") -> Lyrics | None:
        """Get lyrics by artist and title."""
        key = f"{artist} - {title}".strip(" -")
        return self._lyrics.get(key)

    def search(self, query: str) -> list[Lyrics]:
        """Search lyrics by text content."""
        query_lower = query.lower()
        results = []
        for lyrics in self._lyrics.values():
            if (
                query_lower in lyrics.text.lower()
                or query_lower in lyrics.title.lower()
                or query_lower in lyrics.artist.lower()
            ):
                results.append(lyrics)
        return results

    @property
    def all_lyrics(self) -> list[Lyrics]:
        return list(self._lyrics.values())

    def save_database(self, path: str) -> None:
        """Save all lyrics to a JSON database."""
        data = {key: lyrics.to_dict() for key, lyrics in self._lyrics.items()}
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_database(cls, path: str) -> "LyricsManager":
        """Load lyrics database from JSON."""
        manager = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, lyrics_data in data.items():
            lyrics = Lyrics.from_dict(lyrics_data)
            manager._lyrics[key] = lyrics
        return manager

    def import_lrc_directory(self, directory: str) -> int:
        """Import all .lrc files from a directory.

        Returns:
            Number of files imported.
        """
        count = 0
        for lrc_file in Path(directory).glob("*.lrc"):
            lyrics = Lyrics.load_lrc(str(lrc_file))
            if not lyrics.title:
                lyrics.title = lrc_file.stem
            self.add(lyrics)
            count += 1
        return count

    def export_all_lrc(self, directory: str) -> int:
        """Export all lyrics as individual .lrc files.

        Returns:
            Number of files exported.
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for key, lyrics in self._lyrics.items():
            safe_name = re.sub(r'[<>:"/\\|?*]', "_", key)
            lyrics.save_lrc(str(out_dir / f"{safe_name}.lrc"))
            count += 1
        return count

    def word_frequency(self) -> dict[str, int]:
        """Analyze word frequency across all lyrics."""
        freq: dict[str, int] = {}
        for lyrics in self._lyrics.values():
            words = re.findall(r"\b\w+\b", lyrics.text.lower())
            for word in words:
                freq[word] = freq.get(word, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: -x[1]))
