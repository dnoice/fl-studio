"""Album Manager - Album-level structure and track organization.

Handles album assembly, track ordering, disc management, compilation
packaging, and album-level metadata for distribution.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from release.metadata import TrackMetadata


@dataclass
class DiscInfo:
    """Information about a single disc in a multi-disc release."""

    disc_number: int = 1
    disc_title: str = ""
    track_indices: list[int] = field(default_factory=list)  # Indices into Album.tracks


@dataclass
class AlbumTrack:
    """A track within an album context."""

    metadata: TrackMetadata
    audio_path: str = ""
    duration_s: float = 0.0
    gap_before_s: float = 0.0  # Pre-gap silence (CD-style)
    hidden: bool = False  # Hidden/bonus track


@dataclass
class AlbumCredits:
    """Album-level credits."""

    executive_producer: str = ""
    producers: list[str] = field(default_factory=list)
    engineers: list[str] = field(default_factory=list)
    mixing_engineers: list[str] = field(default_factory=list)
    mastering_engineer: str = ""
    artwork_by: str = ""
    photography_by: str = ""
    liner_notes_by: str = ""
    additional_credits: dict[str, str] = field(default_factory=dict)


class Album:
    """Complete album structure and management.

    Organizes tracks, manages disc layout, validates album-level
    metadata, and prepares for distribution.
    """

    def __init__(self, title: str = "", artist: str = ""):
        self.title = title
        self.artist = artist
        self.album_artist: str = artist
        self.genre: str = ""
        self.year: int = 0
        self.release_date: str = ""
        self.label: str = ""
        self.catalog_number: str = ""
        self.upc: str = ""
        self.copyright: str = ""
        self.release_type: str = "album"  # album, ep, single, compilation, mixtape
        self.artwork_path: str = ""
        self.credits = AlbumCredits()

        self._tracks: list[AlbumTrack] = []
        self._discs: list[DiscInfo] = []

    # ─── Track Management ───

    def add_track(
        self, metadata: TrackMetadata, audio_path: str = "", duration_s: float = 0.0
    ) -> int:
        """Add a track to the album.

        Returns:
            Track index (0-based).
        """
        track = AlbumTrack(
            metadata=metadata,
            audio_path=audio_path,
            duration_s=duration_s,
        )

        # Auto-fill album-level fields
        if not metadata.album:
            metadata.album = self.title
        if not metadata.album_artist:
            metadata.album_artist = self.album_artist
        if not metadata.year and self.year:
            metadata.year = self.year
        if not metadata.genre and self.genre:
            metadata.genre = self.genre
        if not metadata.label and self.label:
            metadata.label = self.label
        if not metadata.copyright and self.copyright:
            metadata.copyright = self.copyright
        if not metadata.upc and self.upc:
            metadata.upc = self.upc

        idx = len(self._tracks)
        self._tracks.append(track)

        # Auto-assign track numbers
        self._renumber_tracks()

        return idx

    def insert_track(self, index: int, metadata: TrackMetadata, audio_path: str = "") -> None:
        """Insert a track at a specific position."""
        track = AlbumTrack(metadata=metadata, audio_path=audio_path)
        self._tracks.insert(index, track)
        self._renumber_tracks()

    def remove_track(self, index: int) -> AlbumTrack:
        """Remove and return a track by index."""
        track = self._tracks.pop(index)
        self._renumber_tracks()
        return track

    def move_track(self, from_index: int, to_index: int) -> None:
        """Move a track from one position to another."""
        track = self._tracks.pop(from_index)
        self._tracks.insert(to_index, track)
        self._renumber_tracks()

    def swap_tracks(self, index_a: int, index_b: int) -> None:
        """Swap two tracks."""
        self._tracks[index_a], self._tracks[index_b] = (
            self._tracks[index_b],
            self._tracks[index_a],
        )
        self._renumber_tracks()

    @property
    def tracks(self) -> list[AlbumTrack]:
        return list(self._tracks)

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def _renumber_tracks(self):
        """Re-assign track numbers based on current order."""
        total = len(self._tracks)
        for i, track in enumerate(self._tracks):
            track.metadata.track_number = i + 1
            track.metadata.track_total = total

    # ─── Disc Management ───

    def set_disc_layout(self, disc_splits: list[int]) -> None:
        """Define multi-disc layout.

        Args:
            disc_splits: Number of tracks per disc.
                         e.g., [8, 7] = disc 1 has 8 tracks, disc 2 has 7.
        """
        self._discs.clear()
        offset = 0
        disc_total = len(disc_splits)

        for disc_num, count in enumerate(disc_splits, 1):
            indices = list(range(offset, offset + count))
            self._discs.append(
                DiscInfo(
                    disc_number=disc_num,
                    track_indices=indices,
                )
            )

            # Update track disc info
            local_track = 0
            for idx in indices:
                if idx < len(self._tracks):
                    local_track += 1
                    self._tracks[idx].metadata.disc_number = disc_num
                    self._tracks[idx].metadata.disc_total = disc_total
                    self._tracks[idx].metadata.track_number = local_track
                    self._tracks[idx].metadata.track_total = count

            offset += count

    def auto_disc_split(self, max_duration_min: float = 79.0) -> list[int]:
        """Automatically split tracks across discs based on duration.

        Uses CD limit (79 min) by default.

        Returns:
            List of track counts per disc (same as set_disc_layout input).
        """
        splits = []
        current_duration = 0.0
        current_count = 0
        max_duration_s = max_duration_min * 60

        for track in self._tracks:
            if current_duration + track.duration_s > max_duration_s and current_count > 0:
                splits.append(current_count)
                current_duration = 0.0
                current_count = 0

            current_duration += track.duration_s
            current_count += 1

        if current_count > 0:
            splits.append(current_count)

        if len(splits) > 1:
            self.set_disc_layout(splits)

        return splits

    @property
    def discs(self) -> list[DiscInfo]:
        return (
            list(self._discs)
            if self._discs
            else [DiscInfo(disc_number=1, track_indices=list(range(len(self._tracks))))]
        )

    # ─── Validation ───

    def validate(self) -> list[str]:
        """Validate album for distribution readiness.

        Returns:
            List of warnings and errors.
        """
        issues = []

        if not self.title:
            issues.append("ERROR: Album title is required")
        if not self.artist and not self.album_artist:
            issues.append("ERROR: Album artist is required")
        if not self._tracks:
            issues.append("ERROR: Album has no tracks")
        if not self.upc:
            issues.append("WARN: UPC/EAN code not set (required for most platforms)")
        if not self.artwork_path:
            issues.append("WARN: Album artwork not set")
        if not self.copyright:
            issues.append("WARN: Copyright notice not set")
        if not self.release_date and not self.year:
            issues.append("WARN: Release date not set")

        # Release type validation
        if self.release_type == "single" and len(self._tracks) > 3:
            issues.append("WARN: Singles typically have 1-3 tracks")
        elif self.release_type == "ep" and len(self._tracks) > 6:
            issues.append("WARN: EPs typically have 4-6 tracks")

        # Validate each track
        for i, track in enumerate(self._tracks):
            track_issues = track.metadata.validate()
            for issue in track_issues:
                issues.append(f"Track {i+1} ({track.metadata.title}): {issue}")

        # Check for duplicate ISRCs
        isrcs = [t.metadata.isrc for t in self._tracks if t.metadata.isrc]
        duplicates = {x for x in isrcs if isrcs.count(x) > 1}
        for dup in duplicates:
            issues.append(f"ERROR: Duplicate ISRC: {dup}")

        # Check for missing audio files
        for i, track in enumerate(self._tracks):
            if track.audio_path and not Path(track.audio_path).exists():
                issues.append(f"ERROR: Track {i+1} audio file not found: {track.audio_path}")

        return issues

    # ─── Serialization ───

    @property
    def total_duration_s(self) -> float:
        return sum(t.duration_s for t in self._tracks)

    @property
    def total_duration_str(self) -> str:
        total = self.total_duration_s
        minutes = int(total // 60)
        seconds = int(total % 60)
        return f"{minutes}:{seconds:02d}"

    def to_dict(self) -> dict:
        """Export album as a dictionary."""
        return {
            "title": self.title,
            "artist": self.artist,
            "album_artist": self.album_artist,
            "genre": self.genre,
            "year": self.year,
            "release_date": self.release_date,
            "release_type": self.release_type,
            "label": self.label,
            "catalog_number": self.catalog_number,
            "upc": self.upc,
            "copyright": self.copyright,
            "artwork_path": self.artwork_path,
            "track_count": len(self._tracks),
            "total_duration": self.total_duration_str,
            "credits": {
                "executive_producer": self.credits.executive_producer,
                "producers": self.credits.producers,
                "engineers": self.credits.engineers,
                "mixing_engineers": self.credits.mixing_engineers,
                "mastering_engineer": self.credits.mastering_engineer,
                "artwork_by": self.credits.artwork_by,
                "additional": self.credits.additional_credits,
            },
            "tracks": [
                {
                    "track_number": t.metadata.track_number,
                    "title": t.metadata.title,
                    "artist": t.metadata.artist,
                    "duration_s": t.duration_s,
                    "isrc": t.metadata.isrc,
                    "audio_path": t.audio_path,
                    "explicit": t.metadata.explicit,
                }
                for t in self._tracks
            ],
        }

    def save_json(self, path: str) -> None:
        """Save album project to JSON."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load_json(cls, path: str) -> "Album":
        """Load album from JSON project file."""
        data = json.loads(Path(path).read_text())
        album = cls(title=data.get("title", ""), artist=data.get("artist", ""))
        album.album_artist = data.get("album_artist", album.artist)
        album.genre = data.get("genre", "")
        album.year = data.get("year", 0)
        album.release_date = data.get("release_date", "")
        album.release_type = data.get("release_type", "album")
        album.label = data.get("label", "")
        album.catalog_number = data.get("catalog_number", "")
        album.upc = data.get("upc", "")
        album.copyright = data.get("copyright", "")
        album.artwork_path = data.get("artwork_path", "")

        credits_data = data.get("credits", {})
        album.credits.executive_producer = credits_data.get("executive_producer", "")
        album.credits.producers = credits_data.get("producers", [])
        album.credits.engineers = credits_data.get("engineers", [])
        album.credits.mixing_engineers = credits_data.get("mixing_engineers", [])
        album.credits.mastering_engineer = credits_data.get("mastering_engineer", "")
        album.credits.artwork_by = credits_data.get("artwork_by", "")
        album.credits.additional_credits = credits_data.get("additional", {})

        for track_data in data.get("tracks", []):
            meta = TrackMetadata(
                title=track_data.get("title", ""),
                artist=track_data.get("artist", album.artist),
                isrc=track_data.get("isrc", ""),
                explicit=track_data.get("explicit", False),
            )
            album.add_track(
                metadata=meta,
                audio_path=track_data.get("audio_path", ""),
                duration_s=track_data.get("duration_s", 0.0),
            )

        return album

    def tracklist(self) -> str:
        """Generate formatted tracklist."""
        lines = [f"=== {self.title} ==="]
        if self.artist:
            lines.append(f"by {self.artist}")
        if self.release_date:
            lines.append(f"Released: {self.release_date}")
        elif self.year:
            lines.append(f"Year: {self.year}")
        lines.append("")

        for track in self._tracks:
            m = track.metadata
            dur = ""
            if track.duration_s:
                mins = int(track.duration_s // 60)
                secs = int(track.duration_s % 60)
                dur = f" [{mins}:{secs:02d}]"

            feat = f" (feat. {m.featuring})" if m.featuring else ""
            explicit = " [E]" if m.explicit else ""
            artist_note = f" - {m.artist}" if m.artist != self.artist else ""

            line = f"  {m.track_number:2d}. {m.title}{feat}{artist_note}{explicit}{dur}"
            lines.append(line)

        lines.append(f"\n  Total: {self.total_duration_str} | {self.track_count} tracks")
        if self.label:
            lines.append(f"  {self.copyright or ''} | {self.label}")

        return "\n".join(lines)
