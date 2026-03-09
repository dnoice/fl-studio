"""Metadata Manager - Audio file tagging and metadata pipeline.

Handles reading, writing, and validating metadata for audio files
across formats (MP3/ID3, FLAC/Vorbis, WAV/RIFF, OGG, AIFF).
Supports album art embedding, ISRC codes, and distribution-ready
metadata validation.
"""

import contextlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrackMetadata:
    """Complete metadata for a single track.

    Covers ID3v2.4, Vorbis Comment, and common distribution fields.
    """

    # Core tags
    title: str = ""
    artist: str = ""
    album_artist: str = ""
    album: str = ""
    track_number: int = 0
    track_total: int = 0
    disc_number: int = 1
    disc_total: int = 1
    year: int = 0
    genre: str = ""

    # Extended tags
    composer: str = ""
    lyricist: str = ""
    producer: str = ""
    arranger: str = ""
    conductor: str = ""
    remixer: str = ""
    featuring: str = ""
    encoded_by: str = ""
    encoder_settings: str = ""
    comment: str = ""

    # Distribution identifiers
    isrc: str = ""  # International Standard Recording Code
    upc: str = ""  # Universal Product Code (album-level)
    catalog_number: str = ""
    label: str = ""
    release_date: str = ""  # ISO 8601: YYYY-MM-DD
    original_year: int = 0
    media_type: str = ""  # CD, Digital, Vinyl, etc.

    # Copyright
    copyright: str = ""
    license: str = ""

    # Technical
    bpm: float = 0.0
    key: str = ""  # Musical key (e.g. "Cm", "F#")
    mood: str = ""
    language: str = ""  # ISO 639-2 (e.g. "eng", "spa")
    explicit: bool = False

    # Album art
    artwork_path: str = ""  # Path to cover art image
    artwork_data: bytes = field(default=b"", repr=False)
    artwork_mime: str = "image/jpeg"

    # Lyrics
    lyrics: str = ""
    synced_lyrics: str = ""  # LRC format

    # Custom/extended fields
    custom_tags: dict[str, str] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate metadata for distribution readiness.

        Returns:
            List of validation warnings/errors. Empty if all good.
        """
        issues = []

        if not self.title:
            issues.append("ERROR: Title is required")
        if not self.artist:
            issues.append("ERROR: Artist is required")
        if not self.album:
            issues.append("WARN: Album name is missing")
        if self.track_number <= 0:
            issues.append("WARN: Track number not set")
        if not self.year and not self.release_date:
            issues.append("WARN: Release date/year not set")
        if not self.genre:
            issues.append("WARN: Genre not set")
        if not self.isrc:
            issues.append("WARN: ISRC code not set (required for streaming platforms)")
        if self.isrc and not self._validate_isrc(self.isrc):
            issues.append(f"ERROR: Invalid ISRC format: {self.isrc}")
        if self.upc and not self._validate_upc(self.upc):
            issues.append(f"ERROR: Invalid UPC format: {self.upc}")
        if self.explicit and not self.language:
            issues.append("WARN: Explicit content should have language set")
        if not self.copyright:
            issues.append("WARN: Copyright notice not set")
        if len(self.title) > 200:
            issues.append("WARN: Title exceeds 200 characters")
        if not self.artwork_path and not self.artwork_data:
            issues.append("WARN: No album artwork set")

        return issues

    @staticmethod
    def _validate_isrc(isrc: str) -> bool:
        """Validate ISRC format: CC-XXX-YY-NNNNN."""
        clean = isrc.replace("-", "")
        if len(clean) != 12:
            return False
        return (
            clean[:2].isalpha()
            and clean[2:5].isalnum()
            and clean[5:7].isdigit()
            and clean[7:].isdigit()
        )

    @staticmethod
    def _validate_upc(upc: str) -> bool:
        """Validate UPC/EAN format (12 or 13 digits)."""
        return upc.isdigit() and len(upc) in (12, 13)

    def to_dict(self) -> dict:
        """Export metadata as a dictionary (for JSON serialization)."""
        data = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if k == "artwork_data":
                data[k] = len(v) if v else 0  # Just store size
                continue
            if v and v != 0 and v != 0.0 and v is not False or k == "explicit":
                data[k] = v
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrackMetadata":
        """Create metadata from a dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields and k != "artwork_data"}
        return cls(**filtered)


class MetadataManager:
    """Read, write, and manage audio file metadata.

    Uses a pure-Python approach for basic ID3v2 writing and provides
    integration hooks for the mutagen library when available for
    full format support.
    """

    # Standard ID3 genre list (subset of common genres)
    GENRES = [
        "Blues",
        "Classic Rock",
        "Country",
        "Dance",
        "Disco",
        "Funk",
        "Grunge",
        "Hip-Hop",
        "Jazz",
        "Metal",
        "New Age",
        "Oldies",
        "Other",
        "Pop",
        "R&B",
        "Rap",
        "Reggae",
        "Rock",
        "Techno",
        "Industrial",
        "Alternative",
        "Ska",
        "Death Metal",
        "Pranks",
        "Soundtrack",
        "Euro-Techno",
        "Ambient",
        "Trip-Hop",
        "Vocal",
        "Jazz+Funk",
        "Fusion",
        "Trance",
        "Classical",
        "Instrumental",
        "Acid",
        "House",
        "Game",
        "Sound Clip",
        "Gospel",
        "Noise",
        "AlternRock",
        "Bass",
        "Soul",
        "Punk",
        "Space",
        "Meditative",
        "Instrumental Pop",
        "Instrumental Rock",
        "Ethnic",
        "Gothic",
        "Darkwave",
        "Techno-Industrial",
        "Electronic",
        "Pop-Folk",
        "Eurodance",
        "Dream",
        "Southern Rock",
        "Comedy",
        "Cult",
        "Gangsta",
        "Top 40",
        "Christian Rap",
        "Pop/Funk",
        "Jungle",
        "Native American",
        "Cabaret",
        "New Wave",
        "Psychedelic",
        "Rave",
        "Showtunes",
        "Trailer",
        "Lo-Fi",
        "Tribal",
        "Acid Punk",
        "Acid Jazz",
        "Polka",
        "Retro",
        "Musical",
        "Rock & Roll",
        "Hard Rock",
        "Folk",
        "Folk-Rock",
        "National Folk",
        "Swing",
        "Fast Fusion",
        "Bebop",
        "Latin",
        "Revival",
        "Celtic",
        "Bluegrass",
        "Avantgarde",
        "Gothic Rock",
        "Progressive Rock",
        "Psychedelic Rock",
        "Symphonic Rock",
        "Slow Rock",
        "Big Band",
        "Chorus",
        "Easy Listening",
        "Acoustic",
        "Humour",
        "Speech",
        "Chanson",
        "Opera",
        "Chamber Music",
        "Sonata",
        "Symphony",
        "Booty Bass",
        "Primus",
        "Porn Groove",
        "Satire",
        "Slow Jam",
        "Club",
        "Tango",
        "Samba",
        "Folklore",
        "Ballad",
        "Power Ballad",
        "Rhythmic Soul",
        "Freestyle",
        "Duet",
        "Punk Rock",
        "Drum Solo",
        "Acapella",
        "Euro-House",
        "Dance Hall",
        "Drum & Bass",
        "Garage",
        "Merseybeat",
    ]

    # Streaming platform requirements
    PLATFORM_REQUIREMENTS = {
        "spotify": {
            "artwork_min_px": 3000,
            "artwork_format": ["jpg", "png"],
            "audio_formats": ["wav", "flac"],
            "bit_depth_min": 16,
            "sample_rate_min": 44100,
            "isrc_required": True,
            "upc_required": True,
        },
        "apple_music": {
            "artwork_min_px": 3000,
            "artwork_format": ["jpg"],
            "audio_formats": ["wav", "flac", "aiff"],
            "bit_depth_min": 16,
            "sample_rate_min": 44100,
            "isrc_required": True,
            "upc_required": True,
        },
        "bandcamp": {
            "artwork_min_px": 1400,
            "artwork_format": ["jpg", "png"],
            "audio_formats": ["wav", "flac", "aiff"],
            "bit_depth_min": 16,
            "sample_rate_min": 44100,
            "isrc_required": False,
            "upc_required": False,
        },
        "soundcloud": {
            "artwork_min_px": 800,
            "artwork_format": ["jpg", "png"],
            "audio_formats": ["wav", "flac", "mp3", "ogg", "aiff"],
            "bit_depth_min": 16,
            "sample_rate_min": 44100,
            "isrc_required": False,
            "upc_required": False,
        },
    }

    @classmethod
    def write_id3v2(cls, audio_path: str, metadata: TrackMetadata) -> None:
        """Write ID3v2.4 tags to an MP3 file (pure Python).

        For full format support, use write_tags() which tries mutagen first.
        """
        path = Path(audio_path)
        if path.suffix.lower() != ".mp3":
            raise ValueError(
                "write_id3v2 only supports MP3 files. Use write_tags() for other formats."
            )

        frames = []

        # Text frames
        text_map = {
            "TIT2": metadata.title,
            "TPE1": metadata.artist,
            "TPE2": metadata.album_artist or metadata.artist,
            "TALB": metadata.album,
            "TDRC": metadata.release_date or str(metadata.year) if metadata.year else "",
            "TCON": metadata.genre,
            "TCOM": metadata.composer,
            "TEXT": metadata.lyricist,
            "TIPL": metadata.producer,
            "TPUB": metadata.label,
            "TSRC": metadata.isrc,
            "TBPM": str(int(metadata.bpm)) if metadata.bpm else "",
            "TKEY": metadata.key,
            "TLAN": metadata.language,
            "TCOP": metadata.copyright,
            "TENC": metadata.encoded_by,
            "TSSE": metadata.encoder_settings,
        }

        if metadata.track_number:
            trck = str(metadata.track_number)
            if metadata.track_total:
                trck += f"/{metadata.track_total}"
            text_map["TRCK"] = trck

        if metadata.disc_number:
            tpos = str(metadata.disc_number)
            if metadata.disc_total:
                tpos += f"/{metadata.disc_total}"
            text_map["TPOS"] = tpos

        for frame_id, value in text_map.items():
            if value:
                frame_data = cls._encode_id3_text_frame(frame_id, value)
                frames.append(frame_data)

        # Comment frame
        if metadata.comment:
            frames.append(cls._encode_id3_comment(metadata.comment))

        # Lyrics frame (USLT)
        if metadata.lyrics:
            frames.append(cls._encode_id3_lyrics(metadata.lyrics, metadata.language or "eng"))

        # Album art (APIC)
        art_data = metadata.artwork_data
        if not art_data and metadata.artwork_path:
            art_path = Path(metadata.artwork_path)
            if art_path.exists():
                art_data = art_path.read_bytes()

        if art_data:
            frames.append(cls._encode_id3_picture(art_data, metadata.artwork_mime))

        # Build ID3v2.4 header + frames
        tag_data = b"".join(frames)
        tag_size = len(tag_data)

        # Syncsafe integer encoding
        size_bytes = bytes(
            [
                (tag_size >> 21) & 0x7F,
                (tag_size >> 14) & 0x7F,
                (tag_size >> 7) & 0x7F,
                tag_size & 0x7F,
            ]
        )

        header = b"ID3"
        header += bytes([4, 0])  # Version 2.4.0
        header += bytes([0])  # No flags
        header += size_bytes

        id3_block = header + tag_data

        # Read existing file, strip old ID3v2 tag if present, prepend new one
        file_data = path.read_bytes()
        if file_data[:3] == b"ID3":
            old_size = cls._read_syncsafe_int(file_data[6:10]) + 10
            file_data = file_data[old_size:]

        path.write_bytes(id3_block + file_data)

    @staticmethod
    def _encode_id3_text_frame(frame_id: str, text: str) -> bytes:
        """Encode an ID3v2 text frame."""
        text_data = b"\x03" + text.encode("utf-8")  # UTF-8 encoding
        size = len(text_data)
        size_bytes = struct.pack(">I", size)
        return frame_id.encode("ascii") + size_bytes + b"\x00\x00" + text_data

    @staticmethod
    def _encode_id3_comment(text: str, lang: str = "eng") -> bytes:
        """Encode an ID3v2 COMM frame."""
        payload = b"\x03" + lang.encode("ascii")[:3] + b"\x00" + text.encode("utf-8")
        size = struct.pack(">I", len(payload))
        return b"COMM" + size + b"\x00\x00" + payload

    @staticmethod
    def _encode_id3_lyrics(text: str, lang: str = "eng") -> bytes:
        """Encode an ID3v2 USLT frame."""
        payload = b"\x03" + lang.encode("ascii")[:3] + b"\x00" + text.encode("utf-8")
        size = struct.pack(">I", len(payload))
        return b"USLT" + size + b"\x00\x00" + payload

    @staticmethod
    def _encode_id3_picture(image_data: bytes, mime: str = "image/jpeg") -> bytes:
        """Encode an ID3v2 APIC frame."""
        payload = b"\x00"  # Text encoding (Latin-1)
        payload += mime.encode("ascii") + b"\x00"
        payload += b"\x03"  # Cover (front)
        payload += b"\x00"  # Description (empty)
        payload += image_data
        size = struct.pack(">I", len(payload))
        return b"APIC" + size + b"\x00\x00" + payload

    @staticmethod
    def _read_syncsafe_int(data: bytes) -> int:
        """Read a 4-byte syncsafe integer."""
        return (data[0] << 21) | (data[1] << 14) | (data[2] << 7) | data[3]

    @classmethod
    def write_tags(cls, audio_path: str, metadata: TrackMetadata) -> None:
        """Write metadata tags to any supported audio format.

        Tries mutagen first for full format support, falls back to
        pure-Python ID3 for MP3.
        """
        try:
            cls._write_with_mutagen(audio_path, metadata)
        except ImportError:
            path = Path(audio_path)
            if path.suffix.lower() == ".mp3":
                cls.write_id3v2(audio_path, metadata)
            else:
                raise ImportError(
                    f"mutagen is required for {path.suffix} files. "
                    "Install with: pip install mutagen"
                ) from None

    @classmethod
    def _write_with_mutagen(cls, audio_path: str, metadata: TrackMetadata) -> None:
        """Write tags using the mutagen library."""
        import mutagen
        from mutagen.easyid3 import EasyID3
        from mutagen.flac import FLAC, Picture
        from mutagen.id3 import APIC, ID3, USLT
        from mutagen.oggvorbis import OggVorbis

        path = Path(audio_path)
        ext = path.suffix.lower()

        if ext == ".mp3":
            try:
                tags = EasyID3(audio_path)
            except mutagen.id3.ID3NoHeaderError:
                tags = mutagen.File(audio_path, easy=True)
                tags.add_tags()

            tag_map = {
                "title": metadata.title,
                "artist": metadata.artist,
                "albumartist": metadata.album_artist,
                "album": metadata.album,
                "genre": metadata.genre,
                "composer": metadata.composer,
                "date": metadata.release_date or str(metadata.year),
                "isrc": metadata.isrc,
                "bpm": str(int(metadata.bpm)) if metadata.bpm else "",
            }

            if metadata.track_number:
                trck = str(metadata.track_number)
                if metadata.track_total:
                    trck += f"/{metadata.track_total}"
                tag_map["tracknumber"] = trck

            if metadata.disc_number:
                tpos = str(metadata.disc_number)
                if metadata.disc_total:
                    tpos += f"/{metadata.disc_total}"
                tag_map["discnumber"] = tpos

            for key, val in tag_map.items():
                if val:
                    tags[key] = val
            tags.save()

            # Album art and lyrics need raw ID3
            raw_tags = ID3(audio_path)

            art_data = metadata.artwork_data
            if not art_data and metadata.artwork_path:
                art_path = Path(metadata.artwork_path)
                if art_path.exists():
                    art_data = art_path.read_bytes()
            if art_data:
                raw_tags.add(
                    APIC(
                        encoding=3,
                        mime=metadata.artwork_mime,
                        type=3,
                        desc="Cover",
                        data=art_data,
                    )
                )

            if metadata.lyrics:
                raw_tags.add(
                    USLT(
                        encoding=3,
                        lang=metadata.language or "eng",
                        desc="",
                        text=metadata.lyrics,
                    )
                )

            raw_tags.save()

        elif ext == ".flac":
            audio = FLAC(audio_path)
            audio["title"] = metadata.title
            audio["artist"] = metadata.artist
            audio["albumartist"] = metadata.album_artist or metadata.artist
            audio["album"] = metadata.album
            audio["genre"] = metadata.genre
            audio["date"] = metadata.release_date or str(metadata.year)
            if metadata.track_number:
                audio["tracknumber"] = str(metadata.track_number)
            if metadata.track_total:
                audio["tracktotal"] = str(metadata.track_total)
            if metadata.disc_number:
                audio["discnumber"] = str(metadata.disc_number)
            if metadata.composer:
                audio["composer"] = metadata.composer
            if metadata.isrc:
                audio["isrc"] = metadata.isrc
            if metadata.copyright:
                audio["copyright"] = metadata.copyright
            if metadata.lyrics:
                audio["lyrics"] = metadata.lyrics
            if metadata.label:
                audio["label"] = metadata.label

            art_data = metadata.artwork_data
            if not art_data and metadata.artwork_path:
                art_path = Path(metadata.artwork_path)
                if art_path.exists():
                    art_data = art_path.read_bytes()
            if art_data:
                pic = Picture()
                pic.data = art_data
                pic.type = 3  # Front cover
                pic.mime = metadata.artwork_mime
                audio.clear_pictures()
                audio.add_picture(pic)

            audio.save()

        elif ext == ".ogg":
            audio = OggVorbis(audio_path)  # type: ignore[assignment]
            audio["title"] = metadata.title
            audio["artist"] = metadata.artist
            audio["album"] = metadata.album
            audio["genre"] = metadata.genre
            audio["date"] = metadata.release_date or str(metadata.year)
            if metadata.track_number:
                audio["tracknumber"] = str(metadata.track_number)
            if metadata.composer:
                audio["composer"] = metadata.composer
            if metadata.lyrics:
                audio["lyrics"] = metadata.lyrics
            audio.save()

        else:
            raise ValueError(f"Unsupported format for tagging: {ext}")

    @classmethod
    def read_tags(cls, audio_path: str) -> TrackMetadata:
        """Read metadata from an audio file.

        Requires mutagen for full support.
        """
        try:
            import mutagen
        except ImportError:
            raise ImportError("mutagen is required for reading tags. pip install mutagen") from None

        audio = mutagen.File(audio_path, easy=True)
        if audio is None:
            raise ValueError(f"Could not read audio file: {audio_path}")

        def get(key, default=""):
            val = audio.get(key)
            if val:
                return str(val[0]) if isinstance(val, list) else str(val)
            return default

        meta = TrackMetadata(
            title=get("title"),
            artist=get("artist"),
            album_artist=get("albumartist"),
            album=get("album"),
            genre=get("genre"),
            composer=get("composer"),
            release_date=get("date"),
        )

        # Track number
        trck = get("tracknumber")
        if "/" in trck:
            parts = trck.split("/")
            meta.track_number = int(parts[0]) if parts[0].isdigit() else 0
            meta.track_total = int(parts[1]) if parts[1].isdigit() else 0
        elif trck.isdigit():
            meta.track_number = int(trck)

        # Disc number
        tpos = get("discnumber")
        if "/" in tpos:
            parts = tpos.split("/")
            meta.disc_number = int(parts[0]) if parts[0].isdigit() else 1
            meta.disc_total = int(parts[1]) if parts[1].isdigit() else 1
        elif tpos.isdigit():
            meta.disc_number = int(tpos)

        meta.isrc = get("isrc")

        # BPM
        bpm_str = get("bpm")
        if bpm_str:
            with contextlib.suppress(ValueError):
                meta.bpm = float(bpm_str)

        return meta

    @classmethod
    def validate_for_platform(cls, metadata: TrackMetadata, platform: str = "spotify") -> list[str]:
        """Validate metadata against a specific platform's requirements."""
        issues = metadata.validate()
        reqs = cls.PLATFORM_REQUIREMENTS.get(platform.lower())

        if not reqs:
            issues.append(f"WARN: Unknown platform '{platform}'")
            return issues

        if reqs.get("isrc_required") and not metadata.isrc:
            issues.append(f"ERROR: {platform} requires an ISRC code")
        if reqs.get("upc_required") and not metadata.upc:
            issues.append(f"ERROR: {platform} requires a UPC/EAN code")

        return issues

    @classmethod
    def batch_write(cls, tracks: list[tuple[str, TrackMetadata]]) -> dict:
        """Write metadata to multiple files.

        Args:
            tracks: List of (file_path, metadata) tuples.

        Returns:
            Dict with results per file.
        """
        results = {}
        for path, meta in tracks:
            try:
                cls.write_tags(path, meta)
                results[path] = {"status": "ok", "issues": meta.validate()}
            except Exception as e:
                results[path] = {"status": "error", "error": str(e)}
        return results

    @classmethod
    def export_metadata_json(cls, tracks: list[TrackMetadata], output_path: str) -> None:
        """Export all track metadata to a JSON file for distributor upload."""
        data = {
            "format_version": "1.0",
            "track_count": len(tracks),
            "tracks": [t.to_dict() for t in tracks],
        }
        Path(output_path).write_text(json.dumps(data, indent=2, default=str))
