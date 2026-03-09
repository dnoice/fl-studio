"""Export Pipeline - Full album export and distribution preparation.

Orchestrates the complete release workflow: audio rendering, metadata
tagging, lyrics embedding, licensing validation, artwork attachment,
and platform-specific package generation.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from release.album import Album, AlbumTrack
from release.licensing import LicensingManager
from release.lyrics import LyricsManager
from release.metadata import MetadataManager


@dataclass
class ExportConfig:
    """Configuration for the export pipeline."""

    # Output settings
    output_dir: str = ""
    format: str = "wav"  # wav, flac, mp3, ogg, aiff
    bit_depth: int = 24  # 16, 24, 32
    sample_rate: int = 44100  # 44100, 48000, 88200, 96000
    mp3_bitrate: int = 320  # kbps (for MP3 only)

    # Processing
    normalize: bool = False
    normalize_target_db: float = -1.0
    dither: bool = True  # Apply dither when reducing bit depth
    trim_silence: bool = False
    trim_threshold_db: float = -60.0

    # Metadata
    embed_artwork: bool = True
    embed_lyrics: bool = True
    artwork_max_size_px: int = 3000

    # Organization
    folder_structure: str = "{artist}/{album}"
    file_naming: str = "{track:02d} - {title}"
    create_m3u: bool = True  # Generate playlist file
    create_cue: bool = False  # Generate CUE sheet

    # Platform presets
    platform: str = ""  # spotify, apple_music, bandcamp, etc.

    # Validation
    validate_metadata: bool = True
    validate_licensing: bool = True
    strict_validation: bool = False  # Fail on warnings


class ExportPipeline:
    """Orchestrates the full album export and distribution workflow.

    Typical usage:
        album = Album("My Album", "My Artist")
        album.add_track(TrackMetadata(title="Track 1"), audio_path="track1.wav")

        config = ExportConfig(
            output_dir="/path/to/export",
            format="flac",
            platform="spotify",
        )

        pipeline = ExportPipeline(album, config)
        pipeline.set_lyrics_manager(lyrics_mgr)
        pipeline.set_licensing_manager(licensing_mgr)

        report = pipeline.execute()
    """

    def __init__(self, album: Album, config: ExportConfig | None = None):
        self.album = album
        self.config = config or ExportConfig()
        self._lyrics_mgr: LyricsManager | None = None
        self._licensing_mgr: LicensingManager | None = None
        self._log: list[str] = []

    def set_lyrics_manager(self, mgr: LyricsManager) -> None:
        self._lyrics_mgr = mgr

    def set_licensing_manager(self, mgr: LicensingManager) -> None:
        self._licensing_mgr = mgr

    def _log_msg(self, msg: str) -> None:
        self._log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    # ─── Platform Presets ───

    @staticmethod
    def platform_config(platform: str) -> ExportConfig:
        """Get export config optimized for a specific platform."""
        configs = {
            "spotify": ExportConfig(
                format="flac",
                bit_depth=24,
                sample_rate=44100,
                embed_artwork=True,
                embed_lyrics=True,
                artwork_max_size_px=3000,
                platform="spotify",
            ),
            "apple_music": ExportConfig(
                format="flac",
                bit_depth=24,
                sample_rate=44100,
                embed_artwork=True,
                embed_lyrics=True,
                artwork_max_size_px=3000,
                platform="apple_music",
            ),
            "bandcamp": ExportConfig(
                format="flac",
                bit_depth=24,
                sample_rate=44100,
                embed_artwork=True,
                embed_lyrics=True,
                artwork_max_size_px=1400,
                platform="bandcamp",
            ),
            "soundcloud": ExportConfig(
                format="wav",
                bit_depth=24,
                sample_rate=44100,
                embed_artwork=True,
                platform="soundcloud",
            ),
            "cd": ExportConfig(
                format="wav",
                bit_depth=16,
                sample_rate=44100,
                dither=True,
                create_cue=True,
                platform="cd",
            ),
            "vinyl": ExportConfig(
                format="wav",
                bit_depth=24,
                sample_rate=96000,
                platform="vinyl",
            ),
            "mp3_archive": ExportConfig(
                format="mp3",
                mp3_bitrate=320,
                embed_artwork=True,
                embed_lyrics=True,
                create_m3u=True,
                platform="mp3_archive",
            ),
        }
        return configs.get(platform.lower(), ExportConfig(platform=platform))

    # ─── Validation ───

    def validate(self) -> dict[str, object]:
        """Run all validation checks before export.

        Returns:
            Dict with validation results.
        """
        tracks_issues: dict[str, list[str]] = {}
        licensing_issues: dict[str, list[str]] = {}

        # Album validation
        album_issues = self.album.validate()

        # Platform-specific validation
        if self.config.platform:
            for track in self.album.tracks:
                track_issues = MetadataManager.validate_for_platform(
                    track.metadata, self.config.platform
                )
                if track_issues:
                    tracks_issues[track.metadata.title] = track_issues

        # Licensing validation
        if self.config.validate_licensing and self._licensing_mgr:
            licensing_issues = self._licensing_mgr.validate_all()

        # Check for errors
        all_issues: list[str] = list(album_issues)
        for ti in tracks_issues.values():
            all_issues.extend(ti)

        has_errors = any("ERROR" in i for i in all_issues)
        has_warnings = any("WARN" in i for i in all_issues)

        passed = not (has_errors or (has_warnings and self.config.strict_validation))

        results: dict[str, object] = {
            "album": album_issues,
            "tracks": tracks_issues,
            "licensing": licensing_issues,
            "passed": passed,
        }
        return results

    # ─── Export Execution ───

    def execute(self) -> dict[str, object]:
        """Run the full export pipeline.

        Returns:
            Dict with export results and any issues.
        """
        self._log.clear()
        self._log_msg("Starting export pipeline")

        # Setup output directory
        output_dir = Path(self.config.output_dir)
        if not output_dir.is_absolute():
            output_dir = Path.cwd() / output_dir

        folder = self.config.folder_structure.format(
            artist=self._safe_filename(self.album.artist),
            album=self._safe_filename(self.album.title),
            year=self.album.year,
        )
        export_path = output_dir / folder
        export_path.mkdir(parents=True, exist_ok=True)
        self._log_msg(f"Output: {export_path}")

        # Validation
        if self.config.validate_metadata:
            self._log_msg("Validating metadata...")
            validation = self.validate()
            if not validation["passed"]:
                self._log_msg("VALIDATION FAILED - check issues")
                return {
                    "status": "validation_failed",
                    "validation": validation,
                    "log": list(self._log),
                }

        # Process each track
        exported_files = []
        errors = []

        for i, track in enumerate(self.album.tracks):
            try:
                result = self._export_track(track, export_path, i)
                exported_files.append(result)
                self._log_msg(f"Exported: {result['filename']}")
            except Exception as e:
                error_msg = f"Track {i+1} ({track.metadata.title}): {e}"
                errors.append(error_msg)
                self._log_msg(f"ERROR: {error_msg}")

        # Copy artwork
        if self.album.artwork_path:
            art_src = Path(self.album.artwork_path)
            if art_src.exists():
                art_dst = export_path / f"cover{art_src.suffix}"
                shutil.copy2(str(art_src), str(art_dst))
                self._log_msg(f"Artwork: {art_dst.name}")

        # Generate playlist
        if self.config.create_m3u and exported_files:
            self._create_m3u(export_path, exported_files)

        # Generate CUE sheet
        if self.config.create_cue and exported_files:
            self._create_cue(export_path, exported_files)

        # Export metadata JSON
        self._export_metadata_json(export_path)

        # Export lyrics files
        if self._lyrics_mgr:
            self._export_lyrics(export_path)

        # Export licensing info
        if self._licensing_mgr:
            self._export_licensing(export_path)

        self._log_msg(f"Export complete: {len(exported_files)} tracks")

        return {
            "status": "success" if not errors else "partial",
            "output_dir": str(export_path),
            "tracks_exported": len(exported_files),
            "files": exported_files,
            "errors": errors,
            "log": list(self._log),
        }

    def _export_track(self, track: AlbumTrack, output_dir: Path, index: int) -> dict[str, object]:
        """Export a single track."""
        meta = track.metadata

        # Generate filename
        filename = self.config.file_naming.format(
            track=meta.track_number,
            title=self._safe_filename(meta.title),
            artist=self._safe_filename(meta.artist),
            album=self._safe_filename(meta.album),
        )

        ext = self.config.format.lower()
        if ext == "mp3":
            ext = "mp3"
        output_file = output_dir / f"{filename}.{ext}"

        # Read source audio
        if track.audio_path and Path(track.audio_path).exists():
            audio, sr = sf.read(track.audio_path, dtype="float64")
        else:
            raise FileNotFoundError(f"Audio file not found: {track.audio_path}")

        # Apply processing
        if self.config.trim_silence:
            audio = self._trim_silence(audio, self.config.trim_threshold_db)

        if self.config.normalize:
            peak = np.max(np.abs(audio))
            if peak > 0:
                target = 10 ** (self.config.normalize_target_db / 20)
                audio = audio * (target / peak)

        # Dither if reducing bit depth
        source_bits = 32  # float64 source
        if self.config.dither and self.config.bit_depth < source_bits:
            audio = self._apply_dither(audio, self.config.bit_depth)

        # Determine subtype for soundfile
        subtype = self._get_subtype(self.config.format, self.config.bit_depth)

        # Write output
        if self.config.format.lower() == "mp3":
            # soundfile doesn't support MP3 writing, write WAV first
            temp_wav = output_dir / f"_temp_{filename}.wav"
            sf.write(str(temp_wav), audio, self.config.sample_rate, subtype="PCM_24")
            # Use external encoder or just keep wav
            # For now, save as WAV with mp3 extension note
            output_file = output_dir / f"{filename}.wav"
            shutil.move(str(temp_wav), str(output_file))
            self._log_msg("  Note: MP3 encoding requires lame or ffmpeg")
        else:
            sf.write(str(output_file), audio, self.config.sample_rate, subtype=subtype)

        # Embed metadata
        if self.config.embed_artwork and self.album.artwork_path:
            art_path = Path(self.album.artwork_path)
            if art_path.exists() and not meta.artwork_data:
                meta.artwork_path = str(art_path)

        # Embed lyrics
        if self.config.embed_lyrics and self._lyrics_mgr:
            lyrics = self._lyrics_mgr.get(meta.artist, meta.title)
            if lyrics:
                meta.lyrics = lyrics.text
                if lyrics.is_synced:
                    meta.synced_lyrics = lyrics.to_lrc()

        try:
            MetadataManager.write_tags(str(output_file), meta)
        except (ImportError, ValueError) as e:
            self._log_msg(f"  Metadata warning: {e}")

        return {
            "track_number": meta.track_number,
            "title": meta.title,
            "filename": output_file.name,
            "path": str(output_file),
            "format": self.config.format,
            "duration_s": len(audio) / self.config.sample_rate,
        }

    # ─── Helpers ───

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Make a string safe for use as a filename."""
        unsafe = '<>:"/\\|?*'
        result = name
        for char in unsafe:
            result = result.replace(char, "_")
        return result.strip(". ")

    @staticmethod
    def _trim_silence(audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """Trim silence from start and end."""
        threshold = 10 ** (threshold_db / 20)
        amplitude = np.max(np.abs(audio), axis=1) if audio.ndim > 1 else np.abs(audio)

        above = np.where(amplitude > threshold)[0]
        if len(above) == 0:
            return audio

        return audio[above[0] : above[-1] + 1]

    @staticmethod
    def _apply_dither(audio: np.ndarray, target_bits: int) -> np.ndarray:
        """Apply TPDF dither for bit depth reduction."""
        # TPDF (Triangular Probability Density Function) dither
        lsb = 1.0 / (2 ** (target_bits - 1))
        noise1 = np.random.uniform(-lsb / 2, lsb / 2, audio.shape)
        noise2 = np.random.uniform(-lsb / 2, lsb / 2, audio.shape)
        return audio + noise1 + noise2

    @staticmethod
    def _get_subtype(fmt: str, bit_depth: int) -> str:
        """Get soundfile subtype string."""
        fmt = fmt.lower()
        if fmt == "wav":
            return {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bit_depth, "PCM_24")
        elif fmt == "flac":
            return {16: "PCM_16", 24: "PCM_24"}.get(bit_depth, "PCM_24")
        elif fmt == "aiff":
            return {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bit_depth, "PCM_24")
        elif fmt == "ogg":
            return "VORBIS"
        return "PCM_24"

    def _create_m3u(self, output_dir: Path, files: list[dict]) -> None:
        """Generate M3U playlist file."""
        lines = ["#EXTM3U"]
        for f in files:
            duration = int(f.get("duration_s", 0))
            lines.append(f"#EXTINF:{duration},{f['title']}")
            lines.append(f["filename"])

        playlist_name = self._safe_filename(self.album.title)
        m3u_path = output_dir / f"{playlist_name}.m3u"
        m3u_path.write_text("\n".join(lines), encoding="utf-8")
        self._log_msg(f"Playlist: {m3u_path.name}")

    def _create_cue(self, output_dir: Path, files: list[dict]) -> None:
        """Generate CUE sheet."""
        lines = [
            f'PERFORMER "{self.album.artist}"',
            f'TITLE "{self.album.title}"',
        ]

        offset_ms = 0
        for f in files:
            track_num = f["track_number"]
            lines.extend(
                [
                    f"  TRACK {track_num:02d} AUDIO",
                    f'    TITLE "{f["title"]}"',
                    f'    PERFORMER "{self.album.artist}"',
                    f"    INDEX 01 {self._ms_to_cue_time(offset_ms)}",
                ]
            )
            offset_ms += int(f.get("duration_s", 0) * 1000)

        cue_name = self._safe_filename(self.album.title)
        cue_path = output_dir / f"{cue_name}.cue"
        cue_path.write_text("\n".join(lines), encoding="utf-8")
        self._log_msg(f"CUE sheet: {cue_path.name}")

    @staticmethod
    def _ms_to_cue_time(ms: int) -> str:
        """Convert milliseconds to CUE time format (MM:SS:FF)."""
        minutes = ms // 60000
        seconds = (ms % 60000) // 1000
        frames = ((ms % 1000) * 75) // 1000  # 75 frames per second
        return f"{minutes:02d}:{seconds:02d}:{frames:02d}"

    def _export_metadata_json(self, output_dir: Path) -> None:
        """Export complete metadata as JSON."""
        data = self.album.to_dict()
        json_path = output_dir / "metadata.json"
        json_path.write_text(json.dumps(data, indent=2, default=str))
        self._log_msg(f"Metadata: {json_path.name}")

    def _export_lyrics(self, output_dir: Path) -> None:
        """Export lyrics files alongside audio."""
        assert self._lyrics_mgr is not None
        lyrics_dir = output_dir / "lyrics"
        lyrics_dir.mkdir(exist_ok=True)

        count = 0
        for track in self.album.tracks:
            meta = track.metadata
            lyrics = self._lyrics_mgr.get(meta.artist, meta.title)
            if lyrics:
                safe_name = self._safe_filename(f"{meta.track_number:02d} - {meta.title}")
                if lyrics.is_synced:
                    lyrics.save_lrc(str(lyrics_dir / f"{safe_name}.lrc"))
                lyrics.save_txt(str(lyrics_dir / f"{safe_name}.txt"))
                count += 1

        if count:
            self._log_msg(f"Lyrics: {count} files")

    def _export_licensing(self, output_dir: Path) -> None:
        """Export licensing documentation."""
        assert self._licensing_mgr is not None
        lic_dir = output_dir / "licensing"
        lic_dir.mkdir(exist_ok=True)

        # Split sheets for each track
        for track in self.album.tracks:
            reg = self._licensing_mgr.get_song(track.metadata.title)
            if reg:
                safe_name = self._safe_filename(track.metadata.title)
                sheet = self._licensing_mgr.generate_split_sheet(track.metadata.title)
                (lic_dir / f"split_sheet_{safe_name}.txt").write_text(sheet)

        # Catalog summary
        summary = self._licensing_mgr.catalog_summary()
        (lic_dir / "catalog_summary.txt").write_text(summary)

        # Full licensing database
        self._licensing_mgr.save_database(str(lic_dir / "licensing_database.json"))

        self._log_msg("Licensing docs exported")

    def dry_run(self) -> str:
        """Preview what the export would do without writing files.

        Returns:
            Formatted preview string.
        """
        lines = [
            "=== Export Pipeline Dry Run ===",
            f"Album: {self.album.title} by {self.album.artist}",
            f"Tracks: {self.album.track_count}",
            f"Format: {self.config.format.upper()} {self.config.bit_depth}bit / {self.config.sample_rate}Hz",
            f"Platform: {self.config.platform or 'Generic'}",
            "",
        ]

        # Output structure
        folder = self.config.folder_structure.format(
            artist=self._safe_filename(self.album.artist),
            album=self._safe_filename(self.album.title),
            year=self.album.year,
        )
        lines.append(f"Output: {self.config.output_dir}/{folder}/")
        lines.append("")

        # Track listing
        lines.append("Files to be created:")
        for track in self.album.tracks:
            meta = track.metadata
            filename = self.config.file_naming.format(
                track=meta.track_number,
                title=self._safe_filename(meta.title),
                artist=self._safe_filename(meta.artist),
                album=self._safe_filename(meta.album),
            )
            lines.append(f"  {filename}.{self.config.format}")

        lines.append("")

        # Validation preview
        validation = self.validate()
        album_validation: list[str] = list(validation.get("album", []))  # type: ignore[call-overload]
        if album_validation:
            lines.append("Validation Issues:")
            for issue in album_validation:
                lines.append(f"  {issue}")

        # Extras
        extras = []
        if self.config.create_m3u:
            extras.append("M3U playlist")
        if self.config.create_cue:
            extras.append("CUE sheet")
        if self.config.embed_artwork:
            extras.append("Embedded artwork")
        if self.config.embed_lyrics:
            extras.append("Embedded lyrics")
        if extras:
            lines.append(f"\nExtras: {', '.join(extras)}")

        return "\n".join(lines)
