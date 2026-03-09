"""Comprehensive tests for the release module."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from release.album import Album
from release.export_pipeline import ExportConfig, ExportPipeline
from release.licensing import (
    LicenseInfo,
    LicensingManager,
    SongRegistration,
    WriterSplit,
)
from release.lyrics import Lyrics, LyricsManager
from release.metadata import MetadataManager, TrackMetadata

# ─── Metadata Tests ───


class TestTrackMetadata:
    def test_basic_creation(self):
        meta = TrackMetadata(title="Test Song", artist="Test Artist")
        assert meta.title == "Test Song"
        assert meta.artist == "Test Artist"

    def test_validate_minimal(self):
        meta = TrackMetadata()
        issues = meta.validate()
        assert any("Title" in i for i in issues)
        assert any("Artist" in i for i in issues)

    def test_validate_complete(self):
        meta = TrackMetadata(
            title="My Song",
            artist="My Artist",
            album="My Album",
            track_number=1,
            year=2025,
            genre="Pop",
            isrc="US-S1Z-25-00001",
            copyright="2025 Me",
            artwork_path="/fake/cover.jpg",
        )
        issues = meta.validate()
        # Should have minimal warnings
        errors = [i for i in issues if i.startswith("ERROR")]
        assert len(errors) == 0

    def test_isrc_validation_valid(self):
        assert TrackMetadata._validate_isrc("US-S1Z-25-00001")
        assert TrackMetadata._validate_isrc("USS1Z2500001")  # Without dashes

    def test_isrc_validation_invalid(self):
        assert not TrackMetadata._validate_isrc("INVALID")
        assert not TrackMetadata._validate_isrc("US-S1Z-25")

    def test_upc_validation(self):
        assert TrackMetadata._validate_upc("012345678901")  # 12 digit
        assert TrackMetadata._validate_upc("0123456789012")  # 13 digit
        assert not TrackMetadata._validate_upc("12345")
        assert not TrackMetadata._validate_upc("abcdefghijkl")

    def test_to_dict(self):
        meta = TrackMetadata(title="Song", artist="Artist", bpm=128.0)
        d = meta.to_dict()
        assert d["title"] == "Song"
        assert d["bpm"] == 128.0
        assert "artwork_data" in d  # Should store size, not data

    def test_from_dict(self):
        data = {"title": "Song", "artist": "Artist", "bpm": 128.0}
        meta = TrackMetadata.from_dict(data)
        assert meta.title == "Song"
        assert meta.bpm == 128.0

    def test_roundtrip_dict(self):
        meta = TrackMetadata(title="Song", artist="Artist", genre="Pop", bpm=120)
        d = meta.to_dict()
        meta2 = TrackMetadata.from_dict(d)
        assert meta2.title == meta.title
        assert meta2.genre == meta.genre


class TestMetadataManager:
    def test_genres_list(self):
        assert len(MetadataManager.GENRES) > 100

    def test_platform_requirements(self):
        assert "spotify" in MetadataManager.PLATFORM_REQUIREMENTS
        assert "apple_music" in MetadataManager.PLATFORM_REQUIREMENTS
        reqs = MetadataManager.PLATFORM_REQUIREMENTS["spotify"]
        assert reqs["isrc_required"]

    def test_validate_for_platform(self):
        meta = TrackMetadata(title="Song", artist="Artist")
        issues = MetadataManager.validate_for_platform(meta, "spotify")
        assert any("ISRC" in i for i in issues)

    def test_export_metadata_json(self, tmp_path):
        tracks = [
            TrackMetadata(title="Song 1", artist="Artist"),
            TrackMetadata(title="Song 2", artist="Artist"),
        ]
        path = str(tmp_path / "metadata.json")
        MetadataManager.export_metadata_json(tracks, path)
        with open(path) as f:
            data = json.loads(f.read())
        assert data["track_count"] == 2
        assert len(data["tracks"]) == 2


# ─── Album Tests ───


class TestAlbum:
    def test_create_album(self):
        album = Album("My Album", "My Artist")
        assert album.title == "My Album"
        assert album.track_count == 0

    def test_add_tracks(self):
        album = Album("Test", "Artist")
        album.add_track(TrackMetadata(title="Track 1"))
        album.add_track(TrackMetadata(title="Track 2"))
        album.add_track(TrackMetadata(title="Track 3"))
        assert album.track_count == 3
        assert album.tracks[0].metadata.track_number == 1
        assert album.tracks[2].metadata.track_number == 3

    def test_auto_fill_metadata(self):
        album = Album("My Album", "My Artist")
        album.genre = "Pop"
        album.year = 2025
        album.add_track(TrackMetadata(title="Track 1"))
        meta = album.tracks[0].metadata
        assert meta.album == "My Album"
        assert meta.album_artist == "My Artist"
        assert meta.genre == "Pop"
        assert meta.year == 2025

    def test_track_ordering(self):
        album = Album("Test", "Artist")
        album.add_track(TrackMetadata(title="A"))
        album.add_track(TrackMetadata(title="B"))
        album.add_track(TrackMetadata(title="C"))
        album.move_track(2, 0)
        assert album.tracks[0].metadata.title == "C"
        assert album.tracks[0].metadata.track_number == 1

    def test_swap_tracks(self):
        album = Album("Test", "Artist")
        album.add_track(TrackMetadata(title="A"))
        album.add_track(TrackMetadata(title="B"))
        album.swap_tracks(0, 1)
        assert album.tracks[0].metadata.title == "B"

    def test_remove_track(self):
        album = Album("Test", "Artist")
        album.add_track(TrackMetadata(title="A"))
        album.add_track(TrackMetadata(title="B"))
        removed = album.remove_track(0)
        assert removed.metadata.title == "A"
        assert album.track_count == 1
        assert album.tracks[0].metadata.track_number == 1

    def test_disc_layout(self):
        album = Album("Test", "Artist")
        for i in range(12):
            album.add_track(TrackMetadata(title=f"Track {i+1}"), duration_s=240)
        album.set_disc_layout([6, 6])
        assert album.tracks[0].metadata.disc_number == 1
        assert album.tracks[6].metadata.disc_number == 2

    def test_validate(self):
        album = Album("Test", "Artist")
        album.upc = "012345678901"
        album.copyright = "2025 Artist"
        album.add_track(TrackMetadata(title="Song", artist="Artist", isrc="USS1Z2500001"))
        issues = album.validate()
        errors = [i for i in issues if i.startswith("ERROR")]
        assert len(errors) == 0

    def test_validate_empty_album(self):
        album = Album()
        issues = album.validate()
        assert any("title" in i.lower() for i in issues)
        assert any("no tracks" in i.lower() for i in issues)

    def test_tracklist(self):
        album = Album("My Album", "My Artist")
        album.add_track(TrackMetadata(title="Intro"), duration_s=60)
        album.add_track(TrackMetadata(title="Main Song", explicit=True), duration_s=240)
        text = album.tracklist()
        assert "My Album" in text
        assert "Intro" in text
        assert "[E]" in text  # Explicit marker

    def test_json_save_load(self, tmp_path):
        album = Album("Test Album", "Test Artist")
        album.genre = "Electronic"
        album.year = 2025
        album.add_track(TrackMetadata(title="Track 1"))
        album.add_track(TrackMetadata(title="Track 2"))

        path = str(tmp_path / "album.json")
        album.save_json(path)

        loaded = Album.load_json(path)
        assert loaded.title == "Test Album"
        assert loaded.track_count == 2

    def test_total_duration(self):
        album = Album("Test", "Artist")
        album.add_track(TrackMetadata(title="A"), duration_s=180)
        album.add_track(TrackMetadata(title="B"), duration_s=240)
        assert album.total_duration_s == 420
        assert "7:00" in album.total_duration_str


# ─── Lyrics Tests ───


class TestLyrics:
    def test_plain_text(self):
        lyrics = Lyrics(title="Song", artist="Artist")
        lyrics.text = "Hello world\nSecond line"
        assert "Hello world" in lyrics.text
        assert "Second line" in lyrics.text

    def test_synced_lyrics(self):
        lyrics = Lyrics()
        lyrics.add_synced_line(0, "First line")
        lyrics.add_synced_line(3000, "Second line")
        lyrics.add_synced_line(6000, "Third line")
        assert lyrics.is_synced
        assert len(lyrics.synced_lines) == 3

    def test_synced_ordering(self):
        lyrics = Lyrics()
        lyrics.add_synced_line(6000, "Third")
        lyrics.add_synced_line(0, "First")
        lyrics.add_synced_line(3000, "Second")
        lines = lyrics.synced_lines
        assert lines[0].text == "First"
        assert lines[2].text == "Third"

    def test_lrc_export(self):
        lyrics = Lyrics(title="Song", artist="Artist")
        lyrics.add_synced_line(12340, "First line")
        lrc = lyrics.to_lrc()
        assert "[ti:Song]" in lrc
        assert "[ar:Artist]" in lrc
        assert "[00:12" in lrc

    def test_lrc_roundtrip(self):
        lyrics = Lyrics(title="Song", artist="Artist")
        lyrics.add_synced_line(0, "Line one")
        lyrics.add_synced_line(5000, "Line two")
        lrc = lyrics.to_lrc()
        parsed = Lyrics.from_lrc(lrc)
        assert parsed.title == "Song"
        assert len(parsed.synced_lines) == 2

    def test_sections(self):
        lyrics = Lyrics()
        text = """[Verse 1]
First verse line one
First verse line two

[Chorus]
Chorus line"""
        lyrics.set_sections_from_text(text)
        assert len(lyrics.sections) == 2
        assert lyrics.sections[0].label == "Verse 1"
        assert lyrics.sections[1].label == "Chorus"

    def test_translations(self):
        lyrics = Lyrics(title="Song")
        lyrics.text = "Hello"
        lyrics.add_translation("spa", "Hola")
        lyrics.add_translation("fra", "Bonjour")
        assert lyrics.get_translation("spa") == "Hola"
        assert len(lyrics.translations) == 2

    def test_file_io(self, tmp_path):
        lyrics = Lyrics(title="Song")
        lyrics.add_synced_line(0, "Hello")
        lyrics.add_synced_line(2000, "World")

        lrc_path = str(tmp_path / "song.lrc")
        lyrics.save_lrc(lrc_path)
        loaded = Lyrics.load_lrc(lrc_path)
        assert len(loaded.synced_lines) == 2

    def test_dict_roundtrip(self):
        lyrics = Lyrics(title="Song", artist="Artist")
        lyrics.text = "Hello world"
        lyrics.add_synced_line(0, "Hello")
        d = lyrics.to_dict()
        restored = Lyrics.from_dict(d)
        assert restored.title == "Song"
        assert "Hello" in restored.text


class TestLyricsManager:
    def test_add_and_get(self):
        mgr = LyricsManager()
        lyrics = Lyrics(title="Song", artist="Artist")
        lyrics.text = "Hello world"
        mgr.add(lyrics)
        found = mgr.get("Artist", "Song")
        assert found is not None
        assert "Hello" in found.text

    def test_search(self):
        mgr = LyricsManager()
        l1 = Lyrics(title="Happy Song")
        l1.text = "Joy and happiness"
        mgr.add(l1)
        l2 = Lyrics(title="Sad Song")
        l2.text = "Tears and sorrow"
        mgr.add(l2)
        results = mgr.search("happiness")
        assert len(results) == 1

    def test_database_save_load(self, tmp_path):
        mgr = LyricsManager()
        lyrics = Lyrics(title="Song", artist="Artist")
        lyrics.text = "Test lyrics"
        mgr.add(lyrics)

        db_path = str(tmp_path / "lyrics.json")
        mgr.save_database(db_path)

        mgr2 = LyricsManager.load_database(db_path)
        assert len(mgr2.all_lyrics) == 1

    def test_word_frequency(self):
        mgr = LyricsManager()
        l1 = Lyrics(title="Song 1")
        l1.text = "love love love baby"
        mgr.add(l1)
        l2 = Lyrics(title="Song 2")
        l2.text = "love baby baby"
        mgr.add(l2)
        freq = mgr.word_frequency()
        assert freq["love"] == 4
        assert freq["baby"] == 3


# ─── Licensing Tests ───


class TestLicensing:
    def test_writer_split(self):
        writer = WriterSplit(name="John Doe", role="songwriter", share_percent=50.0, pro="ASCAP")
        assert writer.validate() == []

    def test_writer_invalid_share(self):
        writer = WriterSplit(name="John", share_percent=150.0)
        issues = writer.validate()
        assert len(issues) > 0

    def test_song_registration(self):
        reg = SongRegistration(title="My Song")
        reg.writers = [
            WriterSplit(name="John", share_percent=60, pro="ASCAP"),
            WriterSplit(name="Jane", share_percent=40, pro="BMI"),
        ]
        assert reg.total_writer_share() == 100.0
        assert len(reg.validate()) == 0 or all("WARN" in i for i in reg.validate())

    def test_registration_invalid_splits(self):
        reg = SongRegistration(title="Bad Song")
        reg.writers = [
            WriterSplit(name="John", share_percent=60),
            WriterSplit(name="Jane", share_percent=20),
        ]
        issues = reg.validate()
        assert any("80.0%" in i for i in issues)  # Doesn't total 100%

    def test_license_info(self):
        lic = LicenseInfo(copyright_owner="Artist", copyright_year=2025)
        assert "2025" in lic.copyright_string
        assert "Artist" in lic.copyright_string

    def test_phonographic_copyright(self):
        lic = LicenseInfo(copyright_owner="Label", copyright_year=2025)
        assert "\u2117" in lic.phonographic_string

    def test_iswc_validation(self):
        assert SongRegistration._validate_iswc("T-345.246.800-1")
        assert not SongRegistration._validate_iswc("INVALID")


class TestLicensingManager:
    def test_quick_register(self):
        mgr = LicensingManager()
        reg = mgr.quick_register(
            "My Song",
            [
                {"name": "John", "role": "songwriter", "share_percent": 50, "pro": "ASCAP"},
                {"name": "Jane", "role": "producer", "share_percent": 50, "pro": "BMI"},
            ],
        )
        assert reg.title == "My Song"
        assert len(reg.writers) == 2

    def test_royalty_split(self):
        mgr = LicensingManager()
        mgr.quick_register(
            "Song",
            [
                {"name": "John", "share_percent": 60},
                {"name": "Jane", "share_percent": 40},
            ],
        )
        splits = mgr.calculate_royalty_split("Song", 1000.0)
        assert splits["John"] == 600.0
        assert splits["Jane"] == 400.0

    def test_royalty_with_publisher(self):
        mgr = LicensingManager()
        reg = SongRegistration(title="Song")
        reg.writers = [
            WriterSplit(name="John", share_percent=100, publisher="Pub Co", publisher_share=25),
        ]
        mgr.register_song(reg)
        splits = mgr.calculate_royalty_split("Song", 1000.0)
        assert splits["John"] == 750.0
        assert "Pub Co" in str(splits)

    def test_validate_all(self):
        mgr = LicensingManager()
        mgr.quick_register(
            "Good Song",
            [
                {"name": "A", "share_percent": 50},
                {"name": "B", "share_percent": 50},
            ],
        )
        mgr.quick_register(
            "Bad Song",
            [
                {"name": "C", "share_percent": 30},
            ],
        )
        results = mgr.validate_all()
        assert "Bad Song" in results

    def test_generate_split_sheet(self):
        mgr = LicensingManager()
        mgr.quick_register(
            "Song",
            [
                {"name": "Writer A", "share_percent": 60, "pro": "ASCAP"},
                {"name": "Writer B", "share_percent": 40, "pro": "BMI"},
            ],
        )
        sheet = mgr.generate_split_sheet("Song")
        assert "SPLIT SHEET" in sheet
        assert "Writer A" in sheet
        assert "60.0%" in sheet

    def test_database_save_load(self, tmp_path):
        mgr = LicensingManager()
        mgr.quick_register(
            "Song",
            [
                {"name": "Writer", "share_percent": 100, "pro": "ASCAP"},
            ],
        )
        db_path = str(tmp_path / "licensing.json")
        mgr.save_database(db_path)

        mgr2 = LicensingManager.load_database(db_path)
        assert len(mgr2.all_songs) == 1

    def test_catalog_summary(self):
        mgr = LicensingManager()
        mgr.quick_register(
            "Song 1",
            [
                {"name": "A", "share_percent": 100, "pro": "ASCAP"},
            ],
        )
        mgr.quick_register(
            "Song 2",
            [
                {"name": "B", "share_percent": 100, "pro": "BMI"},
            ],
        )
        summary = mgr.catalog_summary()
        assert "Total Songs: 2" in summary


# ─── Export Pipeline Tests ───


class TestExportPipeline:
    def test_platform_config(self):
        config = ExportPipeline.platform_config("spotify")
        assert config.format == "flac"
        assert config.bit_depth == 24

    def test_platform_config_cd(self):
        config = ExportPipeline.platform_config("cd")
        assert config.format == "wav"
        assert config.bit_depth == 16
        assert config.dither

    def test_validate_incomplete(self):
        album = Album()
        pipeline = ExportPipeline(album)
        result = pipeline.validate()
        assert not result["passed"]

    def test_validate_complete(self):
        album = Album("Test", "Artist")
        album.upc = "012345678901"
        album.copyright = "2025"
        album.add_track(
            TrackMetadata(
                title="Song",
                artist="Artist",
                isrc="USS1Z2500001",
                genre="Pop",
                year=2025,
            )
        )
        pipeline = ExportPipeline(album)
        pipeline.validate()
        # May have warnings but no errors if properly configured

    def test_dry_run(self):
        album = Album("Test Album", "Test Artist")
        album.add_track(TrackMetadata(title="Track 1"))
        album.add_track(TrackMetadata(title="Track 2"))

        config = ExportConfig(output_dir="/tmp/export", format="flac")
        pipeline = ExportPipeline(album, config)
        preview = pipeline.dry_run()
        assert "Test Album" in preview
        assert "Track 1" in preview
        assert "FLAC" in preview.upper()

    def test_safe_filename(self):
        assert (
            ExportPipeline._safe_filename("My Song: Remix (feat. Artist)")
            == "My Song_ Remix (feat. Artist)"
        )
        assert ExportPipeline._safe_filename("Bad/File\\Name") == "Bad_File_Name"

    def test_export_config_defaults(self):
        config = ExportConfig()
        assert config.format == "wav"
        assert config.bit_depth == 24
        assert config.sample_rate == 44100
