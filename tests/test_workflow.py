"""Comprehensive tests for the workflow module."""

import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import soundfile as sf

from workflow.flp_parser import FLPParser, FLPProject
from workflow.preset_manager import PresetManager
from workflow.project_backup import ProjectBackup
from workflow.render_queue import RenderQueue
from workflow.sample_organizer import SampleOrganizer

# ─── FLP Parser Tests ───


@pytest.fixture
def fake_flp(tmp_path):
    """Create a minimal valid FLP file for testing."""
    path = tmp_path / "test.flp"

    with open(path, "wb") as f:
        # FLhd header
        f.write(b"FLhd")
        f.write(struct.pack("<I", 6))  # Header length
        f.write(struct.pack("<H", 0))  # Format type
        f.write(struct.pack("<H", 1))  # Num channels
        f.write(struct.pack("<H", 96))  # PPQ

        # Build event data
        events = bytearray()

        # Tempo event (word): event_id=66, value=140
        events.append(66)
        events.extend(struct.pack("<H", 140))

        # Version text event: event_id=199
        version_text = "FL Studio 2025".encode("utf-16-le")
        events.append(199)
        events.append(len(version_text))  # varint length
        events.extend(version_text)

        # FLdt chunk
        f.write(b"FLdt")
        f.write(struct.pack("<I", len(events)))
        f.write(events)

    return str(path)


class TestFLPParser:
    def test_parse_valid_flp(self, fake_flp):
        project = FLPParser.parse(fake_flp)
        assert isinstance(project, FLPProject)
        assert project.tempo == 140

    def test_parse_version(self, fake_flp):
        project = FLPParser.parse(fake_flp)
        assert "FL Studio" in project.version or project.version == ""

    def test_parse_invalid_file(self, tmp_path):
        bad_file = tmp_path / "bad.flp"
        bad_file.write_bytes(b"NOT_FLP_DATA")
        with pytest.raises(ValueError):
            FLPParser.parse(str(bad_file))

    def test_project_summary(self, fake_flp):
        project = FLPParser.parse(fake_flp)
        summary = project.summary()
        assert "FL Studio Project" in summary

    def test_list_plugins(self, fake_flp):
        plugins = FLPParser.list_plugins(fake_flp)
        assert isinstance(plugins, list)

    def test_get_tempo(self, fake_flp):
        tempo = FLPParser.get_tempo(fake_flp)
        assert tempo == 140

    def test_batch_info(self, tmp_path, fake_flp):
        results = FLPParser.batch_info(tmp_path)
        assert len(results) >= 1


# ─── Preset Manager Tests ───


@pytest.fixture
def preset_dir(tmp_path):
    """Create a mock preset directory."""
    plugin_dir = tmp_path / "Sytrus" / "Pads"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "Warm Pad.fst").write_bytes(b"\x00" * 100)
    (plugin_dir / "Dark Pad.fst").write_bytes(b"\x00" * 200)

    plugin_dir2 = tmp_path / "Harmor" / "Bass"
    plugin_dir2.mkdir(parents=True)
    (plugin_dir2 / "Sub Bass.fst").write_bytes(b"\x00" * 150)

    return str(tmp_path)


class TestPresetManager:
    def test_scan(self, preset_dir):
        pm = PresetManager(preset_dirs=[preset_dir])
        count = pm.scan()
        assert count == 3

    def test_search_by_name(self, preset_dir):
        pm = PresetManager(preset_dirs=[preset_dir])
        pm.scan()
        results = pm.search("Warm")
        assert len(results) == 1
        assert results[0].name == "Warm Pad"

    def test_search_by_plugin(self, preset_dir):
        pm = PresetManager(preset_dirs=[preset_dir])
        pm.scan()
        results = pm.search(plugin="Sytrus")
        assert len(results) == 2

    def test_list_plugins(self, preset_dir):
        pm = PresetManager(preset_dirs=[preset_dir])
        pm.scan()
        plugins = pm.list_plugins()
        assert "Sytrus" in plugins
        assert "Harmor" in plugins

    def test_list_categories(self, preset_dir):
        pm = PresetManager(preset_dirs=[preset_dir])
        pm.scan()
        cats = pm.list_categories()
        assert "Pads" in cats

    def test_tagging(self, preset_dir, tmp_path):
        pm = PresetManager(preset_dirs=[preset_dir])
        pm.scan()
        presets = pm.search("Warm")
        pm.add_tag(presets[0].filepath, "favorite")
        assert "favorite" in presets[0].tags

        pm.remove_tag(presets[0].filepath, "favorite")
        assert "favorite" not in presets[0].tags

    def test_save_load_tags(self, preset_dir, tmp_path):
        pm = PresetManager(preset_dirs=[preset_dir])
        pm.scan()
        presets = pm.search("Dark")
        pm.add_tag(presets[0].filepath, "dark")

        tags_file = str(tmp_path / "tags.json")
        pm.save_tags(tags_file)

        pm2 = PresetManager(preset_dirs=[preset_dir])
        pm2.scan()
        pm2.load_tags(tags_file)
        results = pm2.search(tags=["dark"])
        assert len(results) == 1

    def test_stats(self, preset_dir):
        pm = PresetManager(preset_dirs=[preset_dir])
        pm.scan()
        stats = pm.stats()
        assert stats["total_presets"] == 3
        assert stats["plugins"] == 2


# ─── Sample Organizer Tests ───


@pytest.fixture
def sample_dir(tmp_path):
    """Create a directory with test audio samples."""
    sr = 44100
    for name in [
        "kick_01.wav",
        "snare_hard.wav",
        "hihat_closed.wav",
        "bass_deep.wav",
        "lead_synth.wav",
        "random_sound.wav",
    ]:
        audio = np.random.randn(sr // 4).astype(np.float32) * 0.5
        sf.write(str(tmp_path / name), audio, sr)
    return str(tmp_path)


class TestSampleOrganizer:
    def test_analyze_sample(self, sample_dir):
        path = os.path.join(sample_dir, "kick_01.wav")
        info = SampleOrganizer.analyze_sample(path)
        assert info.category == "kick"
        assert info.sample_rate == 44100
        assert info.duration_ms > 0

    def test_categorization(self):
        assert SampleOrganizer._categorize_by_name("kick_808_hard") == "kick"
        assert SampleOrganizer._categorize_by_name("snare_tight") == "snare"
        assert SampleOrganizer._categorize_by_name("closed_hihat_01") == "hi-hat"
        assert SampleOrganizer._categorize_by_name("fx_riser_long") == "fx"
        assert SampleOrganizer._categorize_by_name("xyzabc") == "uncategorized"

    def test_scan_directory(self, sample_dir):
        results = SampleOrganizer.scan_directory(sample_dir)
        assert len(results) == 6

    def test_organize_dry_run(self, sample_dir, tmp_path):
        output_dir = str(tmp_path / "organized")
        result = SampleOrganizer.organize(sample_dir, output_dir, dry_run=True)
        assert result.total_files == 6
        assert result.categorized > 0
        assert result.moved == 6  # Dry run still counts

    def test_organize_real(self, sample_dir, tmp_path):
        output_dir = str(tmp_path / "organized")
        result = SampleOrganizer.organize(sample_dir, output_dir, copy=True)
        assert result.moved > 0
        assert len(result.errors) == 0
        # Check that subdirectories were created
        assert os.path.exists(os.path.join(output_dir, "kick"))

    def test_generate_report(self, sample_dir):
        samples = SampleOrganizer.scan_directory(sample_dir)
        report = SampleOrganizer.generate_report(samples)
        assert "Sample Library Report" in report
        assert "Total Samples:" in report


# ─── Project Backup Tests ───


@pytest.fixture
def project_dir(tmp_path):
    """Create a fake FL Studio project directory."""
    proj = tmp_path / "My Project"
    proj.mkdir()
    (proj / "song.flp").write_bytes(b"FLhd" + b"\x00" * 100)
    (proj / "kick.wav").write_bytes(b"RIFF" + b"\x00" * 200)
    (proj / "snare.wav").write_bytes(b"RIFF" + b"\x00" * 150)
    return str(proj)


class TestProjectBackup:
    def test_backup(self, project_dir, tmp_path):
        backup_root = str(tmp_path / "backups")
        pb = ProjectBackup(backup_root)
        result = pb.backup(project_dir, notes="Initial backup")
        assert result.success
        assert result.files_backed_up > 0

    def test_incremental_backup(self, project_dir, tmp_path):
        backup_root = str(tmp_path / "backups")
        pb = ProjectBackup(backup_root)

        r1 = pb.backup(project_dir, notes="First")
        assert r1.success

        r2 = pb.backup(project_dir, notes="Second")
        assert r2.success
        assert r2.files_skipped > 0  # Deduplication should kick in

    def test_list_backups(self, project_dir, tmp_path):
        backup_root = str(tmp_path / "backups")
        pb = ProjectBackup(backup_root)
        pb.backup(project_dir)
        backups = pb.list_backups()
        assert len(backups) == 1
        assert "timestamp" in backups[0]

    def test_restore(self, project_dir, tmp_path):
        backup_root = str(tmp_path / "backups")
        restore_dir = str(tmp_path / "restored")
        pb = ProjectBackup(backup_root)
        pb.backup(project_dir)
        success = pb.restore(restore_dir=restore_dir)
        assert success

    def test_cleanup(self, project_dir, tmp_path):
        backup_root = str(tmp_path / "backups")
        pb = ProjectBackup(backup_root)
        for i in range(7):
            # Modify a file to avoid dedup
            with open(os.path.join(project_dir, "song.flp"), "ab") as f:
                f.write(bytes([i]))
            pb.backup(project_dir)

        removed = pb.cleanup(keep_latest=3)
        assert removed > 0
        assert len(pb.list_backups()) == 3

    def test_nonexistent_dir(self, tmp_path):
        pb = ProjectBackup(str(tmp_path / "backups"))
        result = pb.backup("/nonexistent/path")
        assert not result.success


# ─── Render Queue Tests ───


class TestRenderQueue:
    def test_add_job(self, tmp_path):
        queue_file = str(tmp_path / "queue.json")
        rq = RenderQueue(queue_file)
        job = rq.add(str(tmp_path / "song.flp"))
        assert job.id == 1
        assert job.status == "pending"

    def test_add_batch(self, tmp_path):
        # Create fake FLP files
        for i in range(3):
            (tmp_path / f"song_{i}.flp").write_bytes(b"FLhd" + b"\x00" * 50)

        rq = RenderQueue()
        jobs = rq.add_batch(str(tmp_path))
        assert len(jobs) == 3

    def test_remove_job(self, tmp_path):
        rq = RenderQueue()
        job = rq.add(str(tmp_path / "test.flp"))
        rq.remove(job.id)
        assert len(rq.list_jobs()) == 0

    def test_clear_jobs(self, tmp_path):
        rq = RenderQueue()
        rq.add(str(tmp_path / "a.flp"))
        rq.add(str(tmp_path / "b.flp"))
        removed = rq.clear()
        assert removed == 2

    def test_stats(self, tmp_path):
        rq = RenderQueue()
        rq.add(str(tmp_path / "a.flp"))
        rq.add(str(tmp_path / "b.flp"))
        stats = rq.stats()
        assert stats["total"] == 2
        assert stats.get("pending", 0) == 2

    def test_persistence(self, tmp_path):
        queue_file = str(tmp_path / "queue.json")
        rq1 = RenderQueue(queue_file)
        rq1.add(str(tmp_path / "song.flp"))

        rq2 = RenderQueue(queue_file)
        assert len(rq2.list_jobs()) == 1

    def test_render_next_no_fl(self, tmp_path):
        """Render should fail gracefully when FL Studio isn't found."""
        rq = RenderQueue()
        rq.add(str(tmp_path / "song.flp"))
        job = rq.render_next(fl_path="/nonexistent/FL64.exe")
        assert job is not None
        assert job.status == "failed"
        assert job.error  # Error message varies by OS
