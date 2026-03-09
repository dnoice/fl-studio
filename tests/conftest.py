"""Shared test fixtures for fl-studio-toolkit.

Consolidates audio signal generators, WAV file creation, and common
test data used across multiple test modules.
"""

import struct

import numpy as np
import pytest
import soundfile as sf

# ─── Constants ───

SR = 44100
DURATION = 1.0


# ─── In-Memory Audio Signals ───


@pytest.fixture
def mono_signal():
    """1 second 440 Hz mono sine at 44100 Hz, amplitude 0.5."""
    t = np.linspace(0, DURATION, SR, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5, SR


@pytest.fixture
def stereo_signal():
    """1 second stereo signal (440 Hz left, 554 Hz right) at 44100 Hz."""
    t = np.linspace(0, DURATION, SR, endpoint=False)
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    right = np.sin(2 * np.pi * 554 * t).astype(np.float32) * 0.4
    return np.column_stack([left, right]), SR


@pytest.fixture
def loud_signal():
    """1 second near-clipping mono sine (amplitude 0.99)."""
    t = np.linspace(0, DURATION, SR, endpoint=False)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.99, SR


@pytest.fixture
def silent_signal():
    """1 second of silence."""
    return np.zeros(SR, dtype=np.float32), SR


# ─── WAV File Fixtures ───


@pytest.fixture
def sample_wav(tmp_path):
    """Generate a 1-second 440 Hz mono WAV file. Returns (path, audio, sr)."""
    t = np.linspace(0, DURATION, SR, endpoint=False)
    audio = (0.8 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = str(tmp_path / "test_440hz.wav")
    sf.write(path, audio, SR)
    return path, audio, SR


@pytest.fixture
def stereo_wav(tmp_path):
    """Generate a 1-second stereo WAV file. Returns (path, audio, sr)."""
    t = np.linspace(0, DURATION, SR, endpoint=False)
    left = (0.7 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    right = (0.5 * np.sin(2 * np.pi * 554 * t)).astype(np.float32)
    audio = np.column_stack([left, right])
    path = str(tmp_path / "test_stereo.wav")
    sf.write(path, audio, SR)
    return path, audio, SR


@pytest.fixture
def rhythmic_wav(tmp_path):
    """Generate a 4-second rhythmic click track at 120 BPM. Returns (path, audio, sr)."""
    duration = 4.0
    bpm = 120.0
    beat_interval = 60.0 / bpm
    audio = np.zeros(int(SR * duration), dtype=np.float32)

    for beat in range(int(duration / beat_interval)):
        start = int(beat * beat_interval * SR)
        click_len = min(200, len(audio) - start)
        if click_len > 0:
            click = np.sin(2 * np.pi * 1000 * np.arange(click_len) / SR)
            click *= np.exp(-np.arange(click_len) / 50)
            audio[start : start + click_len] += click.astype(np.float32) * 0.9

    path = str(tmp_path / "test_120bpm.wav")
    sf.write(path, audio, SR)
    return path, audio, SR


@pytest.fixture
def silent_wav(tmp_path):
    """Generate a 1-second silent WAV file. Returns (path, audio, sr)."""
    audio = np.zeros(SR, dtype=np.float32)
    path = str(tmp_path / "test_silent.wav")
    sf.write(path, audio, SR)
    return path, audio, SR


# ─── MIDI Helpers ───


def make_notes(pitches, velocity=100, start_tick=0, duration_ticks=480, gap_ticks=0):
    """Helper to create a list of NoteEvent objects from pitches."""
    from midi_tools.midi_file_utils import NoteEvent

    notes = []
    tick = start_tick
    for p in pitches:
        notes.append(
            NoteEvent(
                pitch=p,
                velocity=velocity,
                start_tick=tick,
                duration_ticks=duration_ticks,
            )
        )
        tick += duration_ticks + gap_ticks
    return notes


# ─── Workflow Fixtures ───


@pytest.fixture
def fake_flp(tmp_path):
    """Create a minimal valid FLP file for testing."""
    path = tmp_path / "test.flp"

    with open(path, "wb") as f:
        f.write(b"FLhd")
        f.write(struct.pack("<I", 6))
        f.write(struct.pack("<H", 0))
        f.write(struct.pack("<H", 1))
        f.write(struct.pack("<H", 96))

        events = bytearray()
        events.append(66)
        events.extend(struct.pack("<H", 140))

        version_text = "FL Studio 2025".encode("utf-16-le")
        events.append(199)
        events.append(len(version_text))
        events.extend(version_text)

        f.write(b"FLdt")
        f.write(struct.pack("<I", len(events)))
        f.write(events)

    return str(path)


@pytest.fixture
def preset_dir(tmp_path):
    """Create a mock preset directory with sample .fst files."""
    plugin_dir = tmp_path / "Sytrus" / "Pads"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "Warm Pad.fst").write_bytes(b"\x00" * 100)
    (plugin_dir / "Dark Pad.fst").write_bytes(b"\x00" * 200)

    plugin_dir2 = tmp_path / "Harmor" / "Bass"
    plugin_dir2.mkdir(parents=True)
    (plugin_dir2 / "Sub Bass.fst").write_bytes(b"\x00" * 150)

    return str(tmp_path)


@pytest.fixture
def sample_dir(tmp_path):
    """Create a directory with test audio samples."""
    for name in [
        "kick_01.wav",
        "snare_hard.wav",
        "hihat_closed.wav",
        "bass_deep.wav",
        "lead_synth.wav",
        "random_sound.wav",
    ]:
        audio = np.random.randn(SR // 4).astype(np.float32) * 0.5
        sf.write(str(tmp_path / name), audio, SR)
    return str(tmp_path)


@pytest.fixture
def project_dir(tmp_path):
    """Create a fake FL Studio project directory."""
    proj = tmp_path / "My Project"
    proj.mkdir()
    (proj / "song.flp").write_bytes(b"FLhd" + b"\x00" * 100)
    (proj / "kick.wav").write_bytes(b"RIFF" + b"\x00" * 200)
    (proj / "snare.wav").write_bytes(b"RIFF" + b"\x00" * 150)
    return str(proj)
