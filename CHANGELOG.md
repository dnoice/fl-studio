# Changelog

All notable changes to fl-studio-toolkit will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.0] - 2026-03-01

### Fixed

- **BUG-001**: No-op assignment in `sample_slicer.py` slice merging — short slices now correctly dropped
- **BUG-002**: Negative octave indexing in `midi_analyzer.py` for low MIDI pitches (0-11)
- **BUG-003**: Empty slice in arpeggiator UP_DOWN/DOWN_UP modes with 1-2 notes
- **BUG-004**: Hardcoded stereo assumption in `gain_staging.py` — now handles any channel count
- **BUG-005**: `min()`/`max()` crash on empty duration list in `midi_analyzer.py`
- Flaky `test_render_next_no_fl` test on Windows (error message mismatch)
- 357 ruff lint issues auto-fixed (import sorting, modern type syntax, OSError aliasing)
- 11 ruff lint issues manually fixed (multi-line statements, raise chaining, context managers)
- 9 mypy type annotation issues fixed (missing returns, untyped variables, float/int mismatch)

### Added

- `dsp/__init__.py` — Python fallback API for C++ DSP module (`biquad_filter()`, `compress()`, `db_to_linear()`, `midi_to_freq()`, etc.)
- `tests/test_dsp.py` — 40+ tests for C++ pybind11 bindings (auto-skipped if not built)
- `audio_tools/_dsp_utils.py` — shared `resample()`, `make_window()`, `midi_to_freq()`, `freq_to_midi()`
- `midi_tools/_validation.py` — centralized `validate_pitch()`, `validate_velocity()`, `validate_channel()`
- `tests/conftest.py` — shared fixtures for audio signals, WAV files, MIDI data, workflow dirs
- `tests/test_hardened.py` — 104 new tests: parametrized, negative (20+), edge cases, expanded coverage
- `.github/workflows/ci.yml` — CI/CD with lint, type-check, test matrix (Python 3.10-3.13, ubuntu/windows)
- `.pre-commit-config.yaml` with ruff, mypy, trailing whitespace, YAML checks
- `.gitignore` covering Python, venv, build, IDE, OS, and C++ artifacts
- `README.md` with project overview, installation, quick start, and module docs
- `CONTRIBUTING.md` with coding conventions, error handling patterns, testing requirements
- `CHANGELOG.md` (this file)
- `__all__` declarations in all module `__init__.py` files for proper re-export
- Module-level docstrings with quick-start examples in all `__init__.py` files

### Changed

- **Resampling upgraded** to `scipy.signal.resample_poly` (polyphase) with numpy fallback
- Sample rate and Q factor validation added to `_biquad.py` (raises `ValueError`)
- Frequency clamped to Nyquist range in biquad filter calculations
- EQBand in `effects_chain.py` now clamps frequency >= 1.0 and Q >= 0.01
- `sf.read()` calls wrapped in try/except in `spectrum_analyzer.py`, `sample_slicer.py`
- FLP parser now handles malformed/truncated files with meaningful error messages
- MIDI `extract_notes()` now validates pitch/velocity/channel, skips malformed messages
- MIDI `notes_to_track()` validates channel param, clamps pitch/velocity to valid range
- Symlink protection added to `preset_manager.py` scan
- Path traversal protection added to `install_scripts.py` --fl-path argument
- All bare `-> dict` return types improved to `-> dict[str, object]`
- Biquad return type specified as `tuple[float, float, float, float, float]`
- Refactored `batch_processor.py`, `format_converter.py` to use shared `resample()`
- Refactored `bpm_detector.py`, `sample_slicer.py`, `key_detector.py` to use shared `make_window()`
- Refactored `key_detector.py` to use shared `midi_to_freq()`
- pyproject.toml: ruff, mypy, coverage config; dev dependencies; strict pytest markers
- Version bumped to 0.3.0 across all modules

### Security

- **SEC-001**: Preset manager no longer follows symlinks during directory scan
- **SEC-002**: FL Studio script installer validates and resolves --fl-path argument

## [0.2.0] - 2026-01-15

### Added

- `release/` module — metadata management, album packaging, lyrics, licensing, export pipeline
- `workflow/` module — FLP parser, preset manager, project backup, render queue, sample organizer
- `mixing/mix_analyzer.py` — frequency balance, stereo width, dynamic range analysis
- `mixing/reference_compare.py` — reference track comparison tool
- Integration test suite (`tests/test_real_world.py`) with 36 multi-module workflow tests

## [0.1.0] - 2025-10-01

### Added

- `audio_tools/` module — BPM detection, key detection, sample slicing, batch processing, spectrum analysis, format conversion
- `midi_tools/` module — scale library, chord engine, arpeggiator, drum patterns, MIDI file utils, analyzer, transforms
- `mixing/` module — biquad filters, effects chain, channel strip, mix bus, gain staging, stereo tools
- `dsp/` C++ module — DSP core, effects, filters with pybind11 bindings
- `fl_scripts/` — 6 piano roll + 4 Edison integration scripts
- `install_scripts.py` — FL Studio script installer
