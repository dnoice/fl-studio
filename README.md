# FL Studio Toolkit

Comprehensive Python toolkit for FL Studio production workflows — MIDI tools, audio processing, mixing, workflow automation, release pipeline, and high-performance DSP.

## Modules

| Module | Description |
|--------|-------------|
| **midi_tools** | Scale library (100+ scales), chord engine, arpeggiator, drum patterns, MIDI analysis & transformation |
| **audio_tools** | BPM detection, key detection, sample slicing, batch processing, spectrum analysis, format conversion |
| **mixing** | Effects chain (compressor, limiter, EQ, saturation, de-esser), channel strip, gain staging, stereo tools, mix analysis |
| **workflow** | FLP project file parsing, preset management, project backup, render queue, sample organization |
| **release** | Track metadata (ID3), album management, synced lyrics, licensing/splits, export pipeline |
| **dsp** | C++ DSP engine with pybind11 bindings — reverb, chorus, delay, phaser, biquad/Butterworth filters |
| **fl_scripts** | 10 ready-to-use FL Studio scripts (6 piano roll + 4 Edison) |

## Installation

```bash
# Core dependencies
pip install -e .

# With analysis tools (librosa, matplotlib)
pip install -e ".[analysis]"

# Development (adds pytest, coverage)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

Requires **Python 3.10+**.

## Quick Start

```python
from midi_tools import ScaleLibrary, ChordEngine, Arpeggiator
from audio_tools import BPMDetector, KeyDetector, SampleSlicer
from mixing import EffectsChain

# Detect key and BPM from audio
key = KeyDetector.detect("track.wav")
bpm = BPMDetector.detect("track.wav")

# Generate chords in the detected key
chords = ChordEngine.diatonic_chords("major", root=key.root)

# Process audio through a mastering chain
chain = EffectsChain.master_chain()
chain.process_file("mix.wav", "master.wav")

# Slice samples at transients
slices = SampleSlicer.slice("loop.wav", output_dir="slices/")
```

## FL Studio Scripts

Install scripts directly into FL Studio:

```bash
python install_scripts.py
# Or specify a custom path:
python install_scripts.py --fl-path "C:\Program Files\Image-Line\FL Studio 2025"
```

## Testing

```bash
# Run unit tests
pytest

# Run all tests including slow integration tests
pytest -m ""

# With coverage
pytest --cov=midi_tools --cov=audio_tools --cov=mixing --cov=workflow --cov=release
```

## Project Structure

```
fl-studio/
├── audio_tools/     # Audio processing & analysis
├── midi_tools/      # MIDI manipulation & generation
├── mixing/          # Effects, mixing, mastering
├── workflow/        # FL Studio project automation
├── release/         # Distribution & metadata
├── dsp/             # C++ DSP engine (optional)
├── fl_scripts/      # FL Studio integration scripts
├── tests/           # Test suite (277+ tests)
└── assets/audio/    # Test audio assets
```

## License

MIT
