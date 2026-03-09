# Contributing to FL Studio Toolkit

## Development Setup

```bash
# Clone and install in dev mode
git clone <repo-url> && cd fl-studio
pip install -e ".[dev]"

# Run tests
pytest tests/ -m "not slow"

# Run linter
ruff check .

# Run type checker
mypy midi_tools audio_tools mixing workflow release --ignore-missing-imports
```

## Code Conventions

### Error Handling (ERR-006)

The project follows a clear convention for error signaling:

**Raise exceptions** for programmer errors (bad arguments):
```python
# Wrong type, out-of-range, impossible values
if sr <= 0:
    raise ValueError(f"Sample rate must be positive, got {sr}")
```

**Return result objects** for runtime failures (bad files, external issues):
```python
# File not found, corrupt data, external tool missing
result = ProcessingResult(success=False, error="File not found")
```

Specifically:
- `ValueError` for invalid arguments (bad pitch, negative sample rate, out-of-range Q)
- `IOError` for file I/O failures (corrupt audio, unreadable MIDI)
- `FileNotFoundError` for missing files
- Result dataclasses (`ProcessingResult`, `SliceResult`, etc.) for operations that can partially succeed

### Return Types (API-001)

- **Analysis functions** return dataclass objects (`BPMResult`, `KeyResult`, `SpectrumData`, etc.)
- **Audio processing functions** return `(audio_array, sample_rate)` tuples
- **MIDI manipulation functions** return `mido.MidiFile` objects
- **File operations** return result dataclasses with success/error fields
- **Comparison/info functions** return `dict[str, object]`

### Type Annotations

- All public functions must have full type annotations
- Use `dict[str, object]` instead of bare `dict` for return types
- Use `tuple[float, float, float, float, float]` instead of bare `tuple`
- Use `Union[str, Path]` for file path parameters
- Use `Optional[X]` for nullable parameters

### MIDI Validation

Use the centralized validators from `midi_tools/_validation.py`:
```python
from midi_tools._validation import validate_pitch, validate_velocity, validate_channel

validate_pitch(note)       # Raises ValueError if not 0-127
validate_velocity(vel)     # Raises ValueError if not 0-127
validate_channel(ch)       # Raises ValueError if not 0-15
```

### Shared DSP Utilities

Use the shared utilities from `audio_tools/_dsp_utils.py`:
```python
from audio_tools._dsp_utils import resample, make_window, midi_to_freq, freq_to_midi

audio = resample(audio, src_sr=44100, target_sr=48000)
window = make_window(2048, "hanning")
freq = midi_to_freq(69)  # 440.0 Hz
```

## Testing

### Test Structure

- Unit tests go in `tests/test_<module>.py`
- Hardened tests (parametrized, negative, edge cases) go in `tests/test_hardened.py`
- Integration tests go in `tests/test_real_world.py`
- Shared fixtures are in `tests/conftest.py`

### Test Requirements

- New features must include tests
- Bug fixes must include a regression test
- Use `@pytest.mark.parametrize` when testing multiple inputs
- Include negative tests (invalid inputs that should raise)
- Use fixtures from `conftest.py` instead of creating ad-hoc test data
- Mark slow tests with `@pytest.mark.slow`

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run `ruff check .` and `pytest tests/ -m "not slow"`
4. Submit a PR with a clear description of changes
5. Ensure CI passes before requesting review
