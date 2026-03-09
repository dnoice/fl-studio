"""Sample Slicer - Transient-based automatic audio slicing.

Detects transients in audio and splits the file into individual
sample files, ready for import into FL Studio's FPC, Slicex, or Sampler.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class SliceInfo:
    """Information about a single slice."""

    index: int
    start_sample: int
    end_sample: int
    duration_ms: float
    peak_level: float
    filepath: str | None = None

    @property
    def duration_str(self) -> str:
        if self.duration_ms < 1000:
            return f"{self.duration_ms:.0f}ms"
        return f"{self.duration_ms/1000:.2f}s"


@dataclass
class SliceResult:
    """Result of a slicing operation."""

    source_file: str
    slices: list[SliceInfo] = field(default_factory=list)
    sample_rate: int = 44100
    total_duration_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Source: {self.source_file}",
            f"Sample Rate: {self.sample_rate}Hz",
            f"Total Duration: {self.total_duration_ms/1000:.2f}s",
            f"Slices Found: {len(self.slices)}",
        ]
        for s in self.slices:
            lines.append(f"  [{s.index}] {s.duration_str} (peak: {s.peak_level:.3f})")
        return "\n".join(lines)


class SampleSlicer:
    """Transient-based audio slicer."""

    @staticmethod
    def detect_transients(
        audio: np.ndarray, sr: int, sensitivity: float = 0.5, min_gap_ms: float = 50.0
    ) -> list[int]:
        """Detect transient positions in audio.

        Args:
            audio: Mono audio array
            sr: Sample rate
            sensitivity: Detection sensitivity 0.0-1.0
            min_gap_ms: Minimum gap between transients in ms

        Returns:
            List of sample positions where transients occur
        """
        # Onset detection via spectral flux
        hop = 512
        n_fft = 2048
        from audio_tools._dsp_utils import make_window

        window = make_window(n_fft)

        n_frames = max(1, (len(audio) - n_fft) // hop)
        onset_env = np.zeros(n_frames)

        prev_spec = np.zeros(n_fft // 2 + 1)

        for i in range(n_frames):
            start = i * hop
            frame = audio[start : start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            spec = np.abs(np.fft.rfft(frame * window))
            diff = spec - prev_spec
            onset_env[i] = np.sum(np.maximum(0, diff))
            prev_spec = spec

        if np.max(onset_env) > 0:
            onset_env /= np.max(onset_env)

        # Peak picking
        threshold = (1.0 - sensitivity) * 0.3 + np.mean(onset_env)
        min_gap_frames = int(min_gap_ms * sr / 1000 / hop)

        peaks: list[int] = []
        for i in range(1, len(onset_env) - 1):
            if (
                onset_env[i] > onset_env[i - 1]
                and onset_env[i] > onset_env[i + 1]
                and onset_env[i] > threshold
            ) and (not peaks or (i - peaks[-1]) >= min_gap_frames):
                peaks.append(i)

        # Convert frame positions to sample positions
        return [p * hop for p in peaks]

    @staticmethod
    def slice(
        filepath: str | Path,
        output_dir: str | Path | None = None,
        sensitivity: float = 0.5,
        min_gap_ms: float = 50.0,
        min_duration_ms: float = 20.0,
        fade_ms: float = 2.0,
        normalize: bool = True,
        prefix: str | None = None,
        format: str = "wav",
    ) -> SliceResult:
        """Slice an audio file at transient points.

        Args:
            filepath: Path to audio file
            output_dir: Directory to save slices (None = don't save)
            sensitivity: Transient detection sensitivity 0.0-1.0
            min_gap_ms: Minimum gap between slice points
            min_duration_ms: Minimum slice duration (shorter slices are merged)
            fade_ms: Fade in/out duration for each slice
            normalize: Normalize each slice to peak level
            prefix: Filename prefix for slices
            format: Output format (wav, flac)

        Returns:
            SliceResult with slice information
        """
        filepath = Path(filepath)
        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
        except Exception as e:
            raise OSError(f"Failed to read audio file '{filepath}': {e}") from e

        mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio

        # Detect transients
        positions = SampleSlicer.detect_transients(mono, sr, sensitivity, min_gap_ms)

        # Add start and end
        if not positions or positions[0] > 0:
            positions.insert(0, 0)
        positions.append(len(mono))

        # Filter short slices
        min_samples = int(min_duration_ms * sr / 1000)
        filtered = [positions[0]]
        for pos in positions[1:]:
            if pos - filtered[-1] >= min_samples:
                filtered.append(pos)
            else:
                pass  # Drop short slice, keep previous boundary
        if filtered[-1] != len(mono):
            filtered.append(len(mono))
        positions = filtered

        # Create slices
        result = SliceResult(
            source_file=filepath.name,
            sample_rate=sr,
            total_duration_ms=len(mono) / sr * 1000,
        )

        fade_samples = int(fade_ms * sr / 1000)
        file_prefix = prefix or filepath.stem

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        for i in range(len(positions) - 1):
            start = positions[i]
            end = positions[i + 1]

            slice_audio = audio[start:end].copy() if audio.ndim > 1 else audio[start:end].copy()

            # Apply fade in/out
            if fade_samples > 0 and len(slice_audio) > fade_samples * 2:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                if slice_audio.ndim > 1:
                    for ch in range(slice_audio.shape[1]):
                        slice_audio[:fade_samples, ch] *= fade_in
                        slice_audio[-fade_samples:, ch] *= fade_out
                else:
                    slice_audio[:fade_samples] *= fade_in
                    slice_audio[-fade_samples:] *= fade_out

            # Normalize
            peak = np.max(np.abs(slice_audio))
            if normalize and peak > 0:
                slice_audio = slice_audio / peak * 0.95

            duration_ms = (end - start) / sr * 1000

            slice_info = SliceInfo(
                index=i,
                start_sample=start,
                end_sample=end,
                duration_ms=duration_ms,
                peak_level=float(peak),
            )

            # Save if output directory specified
            if output_dir:
                filename = f"{file_prefix}_{i+1:03d}.{format}"
                out_path = output_path / filename
                sf.write(str(out_path), slice_audio, sr)
                slice_info.filepath = str(out_path)

            result.slices.append(slice_info)

        return result

    @staticmethod
    def slice_uniform(
        filepath: str | Path,
        output_dir: str | Path | None = None,
        slice_duration_ms: float = 500.0,
        **kwargs,
    ) -> SliceResult:
        """Slice audio into uniform-length segments.

        Args:
            filepath: Path to audio file
            output_dir: Directory to save slices
            slice_duration_ms: Duration of each slice in ms
        """
        filepath = Path(filepath)
        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
        except Exception as e:
            raise OSError(f"Failed to read audio file '{filepath}': {e}") from e

        mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio

        slice_samples = int(slice_duration_ms * sr / 1000)
        positions = list(range(0, len(mono), slice_samples))
        positions.append(len(mono))

        result = SliceResult(
            source_file=filepath.name,
            sample_rate=sr,
            total_duration_ms=len(mono) / sr * 1000,
        )

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        file_prefix = kwargs.get("prefix", filepath.stem)
        fmt = kwargs.get("format", "wav")

        for i in range(len(positions) - 1):
            start = positions[i]
            end = min(positions[i + 1], len(audio))
            slice_audio = audio[start:end].copy() if audio.ndim > 1 else audio[start:end].copy()
            peak = float(np.max(np.abs(slice_audio))) if len(slice_audio) > 0 else 0

            slice_info = SliceInfo(
                index=i,
                start_sample=start,
                end_sample=end,
                duration_ms=(end - start) / sr * 1000,
                peak_level=peak,
            )

            if output_dir:
                filename = f"{file_prefix}_{i+1:03d}.{fmt}"
                out_path = output_path / filename
                sf.write(str(out_path), slice_audio, sr)
                slice_info.filepath = str(out_path)

            result.slices.append(slice_info)

        return result
