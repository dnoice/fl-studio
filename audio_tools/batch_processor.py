"""Batch Processor - Batch audio processing operations.

Process multiple audio files with operations like normalize,
trim silence, fade in/out, convert format, resample, and more.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import soundfile as sf


class Operation(Enum):
    NORMALIZE = "normalize"
    TRIM_SILENCE = "trim_silence"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    RESAMPLE = "resample"
    MONO = "mono"
    STEREO = "stereo"
    REVERSE = "reverse"
    DC_OFFSET = "dc_offset"
    GAIN = "gain"


@dataclass
class ProcessingResult:
    """Result of processing a single file."""

    source: str
    output: str
    operations: list[str]
    success: bool
    error: str | None = None
    peak_before: float = 0.0
    peak_after: float = 0.0


class BatchProcessor:
    """Batch audio file processor with chainable operations."""

    def __init__(self):
        self._operations: list[tuple[Operation, dict]] = []

    def normalize(self, target_db: float = -0.3) -> "BatchProcessor":
        """Normalize peak level to target dB."""
        self._operations.append((Operation.NORMALIZE, {"target_db": target_db}))
        return self

    def trim_silence(self, threshold_db: float = -60.0, pad_ms: float = 10.0) -> "BatchProcessor":
        """Trim leading and trailing silence."""
        self._operations.append(
            (Operation.TRIM_SILENCE, {"threshold_db": threshold_db, "pad_ms": pad_ms})
        )
        return self

    def fade_in(self, duration_ms: float = 10.0, curve: str = "linear") -> "BatchProcessor":
        """Apply fade in."""
        self._operations.append((Operation.FADE_IN, {"duration_ms": duration_ms, "curve": curve}))
        return self

    def fade_out(self, duration_ms: float = 50.0, curve: str = "linear") -> "BatchProcessor":
        """Apply fade out."""
        self._operations.append((Operation.FADE_OUT, {"duration_ms": duration_ms, "curve": curve}))
        return self

    def to_mono(self) -> "BatchProcessor":
        """Convert to mono."""
        self._operations.append((Operation.MONO, {}))
        return self

    def to_stereo(self) -> "BatchProcessor":
        """Convert mono to stereo."""
        self._operations.append((Operation.STEREO, {}))
        return self

    def reverse(self) -> "BatchProcessor":
        """Reverse the audio."""
        self._operations.append((Operation.REVERSE, {}))
        return self

    def remove_dc_offset(self) -> "BatchProcessor":
        """Remove DC offset."""
        self._operations.append((Operation.DC_OFFSET, {}))
        return self

    def gain(self, db: float = 0.0) -> "BatchProcessor":
        """Apply gain in dB."""
        self._operations.append((Operation.GAIN, {"db": db}))
        return self

    def process_file(
        self,
        filepath: str | Path,
        output_path: str | Path | None = None,
        output_format: str = "wav",
        output_sr: int | None = None,
    ) -> ProcessingResult:
        """Process a single audio file.

        Args:
            filepath: Input file path
            output_path: Output file path (None = overwrite)
            output_format: Output format
            output_sr: Output sample rate (None = keep original)
        """
        filepath = Path(filepath)
        op_names = []

        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
            peak_before = float(np.max(np.abs(audio))) if len(audio) > 0 else 0

            for op, params in self._operations:
                audio, sr = self._apply_operation(audio, sr, op, params)
                op_names.append(op.value)

            # Resample if requested
            if output_sr and output_sr != sr:
                audio = self._resample(audio, sr, output_sr)
                sr = output_sr

            # Determine output path
            if output_path is None:
                output_path = filepath.with_suffix(f".{output_format}")
            else:
                output_path = Path(output_path)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, sr)

            peak_after = float(np.max(np.abs(audio))) if len(audio) > 0 else 0

            return ProcessingResult(
                source=str(filepath),
                output=str(output_path),
                operations=op_names,
                success=True,
                peak_before=peak_before,
                peak_after=peak_after,
            )
        except Exception as e:
            return ProcessingResult(
                source=str(filepath),
                output=str(output_path) if output_path else "",
                operations=op_names,
                success=False,
                error=str(e),
            )

    def process_batch(
        self,
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        extensions: tuple[str, ...] = (".wav", ".flac", ".aiff", ".mp3"),
        output_format: str = "wav",
        recursive: bool = False,
        **kwargs,
    ) -> list[ProcessingResult]:
        """Process all audio files in a directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory (None = in-place)
            extensions: File extensions to process
            output_format: Output format
            recursive: Search subdirectories
        """
        input_path = Path(input_dir)
        results = []

        pattern = "**/*" if recursive else "*"
        files = sorted(f for f in input_path.glob(pattern) if f.suffix.lower() in extensions)

        for f in files:
            if output_dir:
                rel_path = f.relative_to(input_path)
                out = Path(output_dir) / rel_path.with_suffix(f".{output_format}")
            else:
                out = None

            result = self.process_file(f, out, output_format, **kwargs)
            results.append(result)

        return results

    def _apply_operation(
        self, audio: np.ndarray, sr: int, op: Operation, params: dict
    ) -> tuple[np.ndarray, int]:
        """Apply a single operation to audio data."""

        if op == Operation.NORMALIZE:
            target_linear = 10 ** (params["target_db"] / 20)
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio * (target_linear / peak)

        elif op == Operation.TRIM_SILENCE:
            threshold = 10 ** (params["threshold_db"] / 20)
            pad_samples = int(params["pad_ms"] * sr / 1000)

            envelope = np.max(np.abs(audio), axis=1) if audio.ndim > 1 else np.abs(audio)

            above = np.where(envelope > threshold)[0]
            if len(above) > 0:
                start = max(0, above[0] - pad_samples)
                end = min(len(audio), above[-1] + pad_samples)
                audio = audio[start:end]

        elif op == Operation.FADE_IN:
            samples = int(params["duration_ms"] * sr / 1000)
            samples = min(samples, len(audio))
            if params.get("curve") == "exponential":
                fade = np.linspace(0, 1, samples) ** 2
            else:
                fade = np.linspace(0, 1, samples)
            if audio.ndim > 1:
                audio[:samples] *= fade[:, np.newaxis]
            else:
                audio[:samples] *= fade

        elif op == Operation.FADE_OUT:
            samples = int(params["duration_ms"] * sr / 1000)
            samples = min(samples, len(audio))
            if params.get("curve") == "exponential":
                fade = np.linspace(1, 0, samples) ** 2
            else:
                fade = np.linspace(1, 0, samples)
            if audio.ndim > 1:
                audio[-samples:] *= fade[:, np.newaxis]
            else:
                audio[-samples:] *= fade

        elif op == Operation.MONO:
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

        elif op == Operation.STEREO:
            if audio.ndim == 1:
                audio = np.column_stack([audio, audio])

        elif op == Operation.REVERSE:
            audio = audio[::-1].copy()

        elif op == Operation.DC_OFFSET:
            if audio.ndim > 1:
                for ch in range(audio.shape[1]):
                    audio[:, ch] -= np.mean(audio[:, ch])
            else:
                audio -= np.mean(audio)

        elif op == Operation.GAIN:
            gain_linear = 10 ** (params["db"] / 20)
            audio = audio * gain_linear

        return audio, sr

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling via linear interpolation."""
        from audio_tools._dsp_utils import resample

        return resample(audio, src_sr, target_sr)
