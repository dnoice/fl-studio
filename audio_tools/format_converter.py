"""Format Converter - Audio format conversion utility.

Convert between WAV, FLAC, OGG, and other formats supported by soundfile,
with options for bit depth, sample rate, and channel configuration.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_FORMATS = {
    "wav": {"extension": ".wav", "subtype_default": "PCM_24"},
    "flac": {"extension": ".flac", "subtype_default": "PCM_24"},
    "ogg": {"extension": ".ogg", "subtype_default": "VORBIS"},
    "aiff": {"extension": ".aiff", "subtype_default": "PCM_24"},
}

BIT_DEPTHS = {
    16: "PCM_16",
    24: "PCM_24",
    32: "FLOAT",
}


@dataclass
class ConversionResult:
    """Result of a format conversion."""

    source: str
    output: str
    source_format: str
    output_format: str
    source_sr: int
    output_sr: int
    source_channels: int
    output_channels: int
    success: bool
    error: str | None = None


class FormatConverter:
    """Audio format conversion with sample rate and bit depth options."""

    @staticmethod
    def convert(
        filepath: str | Path,
        output_path: str | Path | None = None,
        output_format: str = "wav",
        bit_depth: int = 24,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> ConversionResult:
        """Convert a single audio file.

        Args:
            filepath: Input file path
            output_path: Output file path (auto-generated if None)
            output_format: Target format (wav, flac, ogg, aiff)
            bit_depth: Bit depth (16, 24, 32)
            sample_rate: Target sample rate (None = keep original)
            channels: Target channels (1=mono, 2=stereo, None=keep)
        """
        filepath = Path(filepath)

        if output_format not in SUPPORTED_FORMATS:
            return ConversionResult(
                str(filepath),
                "",
                "",
                output_format,
                0,
                0,
                0,
                0,
                False,
                f"Unsupported format: {output_format}",
            )

        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
            source_channels = audio.shape[1] if audio.ndim > 1 else 1

            # Channel conversion
            if channels is not None:
                if channels == 1 and audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                elif channels == 2 and audio.ndim == 1:
                    audio = np.column_stack([audio, audio])
                elif channels > 2 and audio.ndim == 1:
                    audio = np.column_stack([audio] * channels)

            output_channels = audio.shape[1] if audio.ndim > 1 else 1

            # Sample rate conversion
            output_sr = sample_rate or sr
            if output_sr != sr:
                audio = FormatConverter._resample(audio, sr, output_sr)

            # Determine output path
            if output_path is None:
                ext = SUPPORTED_FORMATS[output_format]["extension"]
                output_path = filepath.with_suffix(ext)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Determine subtype
            subtype = "VORBIS" if output_format == "ogg" else BIT_DEPTHS.get(bit_depth, "PCM_24")

            sf.write(str(output_path), audio, output_sr, subtype=subtype)

            return ConversionResult(
                source=str(filepath),
                output=str(output_path),
                source_format=filepath.suffix,
                output_format=output_format,
                source_sr=sr,
                output_sr=output_sr,
                source_channels=source_channels,
                output_channels=output_channels,
                success=True,
            )

        except Exception as e:
            return ConversionResult(
                source=str(filepath),
                output=str(output_path) if output_path else "",
                source_format=filepath.suffix if filepath else "",
                output_format=output_format,
                source_sr=0,
                output_sr=0,
                source_channels=0,
                output_channels=0,
                success=False,
                error=str(e),
            )

    @staticmethod
    def convert_batch(
        input_dir: str | Path,
        output_dir: str | Path | None = None,
        output_format: str = "wav",
        extensions: tuple[str, ...] = (".wav", ".flac", ".aiff", ".ogg", ".mp3"),
        recursive: bool = False,
        **kwargs,
    ) -> list[ConversionResult]:
        """Convert all audio files in a directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory (None = same directory)
            output_format: Target format
            extensions: File extensions to process
            recursive: Search subdirectories
        """
        input_path = Path(input_dir)
        results = []

        pattern = "**/*" if recursive else "*"
        files = sorted(f for f in input_path.glob(pattern) if f.suffix.lower() in extensions)

        for f in files:
            if output_dir:
                rel = f.relative_to(input_path)
                ext = SUPPORTED_FORMATS.get(output_format, {}).get("extension", f".{output_format}")
                out = Path(output_dir) / rel.with_suffix(ext)
            else:
                out = None

            result = FormatConverter.convert(f, out, output_format, **kwargs)
            results.append(result)

        return results

    @staticmethod
    def get_file_info(filepath: str | Path) -> dict:
        """Get audio file format information."""
        filepath = Path(filepath)
        info = sf.info(str(filepath))
        return {
            "filename": filepath.name,
            "format": info.format,
            "subtype": info.subtype,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_seconds": info.duration,
            "duration_str": f"{info.duration:.2f}s",
        }

    @staticmethod
    def _resample(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        from audio_tools._dsp_utils import resample

        return resample(audio, src_sr, target_sr)
