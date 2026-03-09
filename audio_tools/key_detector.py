"""Key Detector - Chromagram-based musical key detection for audio files.

Analyzes the pitch class distribution (chromagram) of audio content
and matches it against major and minor key profiles.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_tools._dsp_utils import make_window, midi_to_freq

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler key profiles (perceptually-derived)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class KeyResult:
    """Key detection result."""

    key: str  # e.g., "C Major", "A Minor"
    root: int  # Pitch class (0-11)
    mode: str  # "major" or "minor"
    confidence: float  # 0.0 - 1.0 (correlation coefficient)
    candidates: list[tuple[str, float]]  # Top candidates with scores
    chromagram: list[float]  # Normalized pitch class distribution

    def __repr__(self) -> str:
        return f"KeyResult(key='{self.key}', confidence={self.confidence:.2f})"

    @property
    def camelot(self) -> str:
        """Get Camelot wheel notation (used by DJs)."""
        major_camelot = {
            0: "8B",
            1: "3B",
            2: "10B",
            3: "5B",
            4: "12B",
            5: "7B",
            6: "2B",
            7: "9B",
            8: "4B",
            9: "11B",
            10: "6B",
            11: "1B",
        }
        minor_camelot = {
            0: "5A",
            1: "12A",
            2: "7A",
            3: "2A",
            4: "9A",
            5: "4A",
            6: "11A",
            7: "6A",
            8: "1A",
            9: "8A",
            10: "3A",
            11: "10A",
        }
        if self.mode == "major":
            return major_camelot.get(self.root, "?")
        return minor_camelot.get(self.root, "?")

    @property
    def open_key(self) -> str:
        """Get Open Key notation."""
        major_ok = {
            0: "1d",
            1: "8d",
            2: "3d",
            3: "10d",
            4: "5d",
            5: "12d",
            6: "7d",
            7: "2d",
            8: "9d",
            9: "4d",
            10: "11d",
            11: "6d",
        }
        minor_ok = {
            0: "1m",
            1: "8m",
            2: "3m",
            3: "10m",
            4: "5m",
            5: "12m",
            6: "7m",
            7: "2m",
            8: "9m",
            9: "4m",
            10: "11m",
            11: "6m",
        }
        if self.mode == "major":
            return major_ok.get(self.root, "?")
        return minor_ok.get(self.root, "?")


class KeyDetector:
    """Musical key detection from audio files."""

    @staticmethod
    def detect(filepath: str | Path, duration: float = 0.0) -> KeyResult:
        """Detect the musical key of an audio file.

        Args:
            filepath: Path to audio file
            duration: Max seconds to analyze (0 = full file)

        Returns:
            KeyResult with detected key and confidence
        """
        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
        except Exception as e:
            raise OSError(f"Failed to read audio file '{filepath}': {e}") from e

        if len(audio) == 0:
            return KeyResult("Unknown", 0, "unknown", 0.0, [], [0.0] * 12)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if duration > 0:
            max_samples = int(duration * sr)
            audio = audio[:max_samples]

        chromagram = KeyDetector._compute_chromagram(audio, sr)
        return KeyDetector._match_key(chromagram)

    @staticmethod
    def detect_from_array(audio: np.ndarray, sr: int) -> KeyResult:
        """Detect key from a numpy audio array."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        chromagram = KeyDetector._compute_chromagram(audio, sr)
        return KeyDetector._match_key(chromagram)

    @staticmethod
    def _compute_chromagram(audio: np.ndarray, sr: int) -> np.ndarray:
        """Compute normalized chromagram (pitch class distribution).

        Uses constant-Q-like analysis via windowed DFT at specific frequencies.
        """
        n_fft = 8192
        hop = n_fft // 4
        chroma = np.zeros(12)

        window = make_window(n_fft)

        n_frames = max(1, (len(audio) - n_fft) // hop)

        for frame_idx in range(n_frames):
            start = frame_idx * hop
            frame = audio[start : start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))

            windowed = frame * window
            spectrum = np.abs(np.fft.rfft(windowed))

            # Accumulate energy at each pitch class
            for midi_note in range(24, 96):
                pitch_class = midi_note % 12
                target_freq = midi_to_freq(midi_note)

                # Find closest frequency bin
                bin_idx = int(round(target_freq * n_fft / sr))
                if 0 < bin_idx < len(spectrum) - 1:
                    # Use a small window around the target bin
                    energy = 0.0
                    for offset in range(-2, 3):
                        idx = bin_idx + offset
                        if 0 <= idx < len(spectrum):
                            weight = 1.0 - abs(offset) * 0.2
                            energy += spectrum[idx] ** 2 * weight
                    chroma[pitch_class] += energy

        # Normalize
        total = np.sum(chroma)
        if total > 0:
            chroma /= total

        return chroma

    @staticmethod
    def _match_key(chroma: np.ndarray) -> KeyResult:
        """Match chromagram against key profiles using correlation."""
        candidates = []

        for root in range(12):
            # Rotate chromagram so root is at index 0
            rotated = np.roll(chroma, -root)

            # Correlate with major profile
            corr_matrix = np.corrcoef(rotated, MAJOR_PROFILE)
            maj_corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
            key_name = f"{NOTE_NAMES[root]} Major"
            candidates.append((key_name, root, "major", maj_corr))

            # Correlate with minor profile
            corr_matrix = np.corrcoef(rotated, MINOR_PROFILE)
            min_corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
            key_name = f"{NOTE_NAMES[root]} Minor"
            candidates.append((key_name, root, "minor", min_corr))

        # Sort by correlation
        candidates.sort(key=lambda x: x[3], reverse=True)

        best = candidates[0]
        return KeyResult(
            key=best[0],
            root=best[1],
            mode=best[2],
            confidence=round(max(0, best[3]), 3),
            candidates=[(name, round(score, 3)) for name, _, _, score in candidates[:6]],
            chromagram=[round(float(c), 4) for c in chroma],
        )

    @staticmethod
    def detect_batch(filepaths: list[str | Path], **kwargs) -> dict[str, KeyResult]:
        """Detect key for multiple files."""
        results = {}
        for fp in filepaths:
            path = Path(fp)
            try:
                results[path.name] = KeyDetector.detect(fp, **kwargs)
            except Exception as e:
                results[path.name] = KeyResult(f"Error: {e}", 0, "unknown", 0.0, [], [])
        return results
