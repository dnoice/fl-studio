"""BPM Detector - Onset-based tempo detection for audio files.

Uses onset strength envelope and autocorrelation to estimate BPM,
with support for multiple estimation methods and confidence scoring.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class BPMResult:
    """BPM detection result with confidence metrics."""

    bpm: float
    confidence: float  # 0.0 - 1.0
    bpm_candidates: list[tuple[float, float]]  # (bpm, strength) pairs
    method: str

    def __repr__(self) -> str:
        return f"BPMResult(bpm={self.bpm:.1f}, confidence={self.confidence:.2f})"


class BPMDetector:
    """Audio tempo detection using onset analysis and autocorrelation."""

    @staticmethod
    def detect(
        filepath: str | Path,
        min_bpm: float = 60.0,
        max_bpm: float = 200.0,
        method: str = "autocorrelation",
    ) -> BPMResult:
        """Detect BPM from an audio file.

        Args:
            filepath: Path to audio file (WAV, FLAC, MP3, etc.)
            min_bpm: Minimum expected BPM
            max_bpm: Maximum expected BPM
            method: Detection method - "autocorrelation", "onset_intervals", or "both"

        Returns:
            BPMResult with estimated BPM and confidence
        """
        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
        except Exception as e:
            raise OSError(f"Failed to read audio file '{filepath}': {e}") from e

        if len(audio) == 0:
            return BPMResult(0.0, 0.0, [], "empty_file")

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if method == "autocorrelation":
            return BPMDetector._autocorrelation_method(audio, sr, min_bpm, max_bpm)
        elif method == "onset_intervals":
            return BPMDetector._onset_interval_method(audio, sr, min_bpm, max_bpm)
        else:
            # Run both and pick the most confident
            r1 = BPMDetector._autocorrelation_method(audio, sr, min_bpm, max_bpm)
            r2 = BPMDetector._onset_interval_method(audio, sr, min_bpm, max_bpm)
            return r1 if r1.confidence >= r2.confidence else r2

    @staticmethod
    def _compute_onset_envelope(audio: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
        """Compute onset strength envelope using spectral flux."""
        # STFT parameters
        n_fft = 2048
        from audio_tools._dsp_utils import make_window

        window = make_window(n_fft)

        # Pad audio
        audio_padded = np.pad(audio, (n_fft // 2, n_fft // 2))

        # Compute magnitude spectrogram
        n_frames = 1 + (len(audio_padded) - n_fft) // hop_length
        spec = np.zeros((n_fft // 2 + 1, n_frames))

        for i in range(n_frames):
            start = i * hop_length
            frame = audio_padded[start : start + n_fft] * window
            fft_result = np.fft.rfft(frame)
            spec[:, i] = np.abs(fft_result)

        # Spectral flux (onset strength)
        onset_env = np.zeros(n_frames)
        for i in range(1, n_frames):
            diff = spec[:, i] - spec[:, i - 1]
            onset_env[i] = np.sum(np.maximum(0, diff))  # Half-wave rectified

        # Normalize
        if np.max(onset_env) > 0:
            onset_env /= np.max(onset_env)

        return onset_env

    @staticmethod
    def _autocorrelation_method(
        audio: np.ndarray, sr: int, min_bpm: float, max_bpm: float
    ) -> BPMResult:
        """BPM detection via autocorrelation of onset envelope."""
        hop_length = 512
        onset_env = BPMDetector._compute_onset_envelope(audio, sr, hop_length)

        if len(onset_env) < 4:
            return BPMResult(120.0, 0.0, [], "autocorrelation")

        # Autocorrelation
        onset_centered = onset_env - np.mean(onset_env)
        n = len(onset_centered)
        autocorr = np.correlate(onset_centered, onset_centered, mode="full")
        autocorr = autocorr[n - 1 :]  # Keep positive lags only

        # Normalize
        if autocorr[0] > 0:
            autocorr /= autocorr[0]

        # Convert BPM range to lag range (in frames)
        fps = sr / hop_length  # Frames per second
        min_lag = int(60.0 / max_bpm * fps)
        max_lag = int(60.0 / min_bpm * fps)
        max_lag = min(max_lag, len(autocorr) - 1)
        min_lag = max(1, min_lag)

        if min_lag >= max_lag:
            return BPMResult(120.0, 0.0, [], "autocorrelation")

        # Find peaks in autocorrelation
        search_region = autocorr[min_lag : max_lag + 1]
        candidates = []

        for i in range(1, len(search_region) - 1):
            if search_region[i] > search_region[i - 1] and search_region[i] > search_region[i + 1]:
                lag = i + min_lag
                bpm = 60.0 * fps / lag
                strength = float(search_region[i])
                candidates.append((bpm, strength))

        if not candidates:
            # Fallback: take the maximum
            peak_idx = np.argmax(search_region)
            lag = peak_idx + min_lag
            bpm = 60.0 * fps / lag
            return BPMResult(
                round(bpm, 1), 0.3, [(bpm, float(search_region[peak_idx]))], "autocorrelation"
            )

        # Sort by strength
        candidates.sort(key=lambda x: x[1], reverse=True)

        best_bpm = candidates[0][0]
        confidence = min(1.0, candidates[0][1])

        # Round to common BPM values if close
        for common in [
            60,
            70,
            75,
            80,
            85,
            90,
            95,
            100,
            105,
            110,
            115,
            120,
            125,
            128,
            130,
            135,
            140,
            145,
            150,
            155,
            160,
            170,
            174,
            175,
            180,
        ]:
            if abs(best_bpm - common) < 1.5:
                best_bpm = float(common)
                break

        return BPMResult(
            bpm=round(best_bpm, 1),
            confidence=round(confidence, 3),
            bpm_candidates=[(round(b, 1), round(s, 3)) for b, s in candidates[:5]],
            method="autocorrelation",
        )

    @staticmethod
    def _onset_interval_method(
        audio: np.ndarray, sr: int, min_bpm: float, max_bpm: float
    ) -> BPMResult:
        """BPM detection via inter-onset intervals."""
        hop_length = 512
        onset_env = BPMDetector._compute_onset_envelope(audio, sr, hop_length)
        fps = sr / hop_length

        # Peak picking on onset envelope
        threshold = np.mean(onset_env) + 0.5 * np.std(onset_env)
        min_distance = int(0.1 * fps)  # Minimum 100ms between onsets

        peaks: list[int] = []
        for i in range(1, len(onset_env) - 1):
            if (
                onset_env[i] > onset_env[i - 1]
                and onset_env[i] > onset_env[i + 1]
                and onset_env[i] > threshold
            ) and (not peaks or (i - peaks[-1]) >= min_distance):
                peaks.append(i)

        if len(peaks) < 3:
            return BPMResult(120.0, 0.0, [], "onset_intervals")

        # Compute inter-onset intervals
        iois = np.diff(peaks).astype(float)
        ioi_bpms = 60.0 * fps / iois

        # Filter to valid BPM range (and allow half/double)
        valid_bpms = ioi_bpms[(ioi_bpms >= min_bpm * 0.5) & (ioi_bpms <= max_bpm * 2)]

        if len(valid_bpms) == 0:
            return BPMResult(120.0, 0.0, [], "onset_intervals")

        # Histogram-based estimation
        bins = np.arange(min_bpm, max_bpm + 1, 1.0)
        hist, edges = np.histogram(valid_bpms, bins=bins)

        if np.max(hist) == 0:
            return BPMResult(120.0, 0.0, [], "onset_intervals")

        peak_idx = np.argmax(hist)
        best_bpm = (edges[peak_idx] + edges[peak_idx + 1]) / 2
        confidence = hist[peak_idx] / len(valid_bpms)

        # Get top candidates
        top_indices = np.argsort(hist)[-5:][::-1]
        candidates = [
            (float((edges[i] + edges[i + 1]) / 2), float(hist[i] / len(valid_bpms)))
            for i in top_indices
            if hist[i] > 0
        ]

        return BPMResult(
            bpm=round(best_bpm, 1),
            confidence=round(confidence, 3),
            bpm_candidates=candidates,
            method="onset_intervals",
        )

    @staticmethod
    def detect_batch(filepaths: list[str | Path], **kwargs) -> dict[str, BPMResult]:
        """Detect BPM for multiple files.

        Returns:
            Dict mapping filename -> BPMResult
        """
        results = {}
        for fp in filepaths:
            path = Path(fp)
            try:
                results[path.name] = BPMDetector.detect(fp, **kwargs)
            except Exception as e:
                results[path.name] = BPMResult(0.0, 0.0, [], f"error: {e}")
        return results
