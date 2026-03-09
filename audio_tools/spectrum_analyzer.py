"""Spectrum Analyzer - FFT spectral analysis and visualization.

Provides spectral analysis tools including FFT, spectrogram,
frequency band analysis, and visualization output.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class SpectrumData:
    """FFT spectrum analysis result."""

    frequencies: np.ndarray
    magnitudes: np.ndarray  # In dB
    phase: np.ndarray
    sample_rate: int
    fft_size: int

    @property
    def peak_frequency(self) -> float:
        """Frequency with the highest magnitude."""
        idx = np.argmax(self.magnitudes)
        return float(self.frequencies[idx])

    @property
    def peak_magnitude_db(self) -> float:
        return float(np.max(self.magnitudes))

    def get_band_energy(self, low_hz: float, high_hz: float) -> float:
        """Get average energy in dB for a frequency band."""
        mask = (self.frequencies >= low_hz) & (self.frequencies <= high_hz)
        if not np.any(mask):
            return -120.0
        return float(np.mean(self.magnitudes[mask]))


@dataclass
class BandAnalysis:
    """Frequency band energy analysis."""

    sub_bass: float = 0.0  # 20-60 Hz
    bass: float = 0.0  # 60-250 Hz
    low_mid: float = 0.0  # 250-500 Hz
    mid: float = 0.0  # 500-2000 Hz
    upper_mid: float = 0.0  # 2000-4000 Hz
    presence: float = 0.0  # 4000-6000 Hz
    brilliance: float = 0.0  # 6000-20000 Hz

    def as_dict(self) -> dict[str, float]:
        return {
            "Sub Bass (20-60Hz)": self.sub_bass,
            "Bass (60-250Hz)": self.bass,
            "Low Mid (250-500Hz)": self.low_mid,
            "Mid (500-2kHz)": self.mid,
            "Upper Mid (2-4kHz)": self.upper_mid,
            "Presence (4-6kHz)": self.presence,
            "Brilliance (6-20kHz)": self.brilliance,
        }

    def text_display(self, width: int = 40) -> str:
        """Text-based frequency band display."""
        bands = self.as_dict()
        max_val = max(bands.values())
        min_val = min(bands.values())
        val_range = max_val - min_val if max_val > min_val else 1

        lines = []
        for name, val in bands.items():
            normalized = (val - min_val) / val_range
            bar_len = int(normalized * width)
            bar = "#" * bar_len
            lines.append(f"{name:>25} | {bar} ({val:.1f} dB)")
        return "\n".join(lines)


class SpectrumAnalyzer:
    """FFT-based audio spectrum analyzer."""

    @staticmethod
    def analyze(
        filepath: str | Path, fft_size: int = 4096, window: str = "hanning", channel: int = 0
    ) -> SpectrumData:
        """Compute the average spectrum of an audio file.

        Args:
            filepath: Path to audio file
            fft_size: FFT window size
            window: Window function (hanning, hamming, blackman, rectangular)
            channel: Channel to analyze (0=left/mono)
        """
        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
        except Exception as e:
            raise OSError(f"Failed to read audio file '{filepath}': {e}") from e

        if audio.ndim > 1:
            audio = audio[:, min(channel, audio.shape[1] - 1)]

        return SpectrumAnalyzer.analyze_array(audio, sr, fft_size, window)

    @staticmethod
    def analyze_array(
        audio: np.ndarray, sr: int, fft_size: int = 4096, window: str = "hanning"
    ) -> SpectrumData:
        """Compute average spectrum from audio array."""
        # Window function
        if window == "hamming":
            win = np.hamming(fft_size)
        elif window == "blackman":
            win = np.blackman(fft_size)
        elif window == "rectangular":
            win = np.ones(fft_size)
        else:
            win = np.hanning(fft_size)

        hop = fft_size // 2
        n_frames = max(1, (len(audio) - fft_size) // hop)

        # Accumulate magnitude spectrum
        magnitude_sum = np.zeros(fft_size // 2 + 1)
        phase_sum = np.zeros(fft_size // 2 + 1)

        for i in range(n_frames):
            start = i * hop
            frame = audio[start : start + fft_size]
            if len(frame) < fft_size:
                frame = np.pad(frame, (0, fft_size - len(frame)))

            fft_result = np.fft.rfft(frame * win)
            magnitude_sum += np.abs(fft_result)
            phase_sum += np.angle(fft_result)

        # Average
        magnitude_avg = magnitude_sum / max(1, n_frames)
        phase_avg = phase_sum / max(1, n_frames)

        # Convert to dB
        magnitude_db = 20 * np.log10(np.maximum(magnitude_avg, 1e-10))

        frequencies = np.fft.rfftfreq(fft_size, 1.0 / sr)

        return SpectrumData(
            frequencies=frequencies,
            magnitudes=magnitude_db,
            phase=phase_avg,
            sample_rate=sr,
            fft_size=fft_size,
        )

    @staticmethod
    def band_analysis(filepath: str | Path, fft_size: int = 4096) -> BandAnalysis:
        """Analyze frequency band energy distribution."""
        spectrum = SpectrumAnalyzer.analyze(filepath, fft_size)

        return BandAnalysis(
            sub_bass=spectrum.get_band_energy(20, 60),
            bass=spectrum.get_band_energy(60, 250),
            low_mid=spectrum.get_band_energy(250, 500),
            mid=spectrum.get_band_energy(500, 2000),
            upper_mid=spectrum.get_band_energy(2000, 4000),
            presence=spectrum.get_band_energy(4000, 6000),
            brilliance=spectrum.get_band_energy(6000, 20000),
        )

    @staticmethod
    def spectrogram(
        filepath: str | Path, fft_size: int = 2048, hop_size: int = 512, window: str = "hanning"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram (time-frequency representation).

        Returns:
            Tuple of (times, frequencies, magnitude_db_matrix)
        """
        try:
            audio, sr = sf.read(str(filepath), dtype="float32")
        except Exception as e:
            raise OSError(f"Failed to read audio file '{filepath}': {e}") from e

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        windows = {
            "hanning": np.hanning,
            "hamming": np.hamming,
            "blackman": np.blackman,
        }
        win = windows.get(window, np.hanning)(fft_size)

        n_frames = max(1, (len(audio) - fft_size) // hop_size)
        n_bins = fft_size // 2 + 1

        spec = np.zeros((n_bins, n_frames))

        for i in range(n_frames):
            start = i * hop_size
            frame = audio[start : start + fft_size]
            if len(frame) < fft_size:
                frame = np.pad(frame, (0, fft_size - len(frame)))
            fft_result = np.abs(np.fft.rfft(frame * win))
            spec[:, i] = 20 * np.log10(np.maximum(fft_result, 1e-10))

        frequencies = np.fft.rfftfreq(fft_size, 1.0 / sr)
        times = np.arange(n_frames) * hop_size / sr

        return times, frequencies, spec

    @staticmethod
    def save_plot(
        filepath: str | Path, output_path: str | Path, plot_type: str = "spectrum", **kwargs
    ) -> None:
        """Save a spectrum or spectrogram plot to an image file.

        Args:
            filepath: Audio file path
            output_path: Output image path (.png)
            plot_type: "spectrum", "spectrogram", or "bands"
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if plot_type == "spectrum":
            data = SpectrumAnalyzer.analyze(filepath, **kwargs)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.semilogx(data.frequencies[1:], data.magnitudes[1:])
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_title(f"Spectrum: {Path(filepath).name}")
            ax.set_xlim(20, data.sample_rate // 2)
            ax.grid(True, alpha=0.3)

        elif plot_type == "spectrogram":
            times, freqs, spec = SpectrumAnalyzer.spectrogram(filepath, **kwargs)
            fig, ax = plt.subplots(figsize=(14, 6))
            im = ax.pcolormesh(
                times, freqs, spec, shading="auto", cmap="magma", vmin=np.max(spec) - 80
            )
            ax.set_ylabel("Frequency (Hz)")
            ax.set_xlabel("Time (s)")
            ax.set_title(f"Spectrogram: {Path(filepath).name}")
            ax.set_ylim(20, 20000)
            ax.set_yscale("log")
            plt.colorbar(im, ax=ax, label="dB")

        elif plot_type == "bands":
            bands = SpectrumAnalyzer.band_analysis(filepath)
            band_dict = bands.as_dict()
            fig, ax = plt.subplots(figsize=(10, 6))
            names = list(band_dict.keys())
            values = list(band_dict.values())
            cmap = plt.get_cmap("viridis")
            colors = cmap(np.linspace(0.2, 0.8, len(names)))
            ax.barh(names, values, color=colors)
            ax.set_xlabel("Energy (dB)")
            ax.set_title(f"Band Analysis: {Path(filepath).name}")
            ax.grid(True, axis="x", alpha=0.3)

        else:
            return

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150)
        plt.close()
