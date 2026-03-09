"""Reference Compare - A/B comparison against reference tracks.

Compare your mix against professionally mastered reference tracks to
evaluate loudness, spectral balance, dynamics, and stereo width
differences.
"""

from dataclasses import dataclass

import numpy as np
import soundfile as sf

from mixing.gain_staging import GainStaging
from mixing.mix_analyzer import MixAnalyzer
from mixing.stereo_tools import StereoTools


@dataclass
class ComparisonResult:
    """Results from comparing mix against reference."""

    loudness_diff_lufs: float
    peak_diff_db: float
    spectral_diffs: dict[str, float]  # Band name -> dB difference
    dynamics_diff_db: float
    width_diff: float
    correlation_diff: float
    suggestions: list[str]


class ReferenceCompare:
    """Compare a mix against reference tracks.

    Loads reference audio and provides detailed A/B analysis covering
    loudness, spectral balance, dynamics, and stereo imaging differences.
    """

    def __init__(self):
        self._references: dict[str, tuple[np.ndarray, int]] = {}

    def load_reference(self, name: str, path: str, normalize: bool = True) -> dict:
        """Load a reference track from file.

        Args:
            name: Label for this reference.
            path: Path to audio file.
            normalize: If True, normalize to -1 dBFS peak for fair comparison.

        Returns:
            Dict with reference info (duration, sample rate, peak, RMS).
        """
        try:
            audio, sr = sf.read(path, dtype="float64")
        except Exception as e:
            raise OSError(f"Failed to read reference file '{path}': {e}") from e

        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])

        if normalize:
            peak = np.max(np.abs(audio))
            if peak > 1e-10:
                target_peak = 10 ** (-1.0 / 20)
                audio = audio * (target_peak / peak)

        self._references[name] = (audio, sr)

        peak_db = 20 * np.log10(max(np.max(np.abs(audio)), 1e-10))
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(max(rms, 1e-10))

        return {
            "name": name,
            "duration_s": round(len(audio) / sr, 2),
            "sample_rate": sr,
            "peak_db": round(peak_db, 1),
            "rms_db": round(rms_db, 1),
            "channels": audio.shape[1],
        }

    def load_reference_array(self, name: str, audio: np.ndarray, sr: int):
        """Load a reference from a numpy array."""
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        self._references[name] = (audio.copy(), sr)

    @property
    def reference_names(self) -> list[str]:
        return list(self._references.keys())

    def compare(
        self, mix: np.ndarray, sr: int, reference_name: str | None = None
    ) -> ComparisonResult:
        """Compare mix against a reference track.

        Args:
            mix: Your mix audio (1D or 2D).
            sr: Sample rate of your mix.
            reference_name: Which reference to compare against.
                            If None, uses the first loaded reference.

        Returns:
            ComparisonResult with detailed differences and suggestions.
        """
        if not self._references:
            raise ValueError("No reference tracks loaded. Use load_reference() first.")

        if reference_name is None:
            reference_name = list(self._references.keys())[0]

        ref_audio, ref_sr = self._references[reference_name]

        # Ensure stereo
        if mix.ndim == 1:
            mix = np.column_stack([mix, mix])

        # Match lengths for comparison (use shortest)
        min_len = min(len(mix), len(ref_audio))
        mix_seg = mix[:min_len]
        ref_seg = ref_audio[:min_len]

        # Loudness comparison
        mix_lufs = MixAnalyzer.lufs_integrated(mix_seg, sr)
        ref_lufs = MixAnalyzer.lufs_integrated(ref_seg, ref_sr)
        loudness_diff = mix_lufs - ref_lufs

        # Peak comparison
        mix_peak = GainStaging.peak_db(mix_seg)
        ref_peak = GainStaging.peak_db(ref_seg)
        peak_diff = mix_peak - ref_peak

        # Spectral balance comparison
        mix_spectral = MixAnalyzer.spectral_balance(mix_seg, sr)
        ref_spectral = MixAnalyzer.spectral_balance(ref_seg, ref_sr)

        spectral_diffs = {}
        for field_name in [
            "sub_bass_db",
            "bass_db",
            "low_mid_db",
            "mid_db",
            "upper_mid_db",
            "presence_db",
            "brilliance_db",
        ]:
            band = field_name.replace("_db", "")
            mix_val = getattr(mix_spectral, field_name)
            ref_val = getattr(ref_spectral, field_name)
            spectral_diffs[band] = round(mix_val - ref_val, 1)

        # Dynamics comparison
        mix_dynamics = MixAnalyzer.analyze_dynamics(mix_seg, sr)
        ref_dynamics = MixAnalyzer.analyze_dynamics(ref_seg, ref_sr)
        dynamics_diff = mix_dynamics.dynamic_range_db - ref_dynamics.dynamic_range_db

        # Stereo comparison
        mix_width = StereoTools.stereo_width_meter(mix_seg)
        ref_width = StereoTools.stereo_width_meter(ref_seg)
        width_diff = mix_width - ref_width

        mix_corr = StereoTools.correlation(mix_seg)
        ref_corr = StereoTools.correlation(ref_seg)
        corr_diff = mix_corr - ref_corr

        # Generate suggestions
        suggestions = self._generate_suggestions(
            loudness_diff,
            spectral_diffs,
            dynamics_diff,
            width_diff,
            mix_dynamics.crest_factor_db,
            ref_dynamics.crest_factor_db,
        )

        return ComparisonResult(
            loudness_diff_lufs=round(loudness_diff, 1),
            peak_diff_db=round(peak_diff, 1),
            spectral_diffs=spectral_diffs,
            dynamics_diff_db=round(dynamics_diff, 1),
            width_diff=round(width_diff, 3),
            correlation_diff=round(corr_diff, 3),
            suggestions=suggestions,
        )

    @staticmethod
    def _generate_suggestions(
        loudness_diff, spectral_diffs, dynamics_diff, width_diff, mix_crest, ref_crest
    ) -> list[str]:
        """Generate actionable mixing suggestions based on comparison."""
        suggestions = []

        # Loudness
        if loudness_diff < -3.0:
            suggestions.append(
                f"Mix is {abs(loudness_diff):.1f} LUFS quieter than reference. "
                "Consider increasing overall gain or adjusting limiter."
            )
        elif loudness_diff > 3.0:
            suggestions.append(
                f"Mix is {loudness_diff:.1f} LUFS louder than reference. "
                "May be over-compressed. Back off limiter/compressor."
            )

        # Spectral
        for band, diff in spectral_diffs.items():
            band_label = band.replace("_", " ").title()
            if diff > 4.0:
                suggestions.append(
                    f"{band_label} is {diff:.1f} dB louder than reference. "
                    "Consider cutting this range with EQ."
                )
            elif diff < -4.0:
                suggestions.append(
                    f"{band_label} is {abs(diff):.1f} dB quieter than reference. "
                    "Consider boosting this range with EQ."
                )

        # Dynamics
        if dynamics_diff > 5.0:
            suggestions.append(
                "Mix has more dynamic range than reference. "
                "May need more bus compression or limiting."
            )
        elif dynamics_diff < -5.0:
            suggestions.append(
                "Mix is more compressed than reference. "
                "Consider reducing compression for more dynamics."
            )

        # Crest factor
        crest_diff = mix_crest - ref_crest
        if crest_diff < -3.0:
            suggestions.append(
                "Mix has lower crest factor (less transient punch). "
                "Try reducing compression or using parallel compression."
            )

        # Width
        if width_diff > 0.3:
            suggestions.append(
                "Mix is wider than reference. " "Check mono compatibility and consider narrowing."
            )
        elif width_diff < -0.3:
            suggestions.append(
                "Mix is narrower than reference. " "Consider stereo enhancement or wider panning."
            )

        if not suggestions:
            suggestions.append("Mix closely matches reference profile.")

        return suggestions

    def compare_all(self, mix: np.ndarray, sr: int) -> dict[str, ComparisonResult]:
        """Compare mix against all loaded references.

        Returns:
            Dict mapping reference names to ComparisonResults.
        """
        results = {}
        for name in self._references:
            results[name] = self.compare(mix, sr, name)
        return results

    def match_loudness(
        self, mix: np.ndarray, sr: int, reference_name: str | None = None
    ) -> tuple[np.ndarray, float]:
        """Adjust mix gain to match reference loudness.

        Args:
            mix: Your mix audio.
            sr: Sample rate.
            reference_name: Target reference (or first loaded).

        Returns:
            Tuple of (adjusted audio, gain applied in dB).
        """
        if not self._references:
            raise ValueError("No reference tracks loaded.")

        if reference_name is None:
            reference_name = list(self._references.keys())[0]

        ref_audio, ref_sr = self._references[reference_name]

        mix_lufs = MixAnalyzer.lufs_integrated(mix, sr)
        ref_lufs = MixAnalyzer.lufs_integrated(ref_audio, ref_sr)

        gain_db = ref_lufs - mix_lufs
        gain_lin = 10 ** (gain_db / 20)

        return mix * gain_lin, round(gain_db, 2)

    def report(
        self, mix: np.ndarray, sr: int, reference_name: str | None = None, mix_name: str = "My Mix"
    ) -> str:
        """Generate a detailed comparison report."""
        if reference_name is None and self._references:
            reference_name = list(self._references.keys())[0]

        result = self.compare(mix, sr, reference_name)

        lines = [
            "=== Reference Comparison ===",
            f"  Mix: {mix_name}",
            f"  Reference: {reference_name}",
            "",
            "--- Loudness ---",
            f"  Difference: {result.loudness_diff_lufs:+.1f} LUFS",
            f"  Peak Diff:  {result.peak_diff_db:+.1f} dB",
            "",
            "--- Spectral Balance Differences ---",
        ]

        band_labels = {
            "sub_bass": "Sub Bass (20-60)",
            "bass": "Bass (60-250)",
            "low_mid": "Low Mid (250-500)",
            "mid": "Mid (500-2k)",
            "upper_mid": "Upper Mid (2k-4k)",
            "presence": "Presence (4k-8k)",
            "brilliance": "Brilliance (8k-20k)",
        }

        for band, diff in result.spectral_diffs.items():
            label = band_labels.get(band, band)
            bar = "+" * min(int(abs(diff)), 10) if diff > 0 else "-" * min(int(abs(diff)), 10)
            lines.append(f"  {label:.<28} {diff:+.1f} dB  {bar}")

        lines.extend(
            [
                "",
                "--- Dynamics ---",
                f"  Dynamic Range Diff: {result.dynamics_diff_db:+.1f} dB",
                "",
                "--- Stereo ---",
                f"  Width Diff:       {result.width_diff:+.3f}",
                f"  Correlation Diff: {result.correlation_diff:+.3f}",
                "",
                "--- Suggestions ---",
            ]
        )

        for i, suggestion in enumerate(result.suggestions, 1):
            lines.append(f"  {i}. {suggestion}")

        return "\n".join(lines)
