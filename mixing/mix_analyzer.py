"""Mix Analyzer - Loudness, dynamics, and spectral balance analysis.

Provides LUFS loudness measurement, crest factor analysis, frequency
balance checking, and dynamic range assessment for evaluating mix
quality.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LoudnessResult:
    """Loudness measurement results."""

    integrated_lufs: float
    short_term_max_lufs: float
    momentary_max_lufs: float
    true_peak_dbfs: float
    loudness_range_lu: float


@dataclass
class DynamicsResult:
    """Dynamic range analysis."""

    peak_db: float
    rms_db: float
    crest_factor_db: float
    dynamic_range_db: float
    quiet_sections_db: float
    loud_sections_db: float


@dataclass
class SpectralBalance:
    """Frequency balance analysis across standard bands."""

    sub_bass_db: float  # 20-60 Hz
    bass_db: float  # 60-250 Hz
    low_mid_db: float  # 250-500 Hz
    mid_db: float  # 500-2000 Hz
    upper_mid_db: float  # 2000-4000 Hz
    presence_db: float  # 4000-8000 Hz
    brilliance_db: float  # 8000-20000 Hz


class MixAnalyzer:
    """Comprehensive mix analysis toolkit.

    Measures loudness (approximated LUFS), dynamics, spectral balance,
    and provides mixing insights.
    """

    # Standard frequency bands
    BANDS = {
        "sub_bass": (20, 60),
        "bass": (60, 250),
        "low_mid": (250, 500),
        "mid": (500, 2000),
        "upper_mid": (2000, 4000),
        "presence": (4000, 8000),
        "brilliance": (8000, 20000),
    }

    # ─── Loudness Measurement ───

    @classmethod
    def _k_weight(cls, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply K-weighting filter (simplified) for LUFS measurement.

        Approximates ITU-R BS.1770 K-weighting with a high-shelf boost
        at ~1500 Hz and a high-pass at ~38 Hz.
        """
        from mixing._biquad import biquad_filter

        # Stage 1: High shelf boost at ~1500 Hz (+4 dB)
        weighted = biquad_filter(audio, sr, "highshelf", 1500.0, 0.707, 4.0)
        # Stage 2: High-pass at ~38 Hz (remove sub-lows)
        weighted = biquad_filter(weighted, sr, "highpass", 38.0, 0.5, 0.0)

        return weighted

    @classmethod
    def lufs_integrated(cls, audio: np.ndarray, sr: int) -> float:
        """Measure integrated loudness in LUFS (approximation).

        Args:
            audio: Input audio (1D mono or 2D stereo).
            sr: Sample rate.

        Returns:
            Integrated loudness in LUFS.
        """
        weighted = cls._k_weight(audio, sr)

        # Mean square of K-weighted signal
        if weighted.ndim == 1:
            ms = np.mean(weighted**2)
        else:
            # Sum channels (with channel weighting for surround - L/R equal)
            ms = 0.0
            for ch in range(weighted.shape[1]):
                ms += np.mean(weighted[:, ch] ** 2)

        if ms < 1e-20:
            return -70.0  # Silence floor

        return float(-0.691 + 10 * np.log10(ms))

    @classmethod
    def lufs_momentary(cls, audio: np.ndarray, sr: int, window_ms: float = 400.0) -> np.ndarray:
        """Calculate momentary loudness over time (400ms windows).

        Returns:
            Array of LUFS values, one per window.
        """
        weighted = cls._k_weight(audio, sr)
        window_samples = int(window_ms * 0.001 * sr)

        if weighted.ndim == 1:
            weighted = weighted.reshape(-1, 1)

        n_windows = max(1, len(weighted) // window_samples)
        lufs_values = np.zeros(n_windows)

        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            segment = weighted[start:end]

            ms = 0.0
            for ch in range(segment.shape[1]):
                ms += np.mean(segment[:, ch] ** 2)

            if ms < 1e-20:
                lufs_values[i] = -70.0
            else:
                lufs_values[i] = -0.691 + 10 * np.log10(ms)

        return lufs_values

    @classmethod
    def lufs_short_term(cls, audio: np.ndarray, sr: int, window_ms: float = 3000.0) -> np.ndarray:
        """Calculate short-term loudness (3s windows)."""
        return cls.lufs_momentary(audio, sr, window_ms)

    @classmethod
    def measure_loudness(cls, audio: np.ndarray, sr: int) -> LoudnessResult:
        """Complete loudness measurement."""
        integrated = cls.lufs_integrated(audio, sr)
        momentary = cls.lufs_momentary(audio, sr, 400.0)
        short_term = cls.lufs_momentary(audio, sr, 3000.0)

        # True peak (oversampled)
        true_peak = float(np.max(np.abs(audio)))
        true_peak_db = 20 * np.log10(max(true_peak, 1e-10))

        # Loudness range (difference between quiet and loud short-term)
        # Exclude silence (below -70 LUFS)
        valid_st = short_term[short_term > -70.0]
        if len(valid_st) > 2:
            # LRA: 10th to 95th percentile range
            low = float(np.percentile(valid_st, 10))
            high = float(np.percentile(valid_st, 95))
            lra = high - low
        else:
            lra = 0.0

        return LoudnessResult(
            integrated_lufs=round(integrated, 1),
            short_term_max_lufs=round(float(np.max(short_term)), 1),
            momentary_max_lufs=round(float(np.max(momentary)), 1),
            true_peak_dbfs=round(true_peak_db, 1),
            loudness_range_lu=round(lra, 1),
        )

    # ─── Dynamic Range ───

    @classmethod
    def analyze_dynamics(
        cls, audio: np.ndarray, sr: int, window_ms: float = 50.0
    ) -> DynamicsResult:
        """Analyze dynamic range characteristics.

        Args:
            audio: Input audio.
            sr: Sample rate.
            window_ms: Analysis window size.

        Returns:
            DynamicsResult with measurements.
        """
        peak = float(np.max(np.abs(audio)))
        peak_db = 20 * np.log10(max(peak, 1e-10))

        rms = float(np.sqrt(np.mean(audio**2)))
        rms_db = 20 * np.log10(max(rms, 1e-10))

        crest = peak_db - rms_db

        # Windowed RMS for dynamic range
        window_samples = int(window_ms * 0.001 * sr)
        mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio

        n_windows = max(1, len(mono) // window_samples)
        rms_windows = np.zeros(n_windows)

        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            seg = mono[start:end]
            rms_windows[i] = float(np.sqrt(np.mean(seg**2)))

        rms_db_windows = 20 * np.log10(np.maximum(rms_windows, 1e-10))

        # Exclude silence
        active = rms_db_windows[rms_db_windows > -60.0]

        if len(active) > 2:
            quiet = float(np.percentile(active, 10))
            loud = float(np.percentile(active, 90))
            dynamic_range = loud - quiet
        else:
            quiet = rms_db
            loud = rms_db
            dynamic_range = 0.0

        return DynamicsResult(
            peak_db=round(peak_db, 1),
            rms_db=round(rms_db, 1),
            crest_factor_db=round(crest, 1),
            dynamic_range_db=round(dynamic_range, 1),
            quiet_sections_db=round(quiet, 1),
            loud_sections_db=round(loud, 1),
        )

    # ─── Spectral Analysis ───

    @classmethod
    def spectral_balance(cls, audio: np.ndarray, sr: int, fft_size: int = 4096) -> SpectralBalance:
        """Analyze frequency balance across standard mixing bands.

        Args:
            audio: Input audio.
            sr: Sample rate.
            fft_size: FFT window size.

        Returns:
            SpectralBalance with per-band energy in dB.
        """
        mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio

        # Compute averaged spectrum
        n_frames = max(1, len(mono) // fft_size)
        spectrum = np.zeros(fft_size // 2 + 1)

        for i in range(n_frames):
            start = i * fft_size
            end = start + fft_size
            if end > len(mono):
                break
            frame = mono[start:end] * np.hanning(fft_size)
            fft = np.abs(np.fft.rfft(frame)) ** 2
            spectrum += fft

        if n_frames > 0:
            spectrum /= n_frames

        freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)

        # Measure energy in each band
        band_energies = {}
        for band_name, (low, high) in cls.BANDS.items():
            mask = (freqs >= low) & (freqs < high)
            energy = np.sum(spectrum[mask])
            band_energies[band_name] = 10 * np.log10(max(energy, 1e-20))

        return SpectralBalance(**{f"{k}_db": round(v, 1) for k, v in band_energies.items()})

    @classmethod
    def frequency_masking_check(
        cls,
        track_a: np.ndarray,
        track_b: np.ndarray,
        sr: int,
        fft_size: int = 4096,
        threshold_db: float = 3.0,
    ) -> list[dict]:
        """Check for frequency masking between two tracks.

        Identifies frequency bands where two tracks have significant
        overlapping energy, which may cause masking issues.

        Args:
            track_a: First audio track.
            track_b: Second audio track.
            sr: Sample rate.
            fft_size: FFT size.
            threshold_db: Threshold for flagging overlap.

        Returns:
            List of dicts describing frequency ranges with masking issues.
        """

        def avg_spectrum(audio):
            mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
            n_frames = max(1, len(mono) // fft_size)
            spec = np.zeros(fft_size // 2 + 1)
            for i in range(n_frames):
                start = i * fft_size
                end = start + fft_size
                if end > len(mono):
                    break
                frame = mono[start:end] * np.hanning(fft_size)
                spec += np.abs(np.fft.rfft(frame)) ** 2
            return spec / max(n_frames, 1)

        spec_a = avg_spectrum(track_a)
        spec_b = avg_spectrum(track_b)
        freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)

        issues = []
        for band_name, (low, high) in cls.BANDS.items():
            mask = (freqs >= low) & (freqs < high)
            e_a = 10 * np.log10(max(np.sum(spec_a[mask]), 1e-20))
            e_b = 10 * np.log10(max(np.sum(spec_b[mask]), 1e-20))

            # Both tracks have significant energy in this band
            if e_a > -40 and e_b > -40 and abs(e_a - e_b) < threshold_db:
                issues.append(
                    {
                        "band": band_name,
                        "freq_range": f"{low}-{high} Hz",
                        "track_a_db": round(e_a, 1),
                        "track_b_db": round(e_b, 1),
                        "suggestion": f"Consider EQ separation in {band_name} range ({low}-{high} Hz)",
                    }
                )

        return issues

    # ─── Full Mix Analysis ───

    @classmethod
    def full_analysis(cls, audio: np.ndarray, sr: int) -> dict:
        """Run complete mix analysis.

        Returns:
            Dict containing loudness, dynamics, and spectral results.
        """
        loudness = cls.measure_loudness(audio, sr)
        dynamics = cls.analyze_dynamics(audio, sr)
        spectral = cls.spectral_balance(audio, sr)

        return {
            "loudness": loudness,
            "dynamics": dynamics,
            "spectral": spectral,
        }

    @classmethod
    def report(cls, audio: np.ndarray, sr: int, name: str = "Mix") -> str:
        """Generate a complete mix analysis report."""
        loudness = cls.measure_loudness(audio, sr)
        dynamics = cls.analyze_dynamics(audio, sr)
        spectral = cls.spectral_balance(audio, sr)

        lines = [
            f"=== Mix Analysis: {name} ===",
            "",
            "--- Loudness ---",
            f"  Integrated:       {loudness.integrated_lufs:+.1f} LUFS",
            f"  Short-term Max:   {loudness.short_term_max_lufs:+.1f} LUFS",
            f"  Momentary Max:    {loudness.momentary_max_lufs:+.1f} LUFS",
            f"  True Peak:        {loudness.true_peak_dbfs:+.1f} dBFS",
            f"  Loudness Range:   {loudness.loudness_range_lu:.1f} LU",
            "",
            "--- Dynamics ---",
            f"  Peak:             {dynamics.peak_db:+.1f} dBFS",
            f"  RMS:              {dynamics.rms_db:+.1f} dBFS",
            f"  Crest Factor:     {dynamics.crest_factor_db:.1f} dB",
            f"  Dynamic Range:    {dynamics.dynamic_range_db:.1f} dB",
            f"  Quiet Sections:   {dynamics.quiet_sections_db:+.1f} dBFS",
            f"  Loud Sections:    {dynamics.loud_sections_db:+.1f} dBFS",
            "",
            "--- Spectral Balance ---",
            f"  Sub Bass (20-60):     {spectral.sub_bass_db:+.1f} dB",
            f"  Bass (60-250):        {spectral.bass_db:+.1f} dB",
            f"  Low Mid (250-500):    {spectral.low_mid_db:+.1f} dB",
            f"  Mid (500-2k):         {spectral.mid_db:+.1f} dB",
            f"  Upper Mid (2k-4k):    {spectral.upper_mid_db:+.1f} dB",
            f"  Presence (4k-8k):     {spectral.presence_db:+.1f} dB",
            f"  Brilliance (8k-20k):  {spectral.brilliance_db:+.1f} dB",
        ]

        # Platform target checks
        targets = {
            "Spotify/YouTube": -14.0,
            "Apple Music": -16.0,
            "Amazon Music": -14.0,
            "Broadcast (EBU R128)": -23.0,
        }

        lines.append("")
        lines.append("--- Platform Targets ---")
        for platform, target in targets.items():
            diff = loudness.integrated_lufs - target
            status = "OK" if abs(diff) < 1.5 else ("LOUD" if diff > 0 else "QUIET")
            lines.append(f"  {platform}: {target:.0f} LUFS → {status} ({diff:+.1f})")

        return "\n".join(lines)
