"""Stereo Tools - Width, imaging, and mid/side processing.

Provides mid/side encoding/decoding, stereo width control, channel
balance, mono compatibility checking, and stereo enhancement effects.
"""

import numpy as np


class StereoTools:
    """Stereo field manipulation and analysis.

    All methods accept 2D stereo arrays (samples x 2) unless noted.
    """

    # ─── Mid/Side Processing ───

    @staticmethod
    def to_mid_side(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode stereo to mid/side.

        Args:
            stereo: Stereo audio (N, 2).

        Returns:
            Tuple of (mid, side) as 1D arrays.
        """
        left = stereo[:, 0]
        right = stereo[:, 1]
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        return mid, side

    @staticmethod
    def from_mid_side(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
        """Decode mid/side back to stereo.

        Args:
            mid: Mid signal (1D).
            side: Side signal (1D).

        Returns:
            Stereo audio (N, 2).
        """
        left = mid + side
        right = mid - side
        return np.column_stack([left, right])

    @classmethod
    def mid_side_process(
        cls, stereo: np.ndarray, mid_gain_db: float = 0.0, side_gain_db: float = 0.0
    ) -> np.ndarray:
        """Process mid and side channels independently.

        Args:
            stereo: Input stereo audio (N, 2).
            mid_gain_db: Gain adjustment for mid channel.
            side_gain_db: Gain adjustment for side channel.

        Returns:
            Processed stereo audio.
        """
        mid, side = cls.to_mid_side(stereo)

        if mid_gain_db != 0.0:
            mid = mid * (10 ** (mid_gain_db / 20))
        if side_gain_db != 0.0:
            side = side * (10 ** (side_gain_db / 20))

        return cls.from_mid_side(mid, side)

    # ─── Stereo Width ───

    @classmethod
    def width(cls, stereo: np.ndarray, amount: float = 1.0) -> np.ndarray:
        """Adjust stereo width.

        Args:
            stereo: Input stereo audio (N, 2).
            amount: Width multiplier.
                    0.0 = mono, 1.0 = original, 2.0 = extra wide.

        Returns:
            Width-adjusted stereo audio.
        """
        mid, side = cls.to_mid_side(stereo)
        side = side * amount
        return cls.from_mid_side(mid, side)

    @staticmethod
    def to_mono(stereo: np.ndarray) -> np.ndarray:
        """Convert stereo to mono (summed).

        Returns:
            1D mono audio.
        """
        return (stereo[:, 0] + stereo[:, 1]) * 0.5

    @staticmethod
    def mono_to_stereo(mono: np.ndarray) -> np.ndarray:
        """Convert mono to stereo (dual mono).

        Returns:
            Stereo audio (N, 2).
        """
        return np.column_stack([mono, mono])

    # ─── Balance and Pan ───

    @staticmethod
    def balance(stereo: np.ndarray, position: float = 0.0) -> np.ndarray:
        """Adjust stereo balance.

        Args:
            stereo: Input stereo audio (N, 2).
            position: Balance position. -1.0 = full left, 0.0 = center,
                      1.0 = full right.

        Returns:
            Balance-adjusted stereo audio.
        """
        result = np.copy(stereo)
        if position < 0:
            # Attenuate right
            result[:, 1] *= 1.0 + position
        elif position > 0:
            # Attenuate left
            result[:, 0] *= 1.0 - position
        return result

    @staticmethod
    def pan_constant_power(mono: np.ndarray, pan: float = 0.5) -> np.ndarray:
        """Pan a mono signal using constant-power pan law.

        Args:
            mono: Mono audio (1D).
            pan: Pan position. 0.0 = full left, 0.5 = center, 1.0 = full right.

        Returns:
            Panned stereo audio (N, 2).
        """
        left_gain = np.cos(pan * np.pi / 2)
        right_gain = np.sin(pan * np.pi / 2)
        return np.column_stack([mono * left_gain, mono * right_gain])

    @staticmethod
    def swap_channels(stereo: np.ndarray) -> np.ndarray:
        """Swap left and right channels."""
        return stereo[:, ::-1]

    # ─── Analysis ───

    @classmethod
    def correlation(cls, stereo: np.ndarray) -> float:
        """Measure stereo correlation (mono compatibility).

        Returns:
            Correlation coefficient:
             1.0 = perfectly correlated (mono)
             0.0 = uncorrelated
            -1.0 = perfectly anti-correlated (phase cancellation)
        """
        left = stereo[:, 0]
        right = stereo[:, 1]

        # Pearson correlation
        l_mean = np.mean(left)
        r_mean = np.mean(right)
        l_dev = left - l_mean
        r_dev = right - r_mean

        numerator = np.sum(l_dev * r_dev)
        denominator = np.sqrt(np.sum(l_dev**2) * np.sum(r_dev**2))

        if denominator < 1e-10:
            return 1.0  # Silence is "correlated"
        return float(numerator / denominator)

    @classmethod
    def stereo_width_meter(cls, stereo: np.ndarray) -> float:
        """Estimate perceived stereo width.

        Returns:
            Width value: 0.0 = mono, 1.0 = normal stereo, >1.0 = wide.
        """
        mid, side = cls.to_mid_side(stereo)
        mid_rms = np.sqrt(np.mean(mid**2))
        side_rms = np.sqrt(np.mean(side**2))

        if mid_rms < 1e-10:
            return 0.0
        return float(side_rms / mid_rms)

    @classmethod
    def mono_compatibility_check(cls, stereo: np.ndarray) -> dict:
        """Check for mono compatibility issues.

        Returns:
            Dict with mono compatibility metrics and warnings.
        """
        mono_sum = cls.to_mono(stereo)
        stereo_rms = np.sqrt(np.mean(stereo**2))
        mono_rms = np.sqrt(np.mean(mono_sum**2))

        if stereo_rms < 1e-10:
            loss_db = 0.0
        else:
            loss_db = 20 * np.log10(max(mono_rms, 1e-10)) - 20 * np.log10(max(stereo_rms, 1e-10))

        corr = cls.correlation(stereo)
        width = cls.stereo_width_meter(stereo)

        issues = []
        if corr < 0:
            issues.append("Negative correlation - significant phase cancellation in mono")
        elif corr < 0.3:
            issues.append("Low correlation - some phase issues possible in mono")
        if loss_db < -3.0:
            issues.append(f"Significant level loss in mono ({loss_db:.1f} dB)")
        if width > 1.5:
            issues.append("Very wide stereo - may collapse in mono")

        return {
            "correlation": round(corr, 3),
            "width": round(width, 3),
            "mono_loss_db": round(loss_db, 1),
            "mono_compatible": len(issues) == 0,
            "issues": issues,
        }

    @classmethod
    def channel_levels(cls, stereo: np.ndarray) -> dict:
        """Measure individual channel levels.

        Returns:
            Dict with left/right peak and RMS in dB.
        """
        left = stereo[:, 0]
        right = stereo[:, 1]

        l_peak = float(np.max(np.abs(left)))
        r_peak = float(np.max(np.abs(right)))
        l_rms = float(np.sqrt(np.mean(left**2)))
        r_rms = float(np.sqrt(np.mean(right**2)))

        return {
            "left_peak_db": 20 * np.log10(max(l_peak, 1e-10)),
            "right_peak_db": 20 * np.log10(max(r_peak, 1e-10)),
            "left_rms_db": 20 * np.log10(max(l_rms, 1e-10)),
            "right_rms_db": 20 * np.log10(max(r_rms, 1e-10)),
            "balance_db": 20 * np.log10(max(r_rms, 1e-10)) - 20 * np.log10(max(l_rms, 1e-10)),
        }

    # ─── Effects ───

    @classmethod
    def haas_delay(
        cls, stereo: np.ndarray, sr: int, delay_ms: float = 15.0, side: str = "right"
    ) -> np.ndarray:
        """Apply Haas effect (short delay on one channel for width).

        Args:
            stereo: Input stereo audio.
            sr: Sample rate.
            delay_ms: Delay time (1-30ms typical).
            side: Which channel to delay ("left" or "right").

        Returns:
            Processed stereo audio.
        """
        delay_samples = int(delay_ms * 0.001 * sr)
        result = np.copy(stereo)

        ch = 1 if side == "right" else 0
        delayed = np.zeros(len(stereo[:, ch]) + delay_samples)
        delayed[delay_samples:] = stereo[:, ch]
        result[:, ch] = delayed[: len(stereo)]

        return result

    @classmethod
    def stereo_enhance(
        cls,
        stereo: np.ndarray,
        sr: int,
        low_width: float = 0.5,
        high_width: float = 1.5,
        crossover_freq: float = 300.0,
    ) -> np.ndarray:
        """Frequency-dependent stereo enhancement.

        Narrows low frequencies for a tight bottom end and widens highs
        for an open, spacious feel.

        Args:
            stereo: Input stereo audio (N, 2).
            sr: Sample rate.
            low_width: Width for content below crossover (0.0-2.0).
            high_width: Width for content above crossover (0.0-2.0).
            crossover_freq: Frequency split point in Hz.

        Returns:
            Enhanced stereo audio.
        """
        from mixing._biquad import biquad_filter

        # Split into low and high bands
        low = biquad_filter(stereo, sr, "lowpass", crossover_freq, 0.707, 0.0)
        high = stereo - low

        # Apply different widths
        low_processed = cls.width(low, low_width)
        high_processed = cls.width(high, high_width)

        return low_processed + high_processed

    def report(self, stereo: np.ndarray, name: str = "Signal") -> str:
        """Generate a stereo analysis report."""
        compat = self.mono_compatibility_check(stereo)
        levels = self.channel_levels(stereo)

        lines = [
            f"=== Stereo Analysis: {name} ===",
            f"  Left  Peak: {levels['left_peak_db']:+.1f} dBFS | RMS: {levels['left_rms_db']:+.1f} dBFS",
            f"  Right Peak: {levels['right_peak_db']:+.1f} dBFS | RMS: {levels['right_rms_db']:+.1f} dBFS",
            f"  Balance:    {levels['balance_db']:+.1f} dB {'(centered)' if abs(levels['balance_db']) < 1.0 else ''}",
            f"  Correlation: {compat['correlation']:.3f}",
            f"  Width:       {compat['width']:.3f}",
            f"  Mono Loss:   {compat['mono_loss_db']:+.1f} dB",
            f"  Mono Safe:   {'YES' if compat['mono_compatible'] else 'NO'}",
        ]
        if compat["issues"]:
            lines.append("  Issues:")
            for issue in compat["issues"]:
                lines.append(f"    - {issue}")
        return "\n".join(lines)
