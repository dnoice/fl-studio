"""Gain Staging - Automatic level management for clean mixing.

Provides tools for setting proper gain structure throughout the mix,
ensuring headroom at every stage and preventing clipping or
excessive noise floor.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class GainReport:
    """Analysis report for a single audio signal."""

    peak_db: float
    rms_db: float
    crest_factor_db: float
    suggested_gain_db: float
    clipping: bool
    headroom_db: float
    dc_offset: float


class GainStaging:
    """Automatic gain staging and level management.

    Analyzes audio levels and suggests/applies gain adjustments to
    maintain proper headroom throughout the signal chain.

    Standard targets:
    - Individual tracks: -18 dBFS RMS (analog-style sweet spot)
    - Bus/group: -12 dBFS RMS
    - Master: -6 dBFS peak before limiting
    """

    # Common target levels
    TRACK_TARGET_RMS = -18.0  # dBFS
    BUS_TARGET_RMS = -12.0  # dBFS
    MASTER_TARGET_PEAK = -6.0  # dBFS

    def __init__(self, target_rms_db: float = -18.0, target_peak_db: float = -6.0):
        self.target_rms_db = target_rms_db
        self.target_peak_db = target_peak_db

    @staticmethod
    def peak_db(audio: np.ndarray) -> float:
        """Get peak level in dBFS."""
        peak = float(np.max(np.abs(audio)))
        return 20 * np.log10(max(peak, 1e-10))

    @staticmethod
    def rms_db(audio: np.ndarray) -> float:
        """Get RMS level in dBFS."""
        rms = float(np.sqrt(np.mean(audio**2)))
        return 20 * np.log10(max(rms, 1e-10))

    @staticmethod
    def crest_factor_db(audio: np.ndarray) -> float:
        """Get crest factor (peak-to-RMS ratio) in dB."""
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 1e-10:
            return 0.0
        return 20 * np.log10(peak / rms)

    @staticmethod
    def dc_offset(audio: np.ndarray) -> float:
        """Measure DC offset."""
        if audio.ndim == 1:
            return float(np.mean(audio))
        return float(np.mean(np.mean(audio, axis=1)))

    @staticmethod
    def remove_dc(audio: np.ndarray) -> np.ndarray:
        """Remove DC offset from audio."""
        if audio.ndim == 1:
            return audio - np.mean(audio)
        result = np.copy(audio)
        for ch in range(audio.shape[1]):
            result[:, ch] -= np.mean(audio[:, ch])
        return result

    def analyze(self, audio: np.ndarray) -> GainReport:
        """Analyze audio levels and suggest gain adjustment.

        Returns:
            GainReport with measurements and suggested gain.
        """
        peak = self.peak_db(audio)
        rms = self.rms_db(audio)
        crest = self.crest_factor_db(audio)
        dc = self.dc_offset(audio)

        # Suggest gain to hit target RMS
        suggested = self.target_rms_db - rms

        # But don't let peak exceed target peak
        if peak + suggested > self.target_peak_db:
            suggested = self.target_peak_db - peak

        clipping = peak >= -0.1
        headroom = -peak  # How much room below 0 dBFS

        return GainReport(
            peak_db=peak,
            rms_db=rms,
            crest_factor_db=crest,
            suggested_gain_db=round(suggested, 1),
            clipping=clipping,
            headroom_db=headroom,
            dc_offset=dc,
        )

    def auto_gain(self, audio: np.ndarray, mode: str = "rms") -> tuple[np.ndarray, float]:
        """Apply automatic gain staging.

        Args:
            audio: Input audio.
            mode: "rms" (target RMS level), "peak" (target peak level),
                  or "headroom" (ensure minimum headroom).

        Returns:
            Tuple of (processed audio, gain applied in dB).
        """
        if mode == "rms":
            current = self.rms_db(audio)
            gain_db = self.target_rms_db - current
            # Safety: don't clip
            peak_after = self.peak_db(audio) + gain_db
            if peak_after > -0.3:
                gain_db = -0.3 - self.peak_db(audio)
        elif mode == "peak":
            current = self.peak_db(audio)
            gain_db = self.target_peak_db - current
        elif mode == "headroom":
            current_peak = self.peak_db(audio)
            if current_peak > self.target_peak_db:
                gain_db = self.target_peak_db - current_peak
            else:
                gain_db = 0.0
        else:
            gain_db = 0.0

        gain_lin = 10 ** (gain_db / 20)
        return audio * gain_lin, round(gain_db, 2)

    def stage_multiple(
        self, tracks: dict[str, np.ndarray], target_rms_db: float | None = None
    ) -> dict[str, tuple[np.ndarray, float]]:
        """Apply gain staging to multiple tracks simultaneously.

        Args:
            tracks: Dict mapping track names to audio arrays.
            target_rms_db: Override target RMS level.

        Returns:
            Dict mapping track names to (processed audio, gain_db) tuples.
        """
        target = target_rms_db or self.target_rms_db
        results = {}

        for name, audio in tracks.items():
            current_rms = self.rms_db(audio)
            gain_db = target - current_rms

            # Safety limit
            peak_after = self.peak_db(audio) + gain_db
            if peak_after > -0.3:
                gain_db = -0.3 - self.peak_db(audio)

            gain_lin = 10 ** (gain_db / 20)
            results[name] = (audio * gain_lin, round(gain_db, 2))

        return results

    def check_summing_headroom(
        self, tracks: list[np.ndarray], target_peak_db: float = -6.0
    ) -> dict:
        """Check if summed tracks will clip and suggest track gain reduction.

        Useful before summing to a bus to ensure headroom.

        Args:
            tracks: List of audio arrays to be summed.
            target_peak_db: Target peak level for the sum.

        Returns:
            Dict with analysis and suggested per-track reduction.
        """
        if not tracks:
            return {"sum_peak_db": -np.inf, "reduction_db": 0.0, "safe": True}

        # Find max length
        max_len = max(len(t) if t.ndim == 1 else t.shape[0] for t in tracks)
        n_ch = max(t.shape[1] if t.ndim > 1 else 1 for t in tracks)

        # Sum signals
        summed = np.zeros((max_len, n_ch))
        for t in tracks:
            sig = t
            if sig.ndim == 1:
                sig = sig.reshape(-1, 1)
                if n_ch == 2:
                    sig = np.column_stack([sig, sig])
            length = sig.shape[0]
            summed[:length] += sig

        sum_peak = self.peak_db(summed)
        overshoot = sum_peak - target_peak_db

        if overshoot > 0:
            # Suggest equal reduction across all tracks
            per_track = -overshoot / len(tracks)
            # Or a uniform reduction on each track
            uniform = -overshoot
        else:
            per_track = 0.0
            uniform = 0.0

        return {
            "sum_peak_db": round(sum_peak, 1),
            "target_peak_db": target_peak_db,
            "overshoot_db": round(max(0, overshoot), 1),
            "reduction_per_track_db": round(per_track, 1),
            "uniform_reduction_db": round(uniform, 1),
            "safe": overshoot <= 0,
            "track_count": len(tracks),
        }

    def report(self, audio: np.ndarray, name: str = "Signal") -> str:
        """Generate a text gain staging report."""
        r = self.analyze(audio)
        lines = [
            f"=== Gain Staging: {name} ===",
            f"  Peak:          {r.peak_db:+.1f} dBFS {'** CLIPPING **' if r.clipping else ''}",
            f"  RMS:           {r.rms_db:+.1f} dBFS",
            f"  Crest Factor:  {r.crest_factor_db:.1f} dB",
            f"  Headroom:      {r.headroom_db:.1f} dB",
            f"  DC Offset:     {r.dc_offset:.6f}",
            f"  Suggested Gain: {r.suggested_gain_db:+.1f} dB",
            f"  Target RMS:    {self.target_rms_db:+.1f} dBFS",
        ]
        return "\n".join(lines)
