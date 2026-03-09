"""Channel Strip - Complete channel processing in one module.

Integrates gain staging, HPF, EQ, compression, saturation, and panning
into a single configurable channel strip processor, mirroring a
typical FL Studio mixer channel.
"""

from dataclasses import dataclass, field

import numpy as np

from mixing.effects_chain import (
    CompressorEffect,
    EffectsChain,
    EQBand,
    GainEffect,
    HighPassFilter,
    SaturationEffect,
)


@dataclass
class EQConfig:
    """4-band parametric EQ configuration."""

    low_freq: float = 100.0
    low_gain_db: float = 0.0
    low_q: float = 0.7
    low_type: str = "lowshelf"

    low_mid_freq: float = 400.0
    low_mid_gain_db: float = 0.0
    low_mid_q: float = 1.0

    high_mid_freq: float = 2500.0
    high_mid_gain_db: float = 0.0
    high_mid_q: float = 1.0

    high_freq: float = 8000.0
    high_gain_db: float = 0.0
    high_q: float = 0.7
    high_type: str = "highshelf"


@dataclass
class ChannelConfig:
    """Complete channel strip configuration."""

    name: str = "Channel"
    input_gain_db: float = 0.0
    hpf_enabled: bool = True
    hpf_freq: float = 80.0
    eq: EQConfig = field(default_factory=EQConfig)
    eq_enabled: bool = True
    comp_enabled: bool = False
    comp_threshold_db: float = -15.0
    comp_ratio: float = 3.0
    comp_attack_ms: float = 10.0
    comp_release_ms: float = 100.0
    comp_makeup_db: float = 0.0
    saturation_enabled: bool = False
    saturation_drive: float = 1.2
    saturation_mode: str = "tape"
    saturation_mix: float = 0.3
    pan: float = 0.5  # 0.0=left, 0.5=center, 1.0=right
    output_gain_db: float = 0.0
    mute: bool = False
    solo: bool = False
    phase_invert: bool = False


class ChannelStrip:
    """A complete channel strip processor.

    Processes audio through a standard channel strip signal flow:
    Input Gain -> Phase -> HPF -> EQ -> Compressor -> Saturation -> Pan -> Output Gain
    """

    def __init__(self, config: ChannelConfig | None = None):
        self.config = config or ChannelConfig()
        self._chain: EffectsChain = EffectsChain(self.config.name)
        self._rebuild_chain()

    def _rebuild_chain(self):
        """Rebuild the internal effects chain from config."""
        self._chain = EffectsChain(self.config.name)

        # Input gain
        if self.config.input_gain_db != 0.0:
            self._chain.add(GainEffect(self.config.input_gain_db, name="Input Gain"))

        # High-pass filter
        if self.config.hpf_enabled:
            self._chain.add(HighPassFilter(self.config.hpf_freq, name="HPF"))

        # 4-band EQ
        if self.config.eq_enabled:
            eq = self.config.eq
            if eq.low_gain_db != 0.0:
                self._chain.add(
                    EQBand(eq.low_freq, eq.low_gain_db, eq.low_q, eq.low_type, name="EQ Low")
                )
            if eq.low_mid_gain_db != 0.0:
                self._chain.add(
                    EQBand(
                        eq.low_mid_freq,
                        eq.low_mid_gain_db,
                        eq.low_mid_q,
                        "peaking",
                        name="EQ Low-Mid",
                    )
                )
            if eq.high_mid_gain_db != 0.0:
                self._chain.add(
                    EQBand(
                        eq.high_mid_freq,
                        eq.high_mid_gain_db,
                        eq.high_mid_q,
                        "peaking",
                        name="EQ High-Mid",
                    )
                )
            if eq.high_gain_db != 0.0:
                self._chain.add(
                    EQBand(eq.high_freq, eq.high_gain_db, eq.high_q, eq.high_type, name="EQ High")
                )

        # Compressor
        if self.config.comp_enabled:
            self._chain.add(
                CompressorEffect(
                    self.config.comp_threshold_db,
                    self.config.comp_ratio,
                    self.config.comp_attack_ms,
                    self.config.comp_release_ms,
                    self.config.comp_makeup_db,
                    name="Compressor",
                )
            )

        # Saturation
        if self.config.saturation_enabled:
            sat = SaturationEffect(
                self.config.saturation_drive,
                self.config.saturation_mode,
                name="Saturation",
            )
            sat.dry_wet = self.config.saturation_mix
            self._chain.add(sat)

        # Output gain
        if self.config.output_gain_db != 0.0:
            self._chain.add(GainEffect(self.config.output_gain_db, name="Output Gain"))

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process audio through the channel strip.

        Args:
            audio: Input audio (1D mono or 2D stereo)
            sr: Sample rate

        Returns:
            Processed audio (always 2D stereo)
        """
        if self.config.mute:
            if audio.ndim == 1:
                return np.zeros((len(audio), 2), dtype=audio.dtype)
            return np.zeros_like(audio)

        result = np.copy(audio)

        # Phase invert
        if self.config.phase_invert:
            result = -result

        # Process chain
        result = self._chain.process(result, sr)

        # Convert to stereo if needed for panning
        if result.ndim == 1:
            result = np.column_stack([result, result])

        # Apply panning (constant-power pan law)
        pan = self.config.pan
        left_gain = np.cos(pan * np.pi / 2)
        right_gain = np.sin(pan * np.pi / 2)
        result[:, 0] *= left_gain
        result[:, 1] *= right_gain

        return result

    def update_config(self, **kwargs):
        """Update configuration and rebuild chain."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self._rebuild_chain()

    def summary(self) -> str:
        """Text summary of channel strip settings."""
        c = self.config
        lines = [
            f"=== Channel: {c.name} ===",
            f"  Input Gain: {c.input_gain_db:+.1f} dB",
            f"  HPF: {'ON' if c.hpf_enabled else 'OFF'} @ {c.hpf_freq:.0f} Hz",
            f"  EQ: {'ON' if c.eq_enabled else 'OFF'}",
        ]
        if c.eq_enabled:
            eq = c.eq
            lines.append(
                f"    Low:      {eq.low_freq:.0f}Hz {eq.low_gain_db:+.1f}dB Q:{eq.low_q:.1f}"
            )
            lines.append(
                f"    Low-Mid:  {eq.low_mid_freq:.0f}Hz {eq.low_mid_gain_db:+.1f}dB Q:{eq.low_mid_q:.1f}"
            )
            lines.append(
                f"    High-Mid: {eq.high_mid_freq:.0f}Hz {eq.high_mid_gain_db:+.1f}dB Q:{eq.high_mid_q:.1f}"
            )
            lines.append(
                f"    High:     {eq.high_freq:.0f}Hz {eq.high_gain_db:+.1f}dB Q:{eq.high_q:.1f}"
            )

        lines.extend(
            [
                f"  Comp: {'ON' if c.comp_enabled else 'OFF'}"
                + (
                    f" (thr:{c.comp_threshold_db:.0f}dB ratio:{c.comp_ratio:.1f}:1)"
                    if c.comp_enabled
                    else ""
                ),
                f"  Saturation: {'ON' if c.saturation_enabled else 'OFF'}"
                + (
                    f" ({c.saturation_mode} drive:{c.saturation_drive:.1f})"
                    if c.saturation_enabled
                    else ""
                ),
                f"  Pan: {'L' if c.pan < 0.4 else 'C' if c.pan < 0.6 else 'R'} ({c.pan:.2f})",
                f"  Output: {c.output_gain_db:+.1f} dB",
                f"  Mute: {c.mute} | Phase: {'INV' if c.phase_invert else 'Normal'}",
            ]
        )
        return "\n".join(lines)

    # ─── Channel Presets ───

    @classmethod
    def vocal(cls, name: str = "Vocals") -> "ChannelStrip":
        config = ChannelConfig(
            name=name,
            hpf_freq=100.0,
            eq=EQConfig(
                low_gain_db=-2.0, high_mid_freq=3000.0, high_mid_gain_db=2.5, high_gain_db=1.5
            ),
            comp_enabled=True,
            comp_threshold_db=-18.0,
            comp_ratio=3.0,
            comp_attack_ms=8.0,
            comp_release_ms=80.0,
            comp_makeup_db=4.0,
            saturation_enabled=True,
            saturation_drive=1.2,
            saturation_mix=0.2,
        )
        return cls(config)

    @classmethod
    def kick(cls, name: str = "Kick") -> "ChannelStrip":
        config = ChannelConfig(
            name=name,
            hpf_freq=30.0,
            eq=EQConfig(
                low_freq=60.0,
                low_gain_db=3.0,
                low_type="peaking",
                low_mid_freq=300.0,
                low_mid_gain_db=-3.0,
                high_mid_freq=4000.0,
                high_mid_gain_db=2.0,
            ),
            comp_enabled=True,
            comp_threshold_db=-10.0,
            comp_ratio=4.0,
            comp_attack_ms=5.0,
            comp_release_ms=50.0,
            comp_makeup_db=2.0,
        )
        return cls(config)

    @classmethod
    def snare(cls, name: str = "Snare") -> "ChannelStrip":
        config = ChannelConfig(
            name=name,
            hpf_freq=80.0,
            eq=EQConfig(
                low_freq=200.0,
                low_gain_db=2.0,
                low_type="peaking",
                high_mid_freq=5000.0,
                high_mid_gain_db=3.0,
            ),
            comp_enabled=True,
            comp_threshold_db=-12.0,
            comp_ratio=3.5,
            comp_attack_ms=3.0,
            comp_release_ms=60.0,
            comp_makeup_db=3.0,
            saturation_enabled=True,
            saturation_drive=1.5,
            saturation_mix=0.15,
        )
        return cls(config)

    @classmethod
    def bass(cls, name: str = "Bass") -> "ChannelStrip":
        config = ChannelConfig(
            name=name,
            hpf_freq=30.0,
            eq=EQConfig(
                low_freq=80.0,
                low_gain_db=2.0,
                low_type="peaking",
                low_mid_freq=250.0,
                low_mid_gain_db=-2.0,
            ),
            comp_enabled=True,
            comp_threshold_db=-12.0,
            comp_ratio=4.0,
            comp_attack_ms=5.0,
            comp_release_ms=60.0,
            comp_makeup_db=3.0,
            saturation_enabled=True,
            saturation_drive=1.8,
            saturation_mode="tube",
            saturation_mix=0.3,
        )
        return cls(config)

    @classmethod
    def pad(cls, name: str = "Pad") -> "ChannelStrip":
        config = ChannelConfig(
            name=name,
            hpf_freq=120.0,
            eq=EQConfig(
                low_mid_freq=500.0, low_mid_gain_db=-2.0, high_freq=10000.0, high_gain_db=-1.5
            ),
        )
        return cls(config)
