"""Effects Chain - Composable audio effects pipeline with routing.

Build processing chains by stacking effects in series or parallel,
with per-effect bypass, dry/wet mixing, and gain staging between stages.
"""

from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf


class AudioEffect(ABC):
    """Base class for all audio effects in the chain."""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self.bypassed = False
        self.dry_wet = 1.0  # 0.0 = fully dry, 1.0 = fully wet
        self.input_gain_db = 0.0
        self.output_gain_db = 0.0

    @abstractmethod
    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process audio. Must be implemented by subclasses."""
        pass

    def _apply_gains_and_mix(self, dry: np.ndarray, wet: np.ndarray) -> np.ndarray:
        """Apply input/output gains and dry/wet mixing."""
        if self.dry_wet >= 1.0:
            result = wet
        elif self.dry_wet <= 0.0:
            return dry
        else:
            result = dry * (1.0 - self.dry_wet) + wet * self.dry_wet

        if self.output_gain_db != 0.0:
            result = result * (10 ** (self.output_gain_db / 20))
        return result


# ─── Built-in Effects ───


class GainEffect(AudioEffect):
    """Simple gain adjustment."""

    def __init__(self, gain_db: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.gain_db = gain_db

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        return audio * (10 ** (self.gain_db / 20))


class EQBand(AudioEffect):
    """Parametric EQ band using biquad filter."""

    def __init__(
        self,
        freq: float = 1000.0,
        gain_db: float = 0.0,
        q: float = 1.0,
        filter_type: str = "peaking",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.freq = max(1.0, freq)
        self.gain_db = gain_db
        self.q = max(0.01, q)
        self.filter_type = filter_type  # peaking, lowshelf, highshelf, lowpass, highpass

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from mixing._biquad import biquad_filter

        return biquad_filter(audio, sr, self.filter_type, self.freq, self.q, self.gain_db)


class CompressorEffect(AudioEffect):
    """Dynamic range compressor."""

    def __init__(
        self,
        threshold_db: float = -10.0,
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        makeup_db: float = 0.0,
        knee_db: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.makeup_db = makeup_db
        self.knee_db = knee_db
        self._gain_reduction_db = 0.0

    @property
    def gain_reduction_db(self) -> float:
        return self._gain_reduction_db

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        attack_coeff = np.exp(-1.0 / (self.attack_ms * 0.001 * sr))
        release_coeff = np.exp(-1.0 / (self.release_ms * 0.001 * sr))
        10 ** (self.threshold_db / 20)
        makeup_lin = 10 ** (self.makeup_db / 20)
        half_knee = self.knee_db / 2

        envelope = np.abs(audio) if audio.ndim == 1 else np.max(np.abs(audio), axis=1)

        output = np.copy(audio)
        env = 0.0
        max_gr = 0.0

        for i in range(len(envelope)):
            level = envelope[i]
            coeff = attack_coeff if level > env else release_coeff
            env = env * coeff + level * (1.0 - coeff)

            env_db = 20 * np.log10(max(env, 1e-10))

            # Soft knee
            if self.knee_db > 0 and abs(env_db - self.threshold_db) < half_knee:
                over = env_db - self.threshold_db + half_knee
                gain_reduction = -(over**2) / (2 * self.knee_db) * (1 - 1 / self.ratio)
            elif env_db > self.threshold_db:
                over = env_db - self.threshold_db
                gain_reduction = -over * (1 - 1 / self.ratio)
            else:
                gain_reduction = 0.0

            gain_lin = 10 ** (gain_reduction / 20) * makeup_lin
            max_gr = min(max_gr, gain_reduction)

            if audio.ndim == 1:
                output[i] *= gain_lin
            else:
                output[i, :] *= gain_lin

        self._gain_reduction_db = max_gr
        return output


class LimiterEffect(AudioEffect):
    """Brick-wall limiter."""

    def __init__(self, ceiling_db: float = -0.3, release_ms: float = 50.0, **kwargs):
        super().__init__(**kwargs)
        self.ceiling_db = ceiling_db
        self.release_ms = release_ms

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        ceiling = 10 ** (self.ceiling_db / 20)
        release_coeff = np.exp(-1.0 / (self.release_ms * 0.001 * sr))

        output = np.copy(audio)
        env = 0.0

        for i in range(len(audio)):
            level = abs(audio[i]) if audio.ndim == 1 else np.max(np.abs(audio[i]))

            env = level if level > env else env * release_coeff + level * (1 - release_coeff)

            gain = ceiling / env if env > ceiling else 1.0

            if audio.ndim == 1:
                output[i] *= gain
            else:
                output[i, :] *= gain

        return output


class SaturationEffect(AudioEffect):
    """Harmonic saturation/warmth."""

    def __init__(self, drive: float = 1.0, mode: str = "tape", **kwargs):
        super().__init__(**kwargs)
        self.drive = drive
        self.mode = mode  # tape, tube, digital

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        driven = audio * self.drive

        if self.mode == "tape":
            wet = np.tanh(driven * 1.5) / 1.5
        elif self.mode == "tube":
            pos = 1.0 - np.exp(-np.abs(driven))
            wet = np.sign(driven) * pos
            # Add even harmonics
            wet += 0.05 * driven * driven * np.sign(driven)
            wet = np.clip(wet, -1.0, 1.0)
        elif self.mode == "digital":
            wet = np.clip(driven, -1.0, 1.0)
        else:
            wet = np.tanh(driven)

        # Compensate volume
        wet *= 1.0 / max(1.0, self.drive * 0.7)
        return self._apply_gains_and_mix(audio, wet)


class HighPassFilter(AudioEffect):
    """High-pass filter for removing low-end rumble."""

    def __init__(self, freq: float = 80.0, order: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.freq = freq
        self.order = order

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from mixing._biquad import biquad_filter

        result = audio
        for _ in range(self.order // 2):
            result = biquad_filter(result, sr, "highpass", self.freq, 0.707, 0.0)
        return result


class LowPassFilter(AudioEffect):
    """Low-pass filter."""

    def __init__(self, freq: float = 16000.0, order: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.freq = freq
        self.order = order

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from mixing._biquad import biquad_filter

        result = audio
        for _ in range(self.order // 2):
            result = biquad_filter(result, sr, "lowpass", self.freq, 0.707, 0.0)
        return result


class DeEsser(AudioEffect):
    """Simple de-esser targeting sibilant frequencies."""

    def __init__(
        self,
        freq: float = 6000.0,
        threshold_db: float = -20.0,
        reduction_db: float = -6.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.freq = freq
        self.threshold_db = threshold_db
        self.reduction_db = reduction_db

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from mixing._biquad import biquad_filter

        # Extract sibilant band
        sidechain = biquad_filter(audio, sr, "bandpass", self.freq, 2.0, 0.0)

        threshold = 10 ** (self.threshold_db / 20)
        reduction = 10 ** (self.reduction_db / 20)

        output = np.copy(audio)
        env = 0.0
        coeff = np.exp(-1.0 / (0.001 * sr))

        for i in range(len(audio)):
            level = abs(sidechain[i]) if sidechain.ndim == 1 else np.max(np.abs(sidechain[i]))

            env = max(level, env * coeff)

            if env > threshold:
                gain = reduction + (1.0 - reduction) * (threshold / max(env, 1e-10))
            else:
                gain = 1.0

            if audio.ndim == 1:
                output[i] *= gain
            else:
                output[i, :] *= gain

        return output


# ─── Effects Chain ───


class EffectsChain:
    """Composable effects processing chain.

    Stack effects in series, with per-effect bypass and dry/wet control.
    Supports parallel processing buses and send/return routing.
    """

    def __init__(self, name: str = "Chain"):
        self.name = name
        self._effects: list[AudioEffect] = []
        self._parallel_chains: list[EffectsChain] = []
        self._parallel_gains: list[float] = []

    def add(self, effect: AudioEffect) -> "EffectsChain":
        """Add an effect to the end of the chain."""
        self._effects.append(effect)
        return self

    def insert(self, index: int, effect: AudioEffect) -> "EffectsChain":
        """Insert an effect at a specific position."""
        self._effects.insert(index, effect)
        return self

    def remove(self, index: int) -> AudioEffect:
        """Remove and return an effect by index."""
        return self._effects.pop(index)

    def add_parallel(self, chain: "EffectsChain", gain_db: float = 0.0) -> "EffectsChain":
        """Add a parallel processing chain (summed with main output)."""
        self._parallel_chains.append(chain)
        self._parallel_gains.append(10 ** (gain_db / 20))
        return self

    def bypass_all(self) -> None:
        """Bypass all effects."""
        for fx in self._effects:
            fx.bypassed = True

    def enable_all(self) -> None:
        """Enable all effects."""
        for fx in self._effects:
            fx.bypassed = False

    @property
    def effects(self) -> list[AudioEffect]:
        return list(self._effects)

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process audio through the entire chain.

        Args:
            audio: Input audio (1D mono or 2D multi-channel)
            sr: Sample rate

        Returns:
            Processed audio
        """
        result = np.copy(audio)

        # Main series chain
        for fx in self._effects:
            if fx.bypassed:
                continue

            if fx.input_gain_db != 0.0:
                result = result * (10 ** (fx.input_gain_db / 20))

            dry = np.copy(result)
            wet = fx.process(result, sr)

            result = dry * (1.0 - fx.dry_wet) + wet * fx.dry_wet if fx.dry_wet < 1.0 else wet

            if fx.output_gain_db != 0.0:
                result = result * (10 ** (fx.output_gain_db / 20))

        # Add parallel chains
        for chain, gain in zip(self._parallel_chains, self._parallel_gains, strict=False):
            parallel_out = chain.process(audio, sr)
            result = result + parallel_out * gain

        return result

    def process_file(self, input_path: str, output_path: str) -> dict:
        """Process an audio file through the chain."""
        try:
            audio, sr = sf.read(input_path, dtype="float32")
        except Exception as e:
            raise OSError(f"Failed to read audio file '{input_path}': {e}") from e

        peak_before = float(np.max(np.abs(audio)))
        processed = self.process(audio, sr)
        peak_after = float(np.max(np.abs(processed)))

        sf.write(output_path, processed, sr)

        return {
            "input": input_path,
            "output": output_path,
            "sample_rate": sr,
            "peak_before_db": 20 * np.log10(max(peak_before, 1e-10)),
            "peak_after_db": 20 * np.log10(max(peak_after, 1e-10)),
            "effects": [fx.name for fx in self._effects if not fx.bypassed],
        }

    def summary(self) -> str:
        """Text summary of the chain."""
        lines = [f"=== Effects Chain: {self.name} ==="]
        for i, fx in enumerate(self._effects):
            status = "[BYPASS]" if fx.bypassed else "[ON]"
            wet = f"wet:{fx.dry_wet:.0%}" if fx.dry_wet < 1.0 else ""
            lines.append(f"  {i+1}. {status} {fx.name} {wet}")
        if self._parallel_chains:
            lines.append("  --- Parallel Buses ---")
            for i, chain in enumerate(self._parallel_chains):
                gain_db = 20 * np.log10(max(self._parallel_gains[i], 1e-10))
                lines.append(f"  P{i+1}. {chain.name} ({gain_db:+.1f}dB)")
        return "\n".join(lines)

    # ─── Preset Chains ───

    @classmethod
    def vocal_chain(cls) -> "EffectsChain":
        """Standard vocal processing chain."""
        chain = cls("Vocal Chain")
        chain.add(HighPassFilter(freq=80.0, name="HP Filter"))
        chain.add(DeEsser(freq=6500.0, threshold_db=-25.0, name="De-Esser"))
        chain.add(
            CompressorEffect(
                threshold_db=-18.0,
                ratio=3.0,
                attack_ms=10.0,
                release_ms=80.0,
                makeup_db=4.0,
                name="Compressor",
            )
        )
        chain.add(EQBand(freq=3000.0, gain_db=2.0, q=1.5, name="Presence EQ"))
        chain.add(SaturationEffect(drive=1.2, mode="tape", name="Tape Warmth"))
        chain._effects[-1].dry_wet = 0.3
        return chain

    @classmethod
    def drum_bus_chain(cls) -> "EffectsChain":
        """Drum bus glue processing."""
        chain = cls("Drum Bus")
        chain.add(HighPassFilter(freq=30.0, name="Sub Cleanup"))
        chain.add(
            CompressorEffect(
                threshold_db=-12.0,
                ratio=4.0,
                attack_ms=20.0,
                release_ms=150.0,
                makeup_db=3.0,
                knee_db=6.0,
                name="Bus Comp",
            )
        )
        chain.add(SaturationEffect(drive=1.5, mode="tape", name="Tape Glue"))
        chain._effects[-1].dry_wet = 0.2
        chain.add(EQBand(freq=100.0, gain_db=2.0, q=0.8, filter_type="lowshelf", name="Low Shelf"))
        return chain

    @classmethod
    def master_chain(cls) -> "EffectsChain":
        """Mastering-style chain."""
        chain = cls("Master Chain")
        chain.add(HighPassFilter(freq=25.0, name="Sub Rumble Cut"))
        chain.add(EQBand(freq=200.0, gain_db=-1.0, q=1.0, filter_type="peaking", name="Mud Cut"))
        chain.add(EQBand(freq=3500.0, gain_db=1.5, q=1.2, filter_type="peaking", name="Presence"))
        chain.add(
            CompressorEffect(
                threshold_db=-8.0,
                ratio=2.0,
                attack_ms=30.0,
                release_ms=200.0,
                makeup_db=2.0,
                knee_db=8.0,
                name="Glue Comp",
            )
        )
        chain.add(LimiterEffect(ceiling_db=-0.3, release_ms=50.0, name="Limiter"))
        return chain

    @classmethod
    def bass_chain(cls) -> "EffectsChain":
        """Bass instrument processing."""
        chain = cls("Bass Chain")
        chain.add(HighPassFilter(freq=30.0, name="Sub Clean"))
        chain.add(
            CompressorEffect(
                threshold_db=-15.0,
                ratio=4.0,
                attack_ms=5.0,
                release_ms=60.0,
                makeup_db=3.0,
                name="Fast Comp",
            )
        )
        chain.add(SaturationEffect(drive=1.8, mode="tube", name="Tube Harmonics"))
        chain._effects[-1].dry_wet = 0.4
        chain.add(LowPassFilter(freq=8000.0, name="Top Cut"))
        return chain
