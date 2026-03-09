"""Mix Bus Processor - Summing and bus processing for groups of channels.

Provides sub-mix buses (drums, vocals, instruments, FX returns) and a
master bus with configurable insert chains, send/return routing, and
metering.  Mirrors the FL Studio mixer routing workflow.
"""

from dataclasses import dataclass

import numpy as np

from mixing.channel_strip import ChannelStrip
from mixing.effects_chain import (
    EffectsChain,
)


@dataclass
class SendConfig:
    """Send routing from a channel to a bus."""

    target_bus: str
    gain_db: float = 0.0
    pre_fader: bool = False


class MixBus:
    """A single mix bus that sums multiple inputs and applies processing.

    Can serve as a sub-group (drums, vocals) or as the master bus.
    """

    def __init__(self, name: str = "Bus", chain: EffectsChain | None = None):
        self.name = name
        self.chain = chain or EffectsChain(name)
        self.gain_db: float = 0.0
        self.mute: bool = False
        self.solo: bool = False
        self._buffer: np.ndarray | None = None
        self._input_count: int = 0

    def reset(self, length: int, channels: int = 2, dtype=np.float64):
        """Clear the bus buffer for a new processing pass."""
        self._buffer = np.zeros((length, channels), dtype=dtype)
        self._input_count = 0

    def add_signal(self, audio: np.ndarray, gain_db: float = 0.0):
        """Sum a signal into this bus.

        Args:
            audio: Input audio (1D or 2D). Converted to stereo if needed.
            gain_db: Additional send gain in dB.
        """
        if self._buffer is None:
            if audio.ndim == 1:
                self.reset(len(audio))
            else:
                self.reset(audio.shape[0], audio.shape[1])

        signal = np.copy(audio)

        # Ensure stereo
        if signal.ndim == 1:
            signal = np.column_stack([signal, signal])

        # Match buffer length (truncate or pad)
        assert self._buffer is not None, "Call reset() before add_signal()"
        buf_len = self._buffer.shape[0]
        if len(signal) > buf_len:
            signal = signal[:buf_len]
        elif len(signal) < buf_len:
            pad = np.zeros((buf_len - len(signal), signal.shape[1]), dtype=signal.dtype)
            signal = np.vstack([signal, pad])

        # Apply send gain
        if gain_db != 0.0:
            signal *= 10 ** (gain_db / 20)

        self._buffer += signal
        self._input_count += 1

    def process(self, sr: int) -> np.ndarray:
        """Process the summed bus through its effects chain.

        Returns:
            Processed stereo audio.
        """
        if self._buffer is None:
            return np.zeros((0, 2))

        if self.mute:
            return np.zeros_like(self._buffer)

        result = self._buffer

        # Bus gain
        if self.gain_db != 0.0:
            result = result * (10 ** (self.gain_db / 20))

        # Effects chain
        if self.chain and len(self.chain.effects) > 0:
            result = self.chain.process(result, sr)

        return result

    @property
    def peak_db(self) -> float:
        """Peak level of the current buffer in dB."""
        if self._buffer is None:
            return -np.inf
        peak = float(np.max(np.abs(self._buffer)))
        return 20 * np.log10(max(peak, 1e-10))

    @property
    def input_count(self) -> int:
        return self._input_count

    def summary(self) -> str:
        lines = [f"--- Bus: {self.name} ---"]
        lines.append(f"  Gain: {self.gain_db:+.1f} dB | Mute: {self.mute}")
        lines.append(f"  Inputs: {self._input_count}")
        if self.chain:
            for i, fx in enumerate(self.chain.effects):
                status = "[BYPASS]" if fx.bypassed else "[ON]"
                lines.append(f"  {i+1}. {status} {fx.name}")
        return "\n".join(lines)


class MixBusProcessor:
    """Complete mix bus routing system.

    Manages sub-group buses, send/return buses, and a master bus.
    Processes channels through their assigned routes and sums to the
    master output.

    Typical usage:
        proc = MixBusProcessor()
        proc.add_bus("Drums", chain=EffectsChain.drum_bus_chain())
        proc.add_bus("Vocals", chain=EffectsChain.vocal_chain())
        proc.add_bus("FX Return")

        proc.route("Drums", kick_audio, sr)
        proc.route("Drums", snare_audio, sr)
        proc.route("Vocals", lead_vocal, sr)

        # Send from vocals to FX return
        proc.send("Vocals", "FX Return", gain_db=-6.0)

        master = proc.mixdown(sr)
    """

    def __init__(self):
        self._buses: dict[str, MixBus] = {}
        self._master = MixBus("Master")
        self._sends: list[tuple[str, str, float]] = []  # (from_bus, to_bus, gain_db)
        self._channel_audio: dict[str, list[tuple[np.ndarray, float]]] = {}
        self._sr: int = 44100

    @property
    def master(self) -> MixBus:
        return self._master

    def add_bus(self, name: str, chain: EffectsChain | None = None, gain_db: float = 0.0) -> MixBus:
        """Add a sub-group or return bus."""
        bus = MixBus(name, chain)
        bus.gain_db = gain_db
        self._buses[name] = bus
        return bus

    def get_bus(self, name: str) -> MixBus | None:
        if name == "Master":
            return self._master
        return self._buses.get(name)

    def remove_bus(self, name: str) -> None:
        self._buses.pop(name, None)
        self._sends = [(f, t, g) for f, t, g in self._sends if f != name and t != name]

    def set_master_chain(self, chain: EffectsChain) -> None:
        """Set the master bus effects chain."""
        self._master.chain = chain

    def route(self, bus_name: str, audio: np.ndarray, sr: int, gain_db: float = 0.0):
        """Route audio to a specific bus.

        Args:
            bus_name: Target bus name.
            audio: Input audio signal.
            sr: Sample rate.
            gain_db: Routing gain adjustment.
        """
        self._sr = sr
        bus = self._buses.get(bus_name)
        if bus is None:
            bus = self.add_bus(bus_name)
        bus.add_signal(audio, gain_db)

    def route_channel(self, bus_name: str, channel: ChannelStrip, audio: np.ndarray, sr: int):
        """Route audio through a channel strip into a bus."""
        processed = channel.process(audio, sr)
        self.route(bus_name, processed, sr)

    def send(self, from_bus: str, to_bus: str, gain_db: float = 0.0):
        """Create a send from one bus to another.

        The send taps the output of from_bus and feeds it into to_bus.
        """
        self._sends.append((from_bus, to_bus, gain_db))

    def clear_sends(self):
        self._sends.clear()

    def mixdown(self, sr: int | None = None) -> np.ndarray:
        """Process all buses and sum to master.

        Returns:
            Final stereo master output.
        """
        sr = sr or self._sr

        # Check for solo - if any bus is soloed, mute non-soloed buses
        any_solo = any(b.solo for b in self._buses.values())

        bus_outputs: dict[str, np.ndarray] = {}

        # Process sub-group buses
        for name, bus in self._buses.items():
            if any_solo and not bus.solo:
                bus_outputs[name] = (
                    np.zeros_like(bus._buffer) if bus._buffer is not None else np.zeros((0, 2))
                )
            else:
                bus_outputs[name] = bus.process(sr)

        # Process sends (tap bus outputs and feed to target buses)
        for from_name, to_name, gain_db in self._sends:
            if from_name in bus_outputs and to_name in self._buses:
                target = self._buses[to_name]
                target.add_signal(bus_outputs[from_name], gain_db)
                # Re-process target bus with new input
                if any_solo and not target.solo:
                    bus_outputs[to_name] = (
                        np.zeros_like(target._buffer)
                        if target._buffer is not None
                        else np.zeros((0, 2))
                    )
                else:
                    bus_outputs[to_name] = target.process(sr)

        # Sum all bus outputs to master
        max_len = max((out.shape[0] for out in bus_outputs.values() if out.shape[0] > 0), default=0)
        if max_len == 0:
            return np.zeros((0, 2))

        self._master.reset(max_len)
        for output in bus_outputs.values():
            if output.shape[0] > 0:
                self._master.add_signal(output)

        return self._master.process(sr)

    def reset_all(self):
        """Reset all bus buffers for a new pass."""
        for bus in self._buses.values():
            bus._buffer = None
            bus._input_count = 0
        self._master._buffer = None
        self._master._input_count = 0

    def summary(self) -> str:
        lines = ["=== Mix Bus Processor ==="]
        for _name, bus in self._buses.items():
            lines.append(bus.summary())
        if self._sends:
            lines.append("--- Sends ---")
            for f, t, g in self._sends:
                lines.append(f"  {f} -> {t} ({g:+.1f} dB)")
        lines.append(self._master.summary())
        return "\n".join(lines)

    # ─── Preset Configurations ───

    @classmethod
    def standard_mix(cls) -> "MixBusProcessor":
        """Standard mix bus layout: Drums, Bass, Instruments, Vocals, FX."""
        proc = cls()
        proc.add_bus("Drums", chain=EffectsChain.drum_bus_chain())
        proc.add_bus("Bass", chain=EffectsChain.bass_chain())
        proc.add_bus("Instruments")
        proc.add_bus("Vocals", chain=EffectsChain.vocal_chain())
        proc.add_bus("FX Return", gain_db=-6.0)
        proc.set_master_chain(EffectsChain.master_chain())
        return proc

    @classmethod
    def stem_mix(cls) -> "MixBusProcessor":
        """Minimal stem-based layout for simple projects."""
        proc = cls()
        proc.add_bus("Music")
        proc.add_bus("Vocals")
        proc.set_master_chain(EffectsChain.master_chain())
        return proc
