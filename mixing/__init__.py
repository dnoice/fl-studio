"""FL Studio Toolkit - Mixing & Effects Chaining

Post-production mixing utilities including channel strips, effects chains,
mix bus processing, gain staging, stereo imaging, loudness analysis,
and reference track comparison.

Quick start::

    from mixing import EffectsChain, GainStaging, MixAnalyzer

    chain = EffectsChain().add_eq(freq=800, gain_db=-3).add_compressor()
    audio, sr = chain.process(audio, sr)
    analysis = MixAnalyzer.analyze(audio, sr)   # -> MixAnalysis
"""

from mixing.channel_strip import ChannelStrip
from mixing.effects_chain import AudioEffect, EffectsChain
from mixing.gain_staging import GainStaging
from mixing.mix_analyzer import MixAnalyzer
from mixing.mix_bus import MixBusProcessor
from mixing.reference_compare import ReferenceCompare
from mixing.stereo_tools import StereoTools

__all__ = [
    "ChannelStrip",
    "AudioEffect",
    "EffectsChain",
    "GainStaging",
    "MixAnalyzer",
    "MixBusProcessor",
    "ReferenceCompare",
    "StereoTools",
]

__version__ = "0.3.0"
