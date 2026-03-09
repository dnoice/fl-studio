"""FL Studio Toolkit - Audio Tools

Audio processing utilities including BPM detection, key detection,
sample slicing, batch processing, spectral analysis, and format conversion.

Quick start::

    from audio_tools import BPMDetector, KeyDetector, SampleSlicer

    bpm = BPMDetector.detect("track.wav")          # -> BPMResult
    key = KeyDetector.detect("track.wav")           # -> KeyResult
    slices = SampleSlicer.slice("loop.wav", "out/") # -> SliceResult
"""

from audio_tools.batch_processor import BatchProcessor, Operation, ProcessingResult
from audio_tools.bpm_detector import BPMDetector, BPMResult
from audio_tools.format_converter import ConversionResult, FormatConverter
from audio_tools.key_detector import KeyDetector, KeyResult
from audio_tools.sample_slicer import SampleSlicer, SliceInfo, SliceResult
from audio_tools.spectrum_analyzer import BandAnalysis, SpectrumAnalyzer, SpectrumData

__all__ = [
    "BatchProcessor",
    "Operation",
    "ProcessingResult",
    "BPMDetector",
    "BPMResult",
    "ConversionResult",
    "FormatConverter",
    "KeyDetector",
    "KeyResult",
    "SampleSlicer",
    "SliceInfo",
    "SliceResult",
    "BandAnalysis",
    "SpectrumAnalyzer",
    "SpectrumData",
]

__version__ = "0.3.0"
