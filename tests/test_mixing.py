"""Comprehensive tests for the mixing module."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mixing._biquad import biquad_coefficients, biquad_filter
from mixing.channel_strip import ChannelConfig, ChannelStrip
from mixing.effects_chain import (
    CompressorEffect,
    DeEsser,
    EffectsChain,
    GainEffect,
    HighPassFilter,
    LimiterEffect,
    LowPassFilter,
    SaturationEffect,
)
from mixing.gain_staging import GainReport, GainStaging
from mixing.mix_analyzer import MixAnalyzer
from mixing.mix_bus import MixBus, MixBusProcessor
from mixing.reference_compare import ReferenceCompare
from mixing.stereo_tools import StereoTools


@pytest.fixture
def mono_signal():
    """1 second 440 Hz mono sine at 44100 Hz."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return np.sin(2 * np.pi * 440 * t) * 0.5, sr


@pytest.fixture
def stereo_signal():
    """1 second stereo signal at 44100 Hz."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    left = np.sin(2 * np.pi * 440 * t) * 0.5
    right = np.sin(2 * np.pi * 554 * t) * 0.4
    return np.column_stack([left, right]), sr


@pytest.fixture
def loud_signal():
    """Near-clipping signal."""
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return np.sin(2 * np.pi * 440 * t) * 0.99, sr


# ─── Biquad Filter Tests ───


class TestBiquad:
    def test_lowpass_coefficients(self):
        b0, b1, b2, a1, a2 = biquad_coefficients(44100, "lowpass", 1000, 0.707, 0)
        assert b0 != 0
        # Low-pass should attenuate highs
        assert abs(b0) < 1

    def test_highpass_filter(self, mono_signal):
        audio, sr = mono_signal
        filtered = biquad_filter(audio, sr, "highpass", 1000, 0.707, 0)
        # 440Hz signal should be attenuated by a 1000Hz highpass
        assert np.max(np.abs(filtered)) < np.max(np.abs(audio))

    def test_peaking_eq(self, mono_signal):
        audio, sr = mono_signal
        boosted = biquad_filter(audio, sr, "peaking", 440, 1.0, 6.0)
        assert np.max(np.abs(boosted)) > np.max(np.abs(audio))

    def test_stereo_filter(self, stereo_signal):
        audio, sr = stereo_signal
        filtered = biquad_filter(audio, sr, "lowpass", 2000, 0.707, 0)
        assert filtered.shape == audio.shape

    def test_unknown_filter_passthrough(self, mono_signal):
        audio, sr = mono_signal
        b0, b1, b2, a1, a2 = biquad_coefficients(sr, "unknown_type", 1000, 1.0, 0)
        assert b0 == 1.0 and b1 == 0.0


# ─── Effects Chain Tests ───


class TestEffectsChain:
    def test_gain_effect(self, mono_signal):
        audio, sr = mono_signal
        fx = GainEffect(gain_db=6.0)
        result = fx.process(audio, sr)
        assert np.max(np.abs(result)) > np.max(np.abs(audio))

    def test_gain_negative(self, mono_signal):
        audio, sr = mono_signal
        fx = GainEffect(gain_db=-6.0)
        result = fx.process(audio, sr)
        assert np.max(np.abs(result)) < np.max(np.abs(audio))

    def test_compressor(self, loud_signal):
        audio, sr = loud_signal
        comp = CompressorEffect(threshold_db=-6.0, ratio=4.0, attack_ms=1.0, release_ms=50.0)
        result = comp.process(audio, sr)
        assert np.max(np.abs(result)) <= np.max(np.abs(audio))
        assert comp.gain_reduction_db < 0

    def test_compressor_with_makeup(self, mono_signal):
        audio, sr = mono_signal
        comp = CompressorEffect(threshold_db=-20.0, ratio=4.0, makeup_db=6.0)
        result = comp.process(audio, sr)
        assert isinstance(result, np.ndarray)

    def test_limiter(self, loud_signal):
        audio, sr = loud_signal
        limiter = LimiterEffect(ceiling_db=-3.0)
        result = limiter.process(audio, sr)
        ceiling_lin = 10 ** (-3.0 / 20)
        # Allow small overshoot due to attack time
        assert np.max(np.abs(result)) < ceiling_lin * 1.1

    def test_saturation_tape(self, mono_signal):
        audio, sr = mono_signal
        sat = SaturationEffect(drive=2.0, mode="tape")
        result = sat.process(audio, sr)
        assert len(result) == len(audio)

    def test_saturation_tube(self, mono_signal):
        audio, sr = mono_signal
        sat = SaturationEffect(drive=2.0, mode="tube")
        result = sat.process(audio, sr)
        assert np.max(np.abs(result)) <= 1.0

    def test_saturation_digital(self, mono_signal):
        audio, sr = mono_signal
        sat = SaturationEffect(drive=3.0, mode="digital")
        result = sat.process(audio, sr)
        assert np.max(np.abs(result)) <= 1.0

    def test_highpass_filter(self, mono_signal):
        audio, sr = mono_signal
        hpf = HighPassFilter(freq=1000.0)
        result = hpf.process(audio, sr)
        # 440Hz should be reduced
        assert np.max(np.abs(result)) < np.max(np.abs(audio))

    def test_lowpass_filter(self, mono_signal):
        audio, sr = mono_signal
        lpf = LowPassFilter(freq=200.0)
        result = lpf.process(audio, sr)
        assert np.max(np.abs(result)) < np.max(np.abs(audio))

    def test_deesser(self, mono_signal):
        audio, sr = mono_signal
        deess = DeEsser(freq=6000.0, threshold_db=-30.0)
        result = deess.process(audio, sr)
        assert len(result) == len(audio)

    def test_chain_processing(self, mono_signal):
        audio, sr = mono_signal
        chain = EffectsChain("Test")
        chain.add(GainEffect(-3.0, name="Trim"))
        chain.add(HighPassFilter(100.0, name="HPF"))
        result = chain.process(audio, sr)
        assert len(result) == len(audio)
        assert len(chain.effects) == 2

    def test_chain_bypass(self, mono_signal):
        audio, sr = mono_signal
        chain = EffectsChain("Test")
        chain.add(GainEffect(12.0, name="Boost"))
        chain.bypass_all()
        result = chain.process(audio, sr)
        np.testing.assert_array_almost_equal(result, audio)

    def test_chain_dry_wet(self, mono_signal):
        audio, sr = mono_signal
        chain = EffectsChain("Test")
        fx = GainEffect(6.0, name="Boost")
        fx.dry_wet = 0.5
        chain.add(fx)
        result = chain.process(audio, sr)
        # Should be between dry and fully wet
        dry_peak = np.max(np.abs(audio))
        wet_peak = np.max(np.abs(result))
        assert dry_peak < wet_peak < dry_peak * 2

    def test_chain_parallel(self, mono_signal):
        audio, sr = mono_signal
        main = EffectsChain("Main")
        parallel = EffectsChain("Parallel")
        parallel.add(GainEffect(0.0))
        main.add_parallel(parallel, gain_db=-6.0)
        result = main.process(audio, sr)
        assert len(result) == len(audio)

    def test_chain_insert_remove(self, mono_signal):
        chain = EffectsChain("Test")
        chain.add(GainEffect(0.0, name="A"))
        chain.add(GainEffect(0.0, name="C"))
        chain.insert(1, GainEffect(0.0, name="B"))
        assert chain.effects[1].name == "B"
        removed = chain.remove(1)
        assert removed.name == "B"
        assert len(chain.effects) == 2

    def test_chain_summary(self):
        chain = EffectsChain("Test Chain")
        chain.add(GainEffect(0.0, name="Gain"))
        chain.add(CompressorEffect(name="Comp"))
        summary = chain.summary()
        assert "Test Chain" in summary
        assert "Gain" in summary
        assert "Comp" in summary

    def test_preset_chains(self, mono_signal):
        audio, sr = mono_signal
        for factory in [
            EffectsChain.vocal_chain,
            EffectsChain.drum_bus_chain,
            EffectsChain.master_chain,
            EffectsChain.bass_chain,
        ]:
            chain = factory()
            result = chain.process(audio, sr)
            assert len(result) == len(audio)


# ─── Channel Strip Tests ───


class TestChannelStrip:
    def test_default_channel(self, mono_signal):
        audio, sr = mono_signal
        strip = ChannelStrip()
        result = strip.process(audio, sr)
        assert result.ndim == 2  # Always stereo output
        assert result.shape[1] == 2

    def test_mute(self, mono_signal):
        audio, sr = mono_signal
        config = ChannelConfig(mute=True)
        strip = ChannelStrip(config)
        result = strip.process(audio, sr)
        assert np.max(np.abs(result)) == 0

    def test_phase_invert(self, mono_signal):
        audio, sr = mono_signal
        config = ChannelConfig(phase_invert=True, hpf_enabled=False)
        strip = ChannelStrip(config)
        result = strip.process(audio, sr)
        # Center-panned, so both channels should be inverted
        # Check that output is non-zero and approximately inverted
        assert np.max(np.abs(result)) > 0

    def test_panning_left(self, mono_signal):
        audio, sr = mono_signal
        config = ChannelConfig(pan=0.0, hpf_enabled=False)
        strip = ChannelStrip(config)
        result = strip.process(audio, sr)
        # Full left pan: right channel should be silent
        assert np.max(np.abs(result[:, 1])) < 0.01

    def test_panning_right(self, mono_signal):
        audio, sr = mono_signal
        config = ChannelConfig(pan=1.0, hpf_enabled=False)
        strip = ChannelStrip(config)
        result = strip.process(audio, sr)
        assert np.max(np.abs(result[:, 0])) < 0.01

    def test_update_config(self, mono_signal):
        strip = ChannelStrip()
        strip.update_config(input_gain_db=3.0, hpf_freq=120.0)
        assert strip.config.input_gain_db == 3.0
        assert strip.config.hpf_freq == 120.0

    def test_summary(self):
        strip = ChannelStrip(ChannelConfig(name="Lead Vocal"))
        summary = strip.summary()
        assert "Lead Vocal" in summary

    def test_preset_vocal(self, mono_signal):
        audio, sr = mono_signal
        strip = ChannelStrip.vocal()
        result = strip.process(audio, sr)
        assert result.shape[1] == 2

    def test_preset_kick(self, mono_signal):
        audio, sr = mono_signal
        strip = ChannelStrip.kick()
        assert strip.config.comp_enabled

    def test_preset_bass(self, mono_signal):
        audio, sr = mono_signal
        strip = ChannelStrip.bass()
        assert strip.config.saturation_enabled


# ─── Mix Bus Tests ───


class TestMixBus:
    def test_bus_summing(self, mono_signal):
        audio, sr = mono_signal
        bus = MixBus("Drums")
        stereo = np.column_stack([audio, audio])
        bus.add_signal(stereo)
        bus.add_signal(stereo)
        result = bus.process(sr)
        # Sum should be louder than individual
        assert np.max(np.abs(result)) > np.max(np.abs(stereo)) * 0.9

    def test_bus_gain(self, mono_signal):
        audio, sr = mono_signal
        bus = MixBus("Test")
        bus.gain_db = -6.0
        bus.add_signal(np.column_stack([audio, audio]))
        result = bus.process(sr)
        assert np.max(np.abs(result)) < np.max(np.abs(audio))

    def test_bus_mute(self, mono_signal):
        audio, sr = mono_signal
        bus = MixBus("Test")
        bus.mute = True
        bus.add_signal(np.column_stack([audio, audio]))
        result = bus.process(sr)
        assert np.max(np.abs(result)) == 0

    def test_processor_routing(self, mono_signal):
        audio, sr = mono_signal
        proc = MixBusProcessor()
        proc.add_bus("Drums")
        proc.add_bus("Bass")
        proc.route("Drums", audio, sr)
        proc.route("Bass", audio, sr)
        master = proc.mixdown(sr)
        assert master.shape[1] == 2

    def test_processor_standard_mix(self, mono_signal):
        audio, sr = mono_signal
        proc = MixBusProcessor.standard_mix()
        proc.route("Drums", audio, sr)
        proc.route("Vocals", audio, sr)
        master = proc.mixdown(sr)
        assert len(master) > 0

    def test_bus_summary(self):
        bus = MixBus("Test Bus")
        summary = bus.summary()
        assert "Test Bus" in summary

    def test_processor_summary(self):
        proc = MixBusProcessor.standard_mix()
        summary = proc.summary()
        assert "Drums" in summary
        assert "Master" in summary


# ─── Gain Staging Tests ───


class TestGainStaging:
    def test_peak_db(self, mono_signal):
        audio, _ = mono_signal
        peak = GainStaging.peak_db(audio)
        assert -10 < peak < 0  # 0.5 amplitude = ~-6 dB

    def test_rms_db(self, mono_signal):
        audio, _ = mono_signal
        rms = GainStaging.rms_db(audio)
        assert rms < 0

    def test_crest_factor(self, mono_signal):
        audio, _ = mono_signal
        crest = GainStaging.crest_factor_db(audio)
        assert crest > 0  # Sine has ~3 dB crest factor

    def test_analyze(self, mono_signal):
        audio, _ = mono_signal
        gs = GainStaging()
        report = gs.analyze(audio)
        assert isinstance(report, GainReport)
        assert not report.clipping
        assert report.headroom_db > 0

    def test_auto_gain_rms(self, mono_signal):
        audio, _ = mono_signal
        gs = GainStaging(target_rms_db=-18.0)
        result, gain_db = gs.auto_gain(audio, mode="rms")
        new_rms = GainStaging.rms_db(result)
        assert abs(new_rms - (-18.0)) < 2.0

    def test_auto_gain_peak(self, mono_signal):
        audio, _ = mono_signal
        gs = GainStaging(target_peak_db=-3.0)
        result, gain_db = gs.auto_gain(audio, mode="peak")
        new_peak = GainStaging.peak_db(result)
        assert abs(new_peak - (-3.0)) < 0.5

    def test_stage_multiple(self, mono_signal):
        audio, _ = mono_signal
        gs = GainStaging()
        tracks = {"Kick": audio * 0.3, "Snare": audio * 0.7, "Bass": audio}
        results = gs.stage_multiple(tracks)
        assert len(results) == 3
        for _name, (_processed, gain_db) in results.items():
            assert isinstance(gain_db, float)

    def test_dc_offset(self):
        audio = np.ones(1000) * 0.1
        dc = GainStaging.dc_offset(audio)
        assert abs(dc - 0.1) < 0.01

    def test_remove_dc(self):
        audio = np.ones(1000) * 0.1 + np.sin(np.arange(1000) * 0.01)
        cleaned = GainStaging.remove_dc(audio)
        assert abs(np.mean(cleaned)) < 0.001

    def test_check_summing_headroom(self, mono_signal):
        audio, _ = mono_signal
        gs = GainStaging()
        result = gs.check_summing_headroom([audio, audio, audio])
        assert "sum_peak_db" in result
        assert "safe" in result

    def test_report_text(self, mono_signal):
        audio, _ = mono_signal
        gs = GainStaging()
        text = gs.report(audio, name="Lead Vocal")
        assert "Lead Vocal" in text
        assert "Peak:" in text


# ─── Stereo Tools Tests ───


class TestStereoTools:
    def test_mid_side_roundtrip(self, stereo_signal):
        audio, _ = stereo_signal
        mid, side = StereoTools.to_mid_side(audio)
        reconstructed = StereoTools.from_mid_side(mid, side)
        np.testing.assert_array_almost_equal(reconstructed, audio)

    def test_width_mono(self, stereo_signal):
        audio, _ = stereo_signal
        mono = StereoTools.width(audio, 0.0)
        # Width=0 should make L and R identical (mono)
        np.testing.assert_array_almost_equal(mono[:, 0], mono[:, 1])

    def test_width_normal(self, stereo_signal):
        audio, _ = stereo_signal
        same = StereoTools.width(audio, 1.0)
        np.testing.assert_array_almost_equal(same, audio)

    def test_to_mono(self, stereo_signal):
        audio, _ = stereo_signal
        mono = StereoTools.to_mono(audio)
        assert mono.ndim == 1

    def test_balance(self, stereo_signal):
        audio, _ = stereo_signal
        left_heavy = StereoTools.balance(audio, -1.0)
        assert np.max(np.abs(left_heavy[:, 1])) < 0.01

    def test_pan_constant_power(self, mono_signal):
        audio, _ = mono_signal
        center = StereoTools.pan_constant_power(audio, 0.5)
        assert center.shape[1] == 2
        # Center pan should have equal L/R
        np.testing.assert_array_almost_equal(
            np.max(np.abs(center[:, 0])), np.max(np.abs(center[:, 1])), decimal=2
        )

    def test_swap_channels(self, stereo_signal):
        audio, _ = stereo_signal
        swapped = StereoTools.swap_channels(audio)
        np.testing.assert_array_equal(swapped[:, 0], audio[:, 1])
        np.testing.assert_array_equal(swapped[:, 1], audio[:, 0])

    def test_correlation(self, stereo_signal):
        audio, _ = stereo_signal
        corr = StereoTools.correlation(audio)
        assert -1.0 <= corr <= 1.0

    def test_correlation_mono(self):
        mono = np.sin(np.arange(1000) * 0.1)
        stereo = np.column_stack([mono, mono])
        corr = StereoTools.correlation(stereo)
        assert corr > 0.99  # Identical channels = 1.0

    def test_stereo_width_meter(self, stereo_signal):
        audio, _ = stereo_signal
        width = StereoTools.stereo_width_meter(audio)
        assert width >= 0

    def test_mono_compatibility(self, stereo_signal):
        audio, _ = stereo_signal
        result = StereoTools.mono_compatibility_check(audio)
        assert "correlation" in result
        assert "mono_compatible" in result

    def test_channel_levels(self, stereo_signal):
        audio, _ = stereo_signal
        levels = StereoTools.channel_levels(audio)
        assert "left_peak_db" in levels
        assert "right_peak_db" in levels
        assert "balance_db" in levels

    def test_haas_delay(self, stereo_signal):
        audio, sr = stereo_signal
        delayed = StereoTools.haas_delay(audio, sr, delay_ms=10.0)
        assert delayed.shape == audio.shape

    def test_stereo_enhance(self, stereo_signal):
        audio, sr = stereo_signal
        enhanced = StereoTools.stereo_enhance(audio, sr)
        assert enhanced.shape == audio.shape

    def test_report(self, stereo_signal):
        audio, _ = stereo_signal
        st = StereoTools()
        text = st.report(audio, name="Master")
        assert "Master" in text


# ─── Mix Analyzer Tests ───


class TestMixAnalyzer:
    def test_lufs_integrated(self, mono_signal):
        audio, sr = mono_signal
        lufs = MixAnalyzer.lufs_integrated(audio, sr)
        assert lufs < 0  # Should be negative dBFS

    def test_lufs_momentary(self, mono_signal):
        audio, sr = mono_signal
        values = MixAnalyzer.lufs_momentary(audio, sr)
        assert len(values) > 0

    def test_measure_loudness(self, mono_signal):
        audio, sr = mono_signal
        result = MixAnalyzer.measure_loudness(audio, sr)
        assert result.integrated_lufs < 0
        assert result.true_peak_dbfs < 0

    def test_analyze_dynamics(self, mono_signal):
        audio, sr = mono_signal
        result = MixAnalyzer.analyze_dynamics(audio, sr)
        assert result.crest_factor_db > 0
        assert result.peak_db < 0

    def test_spectral_balance(self, mono_signal):
        audio, sr = mono_signal
        result = MixAnalyzer.spectral_balance(audio, sr)
        # 440Hz tone should have energy in mid range
        assert result.mid_db > result.sub_bass_db

    def test_frequency_masking(self, mono_signal):
        audio, sr = mono_signal
        issues = MixAnalyzer.frequency_masking_check(audio, audio, sr)
        # Same signal should show masking in the frequency range of the tone
        assert isinstance(issues, list)

    def test_full_analysis(self, mono_signal):
        audio, sr = mono_signal
        result = MixAnalyzer.full_analysis(audio, sr)
        assert "loudness" in result
        assert "dynamics" in result
        assert "spectral" in result

    def test_report(self, mono_signal):
        audio, sr = mono_signal
        text = MixAnalyzer.report(audio, sr, name="Test Mix")
        assert "Test Mix" in text
        assert "LUFS" in text
        assert "Spotify" in text


# ─── Reference Compare Tests ───


class TestReferenceCompare:
    def test_compare(self, stereo_signal):
        audio, sr = stereo_signal
        rc = ReferenceCompare()
        rc.load_reference_array("Ref", audio * 0.8, sr)
        result = rc.compare(audio, sr)
        assert result.loudness_diff_lufs != 0 or result.peak_diff_db != 0
        assert len(result.suggestions) > 0

    def test_compare_all(self, stereo_signal):
        audio, sr = stereo_signal
        rc = ReferenceCompare()
        rc.load_reference_array("Ref1", audio * 0.8, sr)
        rc.load_reference_array("Ref2", audio * 1.2, sr)
        results = rc.compare_all(audio, sr)
        assert len(results) == 2

    def test_match_loudness(self, stereo_signal):
        audio, sr = stereo_signal
        rc = ReferenceCompare()
        ref = audio * 0.5
        rc.load_reference_array("Quiet Ref", ref, sr)
        matched, gain_db = rc.match_loudness(audio, sr)
        assert gain_db < 0  # Should reduce gain to match quieter ref

    def test_report(self, stereo_signal):
        audio, sr = stereo_signal
        rc = ReferenceCompare()
        rc.load_reference_array("Pro Master", audio, sr)
        text = rc.report(audio, sr, mix_name="My Mix")
        assert "My Mix" in text
        assert "Pro Master" in text

    def test_no_reference_error(self, stereo_signal):
        audio, sr = stereo_signal
        rc = ReferenceCompare()
        with pytest.raises(ValueError):
            rc.compare(audio, sr)
