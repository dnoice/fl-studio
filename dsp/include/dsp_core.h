#pragma once
/**
 * @file dsp_core.h
 * @brief Core DSP types, buffer management, and utility functions.
 */

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>

namespace fl_dsp {

// ─── Constants ───
constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;
constexpr float HALF_PI = PI / 2.0f;

// ─── Audio Buffer ───

/**
 * Simple audio buffer for processing.
 * Interleaved format: [L0, R0, L1, R1, ...] for stereo.
 */
class AudioBuffer {
public:
    AudioBuffer() = default;
    AudioBuffer(int num_samples, int num_channels = 2);

    void resize(int num_samples, int num_channels = 2);
    void clear();

    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }

    float& sample(int frame, int channel) {
        return data_[frame * channels_ + channel];
    }
    const float& sample(int frame, int channel) const {
        return data_[frame * channels_ + channel];
    }

    int frames() const { return frames_; }
    int channels() const { return channels_; }
    int total_samples() const { return static_cast<int>(data_.size()); }

    // Apply gain to entire buffer
    void apply_gain(float gain);
    void apply_gain_ramp(float start_gain, float end_gain);

    // Get peak level
    float peak_level() const;
    float rms_level() const;

private:
    std::vector<float> data_;
    int frames_ = 0;
    int channels_ = 2;
};

// ─── Utility Functions ───

/** Convert dB to linear gain */
inline float db_to_linear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

/** Convert linear gain to dB */
inline float linear_to_db(float linear) {
    return 20.0f * std::log10(std::max(linear, 1e-10f));
}

/** Convert MIDI note to frequency */
inline float midi_to_freq(int note) {
    return 440.0f * std::pow(2.0f, (note - 69) / 12.0f);
}

/** Convert frequency to MIDI note */
inline float freq_to_midi(float freq) {
    return 69.0f + 12.0f * std::log2(freq / 440.0f);
}

/** Soft clip (tanh-like) */
inline float soft_clip(float x) {
    if (x > 1.0f) return 1.0f - 1.0f / (1.0f + x);
    if (x < -1.0f) return -1.0f + 1.0f / (1.0f - x);
    return x;
}

/** Linear interpolation */
inline float lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

/** Clamp value to range */
inline float clamp(float x, float lo, float hi) {
    return std::min(std::max(x, lo), hi);
}

/** One-pole smoothing coefficient from time constant */
inline float smooth_coeff(float time_ms, float sample_rate) {
    if (time_ms <= 0.0f) return 1.0f;
    return 1.0f - std::exp(-1.0f / (time_ms * 0.001f * sample_rate));
}

} // namespace fl_dsp
