#pragma once
/**
 * @file effects.h
 * @brief Audio effects: delay, reverb, distortion, compressor, chorus.
 */

#include "dsp_core.h"
#include "filters.h"
#include <array>

namespace fl_dsp {

// ─── Delay Line ───

/**
 * Fractional delay line with linear interpolation.
 * Supports feedback and cross-feedback for stereo effects.
 */
class DelayLine {
public:
    DelayLine() = default;

    void init(int max_delay_samples);
    void set_delay(float delay_samples);
    void reset();

    void write(float sample);
    float read() const;
    float read_at(float delay_samples) const;

    float process(float input, float feedback = 0.0f);

private:
    std::vector<float> buffer_;
    int write_pos_ = 0;
    float delay_ = 0.0f;
    int delay_int_ = 0;
    float delay_frac_ = 0.0f;
};

// ─── Stereo Delay ───

/**
 * Stereo delay with independent left/right times, feedback, and filtering.
 */
class StereoDelay {
public:
    StereoDelay() = default;

    void init(float max_delay_ms, float sample_rate);
    void set_params(float left_ms, float right_ms, float feedback,
                    float mix, float filter_freq);
    void reset();

    void process(float in_l, float in_r, float& out_l, float& out_r);

private:
    DelayLine delay_l_, delay_r_;
    OnePole filter_l_, filter_r_;
    float feedback_ = 0.3f;
    float mix_ = 0.5f;
    float sample_rate_ = 44100.0f;
};

// ─── Simple Reverb (Schroeder) ───

/**
 * Schroeder reverb - 4 parallel comb filters + 2 series allpass filters.
 * Simple but effective algorithmic reverb.
 */
class SchroederReverb {
public:
    SchroederReverb() = default;

    void init(float sample_rate);
    void set_params(float room_size, float damping, float mix);
    void reset();

    void process(float in_l, float in_r, float& out_l, float& out_r);
    float process_mono(float input);

private:
    static constexpr int NUM_COMBS = 4;
    static constexpr int NUM_ALLPASS = 2;

    std::array<CombFilter, NUM_COMBS> combs_;
    std::array<AllpassFilter, NUM_ALLPASS> allpasses_;

    float mix_ = 0.3f;
    float sample_rate_ = 44100.0f;
};

// ─── Distortion ───

enum class DistortionType {
    SoftClip,
    HardClip,
    Tanh,
    Foldback,
    BitCrush,
    Tube,
};

/**
 * Multi-mode distortion/saturation effect.
 */
class Distortion {
public:
    Distortion() = default;

    void set_params(DistortionType type, float drive, float mix, float tone);
    void reset();

    float process(float input);
    void process_block(float* data, int num_samples, int stride = 1);

private:
    DistortionType type_ = DistortionType::SoftClip;
    float drive_ = 1.0f;
    float mix_ = 1.0f;
    OnePole tone_filter_;
    float tone_ = 0.5f;
};

// ─── Compressor ───

/**
 * Dynamic range compressor with sidechain input support.
 */
class Compressor {
public:
    Compressor() = default;

    void set_params(float threshold_db, float ratio, float attack_ms,
                    float release_ms, float makeup_db, float sample_rate);
    void reset();

    float process(float input);
    float process_sidechain(float input, float sidechain);
    void process_block(float* data, int num_samples, int stride = 1);

    float get_gain_reduction_db() const { return gain_reduction_db_; }

private:
    float threshold_ = -10.0f;
    float ratio_ = 4.0f;
    float attack_coeff_ = 0.0f;
    float release_coeff_ = 0.0f;
    float makeup_ = 1.0f;
    float envelope_ = 0.0f;
    float gain_reduction_db_ = 0.0f;
};

// ─── Limiter ───

/**
 * Brick-wall lookahead limiter.
 */
class Limiter {
public:
    Limiter() = default;

    void init(float sample_rate, float lookahead_ms = 5.0f);
    void set_params(float ceiling_db, float release_ms);
    void reset();

    float process(float input);

private:
    DelayLine lookahead_;
    float ceiling_ = 1.0f;
    float release_coeff_ = 0.0f;
    float envelope_ = 0.0f;
    float sample_rate_ = 44100.0f;
};

// ─── Chorus ───

/**
 * Stereo chorus effect using modulated delay lines.
 */
class Chorus {
public:
    Chorus() = default;

    void init(float sample_rate);
    void set_params(float rate_hz, float depth_ms, float mix, float feedback);
    void reset();

    void process(float in_l, float in_r, float& out_l, float& out_r);

private:
    DelayLine delay_l_, delay_r_;
    float phase_ = 0.0f;
    float phase_inc_ = 0.0f;
    float depth_samples_ = 0.0f;
    float center_samples_ = 0.0f;
    float mix_ = 0.5f;
    float feedback_ = 0.0f;
    float sample_rate_ = 44100.0f;
};

} // namespace fl_dsp
