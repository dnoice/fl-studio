#pragma once
/**
 * @file filters.h
 * @brief Digital audio filters: biquad, state-variable, comb, allpass.
 */

#include "dsp_core.h"

namespace fl_dsp {

// ─── Biquad Filter ───

enum class BiquadType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
    Peaking,
    LowShelf,
    HighShelf,
    AllPass,
};

/**
 * Standard biquad (second-order IIR) filter.
 * Implements all common filter types.
 */
class BiquadFilter {
public:
    BiquadFilter() = default;

    void set_params(BiquadType type, float freq, float q, float gain_db,
                    float sample_rate);
    void reset();

    float process(float input);
    void process_block(float* data, int num_samples, int stride = 1);

    // Direct coefficient access
    void set_coefficients(float b0, float b1, float b2, float a1, float a2);

private:
    // Coefficients
    float b0_ = 1.0f, b1_ = 0.0f, b2_ = 0.0f;
    float a1_ = 0.0f, a2_ = 0.0f;

    // State
    float x1_ = 0.0f, x2_ = 0.0f;  // Input delay line
    float y1_ = 0.0f, y2_ = 0.0f;  // Output delay line
};

// ─── State Variable Filter ───

/**
 * State variable filter (SVF) - more stable at high frequencies than biquad.
 * Provides simultaneous low-pass, high-pass, and band-pass outputs.
 */
class StateVariableFilter {
public:
    StateVariableFilter() = default;

    void set_params(float freq, float q, float sample_rate);
    void reset();

    struct Output {
        float low;
        float high;
        float band;
        float notch;
    };

    Output process(float input);
    void process_block_lowpass(float* data, int num_samples, int stride = 1);
    void process_block_highpass(float* data, int num_samples, int stride = 1);
    void process_block_bandpass(float* data, int num_samples, int stride = 1);

private:
    float f_ = 0.0f;   // Frequency coefficient
    float q_ = 0.5f;   // Damping (1/Q)
    float low_ = 0.0f;
    float band_ = 0.0f;
};

// ─── Comb Filter ───

/**
 * Feedback comb filter - basis for reverb and flanging effects.
 */
class CombFilter {
public:
    CombFilter() = default;

    void init(int max_delay_samples);
    void set_params(int delay_samples, float feedback, float damping);
    void set_feedback(float feedback, float damping);
    void reset();

    float process(float input);

private:
    std::vector<float> buffer_;
    int write_pos_ = 0;
    int delay_ = 0;
    float feedback_ = 0.5f;
    float damping_ = 0.0f;
    float prev_ = 0.0f;  // For damping (one-pole lowpass)
};

// ─── Allpass Filter ───

/**
 * Allpass filter - passes all frequencies but changes phase.
 * Used in reverb diffusion networks and phaser effects.
 */
class AllpassFilter {
public:
    AllpassFilter() = default;

    void init(int max_delay_samples);
    void set_params(int delay_samples, float coefficient);
    void reset();

    float process(float input);

private:
    std::vector<float> buffer_;
    int write_pos_ = 0;
    int delay_ = 0;
    float coeff_ = 0.5f;
};

// ─── DC Blocker ───

/**
 * Simple DC blocking filter (first-order highpass).
 */
class DCBlocker {
public:
    DCBlocker(float coeff = 0.995f) : coeff_(coeff) {}

    void reset() { x1_ = y1_ = 0.0f; }

    float process(float input) {
        float output = input - x1_ + coeff_ * y1_;
        x1_ = input;
        y1_ = output;
        return output;
    }

private:
    float coeff_;
    float x1_ = 0.0f;
    float y1_ = 0.0f;
};

// ─── One-Pole Filter ───

/**
 * Simple one-pole (first-order) lowpass/highpass filter.
 * Good for smoothing parameters.
 */
class OnePole {
public:
    OnePole() = default;

    void set_lowpass(float freq, float sample_rate);
    void set_highpass(float freq, float sample_rate);
    void set_coefficient(float coeff) { a0_ = coeff; b1_ = 1.0f - coeff; }
    void reset() { y1_ = 0.0f; }

    float process(float input) {
        y1_ = input * a0_ + y1_ * b1_;
        return y1_;
    }

    float current() const { return y1_; }

private:
    float a0_ = 1.0f;
    float b1_ = 0.0f;
    float y1_ = 0.0f;
};

} // namespace fl_dsp
