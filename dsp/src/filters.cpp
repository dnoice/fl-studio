#include "filters.h"
#include <cstring>

namespace fl_dsp {

// ─── Biquad Filter ───

void BiquadFilter::set_params(BiquadType type, float freq, float q, float gain_db,
                               float sample_rate) {
    float w0 = TWO_PI * freq / sample_rate;
    float cos_w0 = std::cos(w0);
    float sin_w0 = std::sin(w0);
    float alpha = sin_w0 / (2.0f * q);

    float a0;

    switch (type) {
    case BiquadType::LowPass:
        b0_ = (1.0f - cos_w0) / 2.0f;
        b1_ = 1.0f - cos_w0;
        b2_ = (1.0f - cos_w0) / 2.0f;
        a0 = 1.0f + alpha;
        a1_ = -2.0f * cos_w0;
        a2_ = 1.0f - alpha;
        break;

    case BiquadType::HighPass:
        b0_ = (1.0f + cos_w0) / 2.0f;
        b1_ = -(1.0f + cos_w0);
        b2_ = (1.0f + cos_w0) / 2.0f;
        a0 = 1.0f + alpha;
        a1_ = -2.0f * cos_w0;
        a2_ = 1.0f - alpha;
        break;

    case BiquadType::BandPass:
        b0_ = alpha;
        b1_ = 0.0f;
        b2_ = -alpha;
        a0 = 1.0f + alpha;
        a1_ = -2.0f * cos_w0;
        a2_ = 1.0f - alpha;
        break;

    case BiquadType::Notch:
        b0_ = 1.0f;
        b1_ = -2.0f * cos_w0;
        b2_ = 1.0f;
        a0 = 1.0f + alpha;
        a1_ = -2.0f * cos_w0;
        a2_ = 1.0f - alpha;
        break;

    case BiquadType::Peaking: {
        float A = std::pow(10.0f, gain_db / 40.0f);
        b0_ = 1.0f + alpha * A;
        b1_ = -2.0f * cos_w0;
        b2_ = 1.0f - alpha * A;
        a0 = 1.0f + alpha / A;
        a1_ = -2.0f * cos_w0;
        a2_ = 1.0f - alpha / A;
        break;
    }

    case BiquadType::LowShelf: {
        float A = std::pow(10.0f, gain_db / 40.0f);
        float sq = 2.0f * std::sqrt(A) * alpha;
        b0_ = A * ((A + 1.0f) - (A - 1.0f) * cos_w0 + sq);
        b1_ = 2.0f * A * ((A - 1.0f) - (A + 1.0f) * cos_w0);
        b2_ = A * ((A + 1.0f) - (A - 1.0f) * cos_w0 - sq);
        a0 = (A + 1.0f) + (A - 1.0f) * cos_w0 + sq;
        a1_ = -2.0f * ((A - 1.0f) + (A + 1.0f) * cos_w0);
        a2_ = (A + 1.0f) + (A - 1.0f) * cos_w0 - sq;
        break;
    }

    case BiquadType::HighShelf: {
        float A = std::pow(10.0f, gain_db / 40.0f);
        float sq = 2.0f * std::sqrt(A) * alpha;
        b0_ = A * ((A + 1.0f) + (A - 1.0f) * cos_w0 + sq);
        b1_ = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cos_w0);
        b2_ = A * ((A + 1.0f) + (A - 1.0f) * cos_w0 - sq);
        a0 = (A + 1.0f) - (A - 1.0f) * cos_w0 + sq;
        a1_ = 2.0f * ((A - 1.0f) - (A + 1.0f) * cos_w0);
        a2_ = (A + 1.0f) - (A - 1.0f) * cos_w0 - sq;
        break;
    }

    case BiquadType::AllPass:
        b0_ = 1.0f - alpha;
        b1_ = -2.0f * cos_w0;
        b2_ = 1.0f + alpha;
        a0 = 1.0f + alpha;
        a1_ = -2.0f * cos_w0;
        a2_ = 1.0f - alpha;
        break;
    }

    // Normalize
    b0_ /= a0;
    b1_ /= a0;
    b2_ /= a0;
    a1_ /= a0;
    a2_ /= a0;
}

void BiquadFilter::reset() {
    x1_ = x2_ = y1_ = y2_ = 0.0f;
}

float BiquadFilter::process(float input) {
    float output = b0_ * input + b1_ * x1_ + b2_ * x2_ - a1_ * y1_ - a2_ * y2_;
    x2_ = x1_; x1_ = input;
    y2_ = y1_; y1_ = output;
    return output;
}

void BiquadFilter::process_block(float* data, int num_samples, int stride) {
    for (int i = 0; i < num_samples; ++i) {
        data[i * stride] = process(data[i * stride]);
    }
}

void BiquadFilter::set_coefficients(float b0, float b1, float b2, float a1, float a2) {
    b0_ = b0; b1_ = b1; b2_ = b2; a1_ = a1; a2_ = a2;
}

// ─── State Variable Filter ───

void StateVariableFilter::set_params(float freq, float q, float sample_rate) {
    f_ = 2.0f * std::sin(PI * freq / sample_rate);
    q_ = 1.0f / q;
}

void StateVariableFilter::reset() {
    low_ = band_ = 0.0f;
}

StateVariableFilter::Output StateVariableFilter::process(float input) {
    float high = input - low_ - q_ * band_;
    band_ += f_ * high;
    low_ += f_ * band_;
    float notch = high + low_;
    return {low_, high, band_, notch};
}

void StateVariableFilter::process_block_lowpass(float* data, int num_samples, int stride) {
    for (int i = 0; i < num_samples; ++i) {
        auto out = process(data[i * stride]);
        data[i * stride] = out.low;
    }
}

void StateVariableFilter::process_block_highpass(float* data, int num_samples, int stride) {
    for (int i = 0; i < num_samples; ++i) {
        auto out = process(data[i * stride]);
        data[i * stride] = out.high;
    }
}

void StateVariableFilter::process_block_bandpass(float* data, int num_samples, int stride) {
    for (int i = 0; i < num_samples; ++i) {
        auto out = process(data[i * stride]);
        data[i * stride] = out.band;
    }
}

// ─── Comb Filter ───

void CombFilter::init(int max_delay_samples) {
    buffer_.resize(max_delay_samples, 0.0f);
    write_pos_ = 0;
}

void CombFilter::set_params(int delay_samples, float feedback, float damping) {
    delay_ = std::min(delay_samples, static_cast<int>(buffer_.size()) - 1);
    feedback_ = feedback;
    damping_ = damping;
}

void CombFilter::set_feedback(float feedback, float damping) {
    feedback_ = feedback;
    damping_ = damping;
}

void CombFilter::reset() {
    std::fill(buffer_.begin(), buffer_.end(), 0.0f);
    prev_ = 0.0f;
    write_pos_ = 0;
}

float CombFilter::process(float input) {
    int read_pos = write_pos_ - delay_;
    if (read_pos < 0) read_pos += static_cast<int>(buffer_.size());

    float output = buffer_[read_pos];

    // Apply damping (one-pole lowpass on feedback path)
    float filtered = output * (1.0f - damping_) + prev_ * damping_;
    prev_ = filtered;

    buffer_[write_pos_] = input + filtered * feedback_;

    write_pos_++;
    if (write_pos_ >= static_cast<int>(buffer_.size())) write_pos_ = 0;

    return output;
}

// ─── Allpass Filter ───

void AllpassFilter::init(int max_delay_samples) {
    buffer_.resize(max_delay_samples, 0.0f);
    write_pos_ = 0;
}

void AllpassFilter::set_params(int delay_samples, float coefficient) {
    delay_ = std::min(delay_samples, static_cast<int>(buffer_.size()) - 1);
    coeff_ = coefficient;
}

void AllpassFilter::reset() {
    std::fill(buffer_.begin(), buffer_.end(), 0.0f);
    write_pos_ = 0;
}

float AllpassFilter::process(float input) {
    int read_pos = write_pos_ - delay_;
    if (read_pos < 0) read_pos += static_cast<int>(buffer_.size());

    float delayed = buffer_[read_pos];
    float output = -input * coeff_ + delayed;
    buffer_[write_pos_] = input + delayed * coeff_;

    write_pos_++;
    if (write_pos_ >= static_cast<int>(buffer_.size())) write_pos_ = 0;

    return output;
}

// ─── One-Pole Filter ───

void OnePole::set_lowpass(float freq, float sample_rate) {
    float x = std::exp(-TWO_PI * freq / sample_rate);
    a0_ = 1.0f - x;
    b1_ = x;
}

void OnePole::set_highpass(float freq, float sample_rate) {
    float x = std::exp(-TWO_PI * freq / sample_rate);
    a0_ = (1.0f + x) / 2.0f;
    b1_ = -x;
}

} // namespace fl_dsp
