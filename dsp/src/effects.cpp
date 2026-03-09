#include "effects.h"
#include <cstring>

namespace fl_dsp {

// ─── Delay Line ───

void DelayLine::init(int max_delay_samples) {
    buffer_.resize(max_delay_samples + 1, 0.0f);
    write_pos_ = 0;
}

void DelayLine::set_delay(float delay_samples) {
    delay_ = delay_samples;
    delay_int_ = static_cast<int>(delay_samples);
    delay_frac_ = delay_samples - static_cast<float>(delay_int_);
}

void DelayLine::reset() {
    std::fill(buffer_.begin(), buffer_.end(), 0.0f);
    write_pos_ = 0;
}

void DelayLine::write(float sample) {
    buffer_[write_pos_] = sample;
    write_pos_++;
    if (write_pos_ >= static_cast<int>(buffer_.size())) write_pos_ = 0;
}

float DelayLine::read() const {
    return read_at(delay_);
}

float DelayLine::read_at(float delay_samples) const {
    int d_int = static_cast<int>(delay_samples);
    float d_frac = delay_samples - static_cast<float>(d_int);
    int size = static_cast<int>(buffer_.size());

    int pos1 = write_pos_ - d_int - 1;
    if (pos1 < 0) pos1 += size;
    int pos2 = pos1 - 1;
    if (pos2 < 0) pos2 += size;

    // Linear interpolation
    return buffer_[pos1] + d_frac * (buffer_[pos2] - buffer_[pos1]);
}

float DelayLine::process(float input, float feedback) {
    float output = read();
    write(input + output * feedback);
    return output;
}

// ─── Stereo Delay ───

void StereoDelay::init(float max_delay_ms, float sample_rate) {
    sample_rate_ = sample_rate;
    int max_samples = static_cast<int>(max_delay_ms * sample_rate / 1000.0f) + 1;
    delay_l_.init(max_samples);
    delay_r_.init(max_samples);
}

void StereoDelay::set_params(float left_ms, float right_ms, float feedback,
                              float mix, float filter_freq) {
    delay_l_.set_delay(left_ms * sample_rate_ / 1000.0f);
    delay_r_.set_delay(right_ms * sample_rate_ / 1000.0f);
    feedback_ = clamp(feedback, 0.0f, 0.95f);
    mix_ = clamp(mix, 0.0f, 1.0f);
    filter_l_.set_lowpass(filter_freq, sample_rate_);
    filter_r_.set_lowpass(filter_freq, sample_rate_);
}

void StereoDelay::reset() {
    delay_l_.reset();
    delay_r_.reset();
    filter_l_.reset();
    filter_r_.reset();
}

void StereoDelay::process(float in_l, float in_r, float& out_l, float& out_r) {
    float del_l = filter_l_.process(delay_l_.read());
    float del_r = filter_r_.process(delay_r_.read());

    delay_l_.write(in_l + del_l * feedback_);
    delay_r_.write(in_r + del_r * feedback_);

    out_l = in_l * (1.0f - mix_) + del_l * mix_;
    out_r = in_r * (1.0f - mix_) + del_r * mix_;
}

// ─── Schroeder Reverb ───

void SchroederReverb::init(float sample_rate) {
    sample_rate_ = sample_rate;

    // Comb filter delay times (in samples, based on Schroeder's original values)
    int comb_delays[] = {1557, 1617, 1491, 1422};
    float scale = sample_rate / 44100.0f;

    for (int i = 0; i < NUM_COMBS; ++i) {
        int delay = static_cast<int>(comb_delays[i] * scale);
        combs_[i].init(delay + 100);
        combs_[i].set_params(delay, 0.84f, 0.2f);
    }

    // Allpass delay times
    int ap_delays[] = {225, 556};
    for (int i = 0; i < NUM_ALLPASS; ++i) {
        int delay = static_cast<int>(ap_delays[i] * scale);
        allpasses_[i].init(delay + 100);
        allpasses_[i].set_params(delay, 0.5f);
    }
}

void SchroederReverb::set_params(float room_size, float damping, float mix) {
    mix_ = clamp(mix, 0.0f, 1.0f);

    float feedback = 0.7f + room_size * 0.28f;
    for (auto& comb : combs_) {
        comb.set_feedback(feedback, damping);
    }
}

void SchroederReverb::reset() {
    for (auto& comb : combs_) comb.reset();
    for (auto& ap : allpasses_) ap.reset();
}

float SchroederReverb::process_mono(float input) {
    // Parallel comb filters
    float comb_sum = 0.0f;
    for (auto& comb : combs_) {
        comb_sum += comb.process(input);
    }
    comb_sum /= static_cast<float>(NUM_COMBS);

    // Series allpass filters
    float output = comb_sum;
    for (auto& ap : allpasses_) {
        output = ap.process(output);
    }

    return input * (1.0f - mix_) + output * mix_;
}

void SchroederReverb::process(float in_l, float in_r, float& out_l, float& out_r) {
    float mono = (in_l + in_r) * 0.5f;
    float wet = process_mono(mono) - mono * (1.0f - mix_);  // Extract wet signal

    out_l = in_l * (1.0f - mix_) + wet * mix_;
    out_r = in_r * (1.0f - mix_) + wet * mix_;
}

// ─── Distortion ───

void Distortion::set_params(DistortionType type, float drive, float mix, float tone) {
    type_ = type;
    drive_ = std::max(0.1f, drive);
    mix_ = clamp(mix, 0.0f, 1.0f);
    tone_ = tone;
}

void Distortion::reset() {
    tone_filter_.reset();
}

float Distortion::process(float input) {
    float driven = input * drive_;
    float distorted;

    switch (type_) {
    case DistortionType::SoftClip:
        distorted = soft_clip(driven);
        break;

    case DistortionType::HardClip:
        distorted = clamp(driven, -1.0f, 1.0f);
        break;

    case DistortionType::Tanh:
        distorted = std::tanh(driven);
        break;

    case DistortionType::Foldback: {
        float x = driven;
        while (x > 1.0f || x < -1.0f) {
            if (x > 1.0f) x = 2.0f - x;
            if (x < -1.0f) x = -2.0f - x;
        }
        distorted = x;
        break;
    }

    case DistortionType::BitCrush: {
        int bits = std::max(1, static_cast<int>(16.0f / drive_));
        float levels = static_cast<float>(1 << bits);
        distorted = std::round(driven * levels) / levels;
        break;
    }

    case DistortionType::Tube: {
        // Asymmetric soft clipping (tube-like)
        if (driven >= 0.0f) {
            distorted = 1.0f - std::exp(-driven);
        } else {
            distorted = -1.0f + std::exp(driven);
        }
        // Slight even harmonic generation
        distorted += 0.1f * driven * driven * (driven > 0 ? 1.0f : -1.0f);
        distorted = clamp(distorted, -1.0f, 1.0f);
        break;
    }

    default:
        distorted = soft_clip(driven);
    }

    // Apply tone filter
    distorted = tone_filter_.process(distorted);

    // Compensate for gain increase
    float compensation = 1.0f / std::max(1.0f, drive_ * 0.5f);
    distorted *= compensation;

    return input * (1.0f - mix_) + distorted * mix_;
}

void Distortion::process_block(float* data, int num_samples, int stride) {
    for (int i = 0; i < num_samples; ++i) {
        data[i * stride] = process(data[i * stride]);
    }
}

// ─── Compressor ───

void Compressor::set_params(float threshold_db, float ratio, float attack_ms,
                             float release_ms, float makeup_db, float sample_rate) {
    threshold_ = threshold_db;
    ratio_ = std::max(1.0f, ratio);
    attack_coeff_ = smooth_coeff(attack_ms, sample_rate);
    release_coeff_ = smooth_coeff(release_ms, sample_rate);
    makeup_ = db_to_linear(makeup_db);
}

void Compressor::reset() {
    envelope_ = 0.0f;
    gain_reduction_db_ = 0.0f;
}

float Compressor::process(float input) {
    return process_sidechain(input, input);
}

float Compressor::process_sidechain(float input, float sidechain) {
    float input_db = linear_to_db(std::abs(sidechain));

    // Gain computation
    float over_db = input_db - threshold_;
    float target_db = 0.0f;
    if (over_db > 0.0f) {
        target_db = over_db * (1.0f - 1.0f / ratio_);
    }

    // Envelope follower (peak)
    float coeff = (target_db > envelope_) ? attack_coeff_ : release_coeff_;
    envelope_ += (target_db - envelope_) * coeff;

    gain_reduction_db_ = -envelope_;
    float gain = db_to_linear(-envelope_) * makeup_;

    return input * gain;
}

void Compressor::process_block(float* data, int num_samples, int stride) {
    for (int i = 0; i < num_samples; ++i) {
        data[i * stride] = process(data[i * stride]);
    }
}

// ─── Limiter ───

void Limiter::init(float sample_rate, float lookahead_ms) {
    sample_rate_ = sample_rate;
    int lookahead_samples = static_cast<int>(lookahead_ms * sample_rate / 1000.0f);
    lookahead_.init(lookahead_samples + 1);
    lookahead_.set_delay(static_cast<float>(lookahead_samples));
}

void Limiter::set_params(float ceiling_db, float release_ms) {
    ceiling_ = db_to_linear(ceiling_db);
    release_coeff_ = smooth_coeff(release_ms, sample_rate_);
}

void Limiter::reset() {
    lookahead_.reset();
    envelope_ = 0.0f;
}

float Limiter::process(float input) {
    float abs_input = std::abs(input);

    // Track envelope
    if (abs_input > envelope_) {
        envelope_ = abs_input;
    } else {
        envelope_ += (abs_input - envelope_) * release_coeff_;
    }

    // Compute gain
    float gain = 1.0f;
    if (envelope_ > ceiling_) {
        gain = ceiling_ / envelope_;
    }

    // Apply to delayed signal
    float delayed = lookahead_.process(input, 0.0f);
    return delayed * gain;
}

// ─── Chorus ───

void Chorus::init(float sample_rate) {
    sample_rate_ = sample_rate;
    int max_samples = static_cast<int>(50.0f * sample_rate / 1000.0f);
    delay_l_.init(max_samples);
    delay_r_.init(max_samples);
    center_samples_ = 7.0f * sample_rate / 1000.0f;  // 7ms center delay
}

void Chorus::set_params(float rate_hz, float depth_ms, float mix, float feedback) {
    phase_inc_ = rate_hz / sample_rate_;
    depth_samples_ = depth_ms * sample_rate_ / 1000.0f;
    mix_ = clamp(mix, 0.0f, 1.0f);
    feedback_ = clamp(feedback, 0.0f, 0.9f);
}

void Chorus::reset() {
    delay_l_.reset();
    delay_r_.reset();
    phase_ = 0.0f;
}

void Chorus::process(float in_l, float in_r, float& out_l, float& out_r) {
    // LFO modulation
    float mod_l = std::sin(TWO_PI * phase_);
    float mod_r = std::sin(TWO_PI * phase_ + HALF_PI);  // 90 degree offset for stereo

    float delay_l = center_samples_ + depth_samples_ * mod_l;
    float delay_r = center_samples_ + depth_samples_ * mod_r;

    delay_l_.set_delay(delay_l);
    delay_r_.set_delay(delay_r);

    float wet_l = delay_l_.process(in_l, feedback_);
    float wet_r = delay_r_.process(in_r, feedback_);

    out_l = in_l * (1.0f - mix_) + wet_l * mix_;
    out_r = in_r * (1.0f - mix_) + wet_r * mix_;

    phase_ += phase_inc_;
    if (phase_ >= 1.0f) phase_ -= 1.0f;
}

} // namespace fl_dsp
