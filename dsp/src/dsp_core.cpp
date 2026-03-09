#include "dsp_core.h"
#include <cstring>
#include <numeric>

namespace fl_dsp {

AudioBuffer::AudioBuffer(int num_samples, int num_channels)
    : data_(num_samples * num_channels, 0.0f)
    , frames_(num_samples)
    , channels_(num_channels)
{
}

void AudioBuffer::resize(int num_samples, int num_channels) {
    frames_ = num_samples;
    channels_ = num_channels;
    data_.resize(num_samples * num_channels, 0.0f);
}

void AudioBuffer::clear() {
    std::fill(data_.begin(), data_.end(), 0.0f);
}

void AudioBuffer::apply_gain(float gain) {
    for (auto& s : data_) {
        s *= gain;
    }
}

void AudioBuffer::apply_gain_ramp(float start_gain, float end_gain) {
    int total = frames_;
    for (int i = 0; i < total; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(total - 1);
        float gain = lerp(start_gain, end_gain, t);
        for (int ch = 0; ch < channels_; ++ch) {
            data_[i * channels_ + ch] *= gain;
        }
    }
}

float AudioBuffer::peak_level() const {
    float peak = 0.0f;
    for (const auto& s : data_) {
        float abs_s = std::abs(s);
        if (abs_s > peak) peak = abs_s;
    }
    return peak;
}

float AudioBuffer::rms_level() const {
    if (data_.empty()) return 0.0f;
    float sum_sq = 0.0f;
    for (const auto& s : data_) {
        sum_sq += s * s;
    }
    return std::sqrt(sum_sq / static_cast<float>(data_.size()));
}

} // namespace fl_dsp
