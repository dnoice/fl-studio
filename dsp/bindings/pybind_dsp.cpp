/**
 * @file pybind_dsp.cpp
 * @brief Python bindings for the FL DSP engine using pybind11.
 *
 * Build: cmake -DCMAKE_BUILD_TYPE=Release .. && make
 * Requires: pip install pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "dsp_core.h"
#include "filters.h"
#include "effects.h"

namespace py = pybind11;
using namespace fl_dsp;

PYBIND11_MODULE(fl_dsp_py, m)
{
    m.doc() = "FL Studio DSP Engine - High-performance audio processing";

    // ─── Utility Functions ───
    m.def("db_to_linear", &db_to_linear, "Convert dB to linear gain");
    m.def("linear_to_db", &linear_to_db, "Convert linear gain to dB");
    m.def("midi_to_freq", &midi_to_freq, "Convert MIDI note to frequency");
    m.def("freq_to_midi", &freq_to_midi, "Convert frequency to MIDI note");

    // ─── AudioBuffer ───
    py::class_<AudioBuffer>(m, "AudioBuffer")
        .def(py::init<int, int>(), py::arg("num_samples"), py::arg("num_channels") = 2)
        .def("clear", &AudioBuffer::clear)
        .def("apply_gain", &AudioBuffer::apply_gain)
        .def("peak_level", &AudioBuffer::peak_level)
        .def("rms_level", &AudioBuffer::rms_level)
        .def("frames", &AudioBuffer::frames)
        .def("channels", &AudioBuffer::channels);

    // ─── Biquad Filter ───
    py::enum_<BiquadType>(m, "BiquadType")
        .value("LowPass", BiquadType::LowPass)
        .value("HighPass", BiquadType::HighPass)
        .value("Notch", BiquadType::Notch)
        .value("Peaking", BiquadType::Peaking)
        .value("LowShelf", BiquadType::LowShelf)
        .value("HighShelf", BiquadType::HighShelf)
        .value("AllPass", BiquadType::AllPass);

    py::class_<BiquadFilter>(m, "BiquadFilter")
        .def(py::init<>())
        .def("set_params", &BiquadFilter::set_params)
        .def("reset", &BiquadFilter::reset)
        .def("process", &BiquadFilter::process)
        .def("process_array", [](BiquadFilter &self, py::array_t<float> data)
             {
            auto buf = data.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < buf.shape(0); i++) {
                buf(i) = self.process(buf(i));
            } });

    // ─── State Variable Filter ───
    py::class_<StateVariableFilter>(m, "StateVariableFilter")
        .def(py::init<>())
        .def("set_params", &StateVariableFilter::set_params)
        .def("reset", &StateVariableFilter::reset);

    // ─── Distortion ───
    py::enum_<DistortionType>(m, "DistortionType")
        .value("SoftClip", DistortionType::SoftClip)
        .value("HardClip", DistortionType::HardClip)
        .value("Tanh", DistortionType::Tanh)
        .value("Foldback", DistortionType::Foldback)
        .value("BitCrush", DistortionType::BitCrush)
        .value("Tube", DistortionType::Tube);

    py::class_<Distortion>(m, "Distortion")
        .def(py::init<>())
        .def("set_params", &Distortion::set_params)
        .def("reset", &Distortion::reset)
        .def("process", &Distortion::process)
        .def("process_array", [](Distortion &self, py::array_t<float> data)
             {
            auto buf = data.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < buf.shape(0); i++) {
                buf(i) = self.process(buf(i));
            } });

    // ─── Compressor ───
    py::class_<Compressor>(m, "Compressor")
        .def(py::init<>())
        .def("set_params", &Compressor::set_params)
        .def("reset", &Compressor::reset)
        .def("process", &Compressor::process)
        .def("get_gain_reduction_db", &Compressor::get_gain_reduction_db)
        .def("process_array", [](Compressor &self, py::array_t<float> data)
             {
            auto buf = data.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < buf.shape(0); i++) {
                buf(i) = self.process(buf(i));
            } });

    // ─── Delay ───
    py::class_<StereoDelay>(m, "StereoDelay")
        .def(py::init<>())
        .def("init", &StereoDelay::init)
        .def("set_params", &StereoDelay::set_params)
        .def("reset", &StereoDelay::reset);

    // ─── Reverb ───
    py::class_<SchroederReverb>(m, "SchroederReverb")
        .def(py::init<>())
        .def("init", &SchroederReverb::init)
        .def("set_params", &SchroederReverb::set_params)
        .def("reset", &SchroederReverb::reset)
        .def("process_mono", &SchroederReverb::process_mono)
        .def("process_mono_array", [](SchroederReverb &self, py::array_t<float> data)
             {
            auto buf = data.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < buf.shape(0); i++) {
                buf(i) = self.process_mono(buf(i));
            } });

    // ─── Chorus ───
    py::class_<Chorus>(m, "Chorus")
        .def(py::init<>())
        .def("init", &Chorus::init)
        .def("set_params", &Chorus::set_params)
        .def("reset", &Chorus::reset);

    // ─── Limiter ───
    py::class_<Limiter>(m, "Limiter")
        .def(py::init<>())
        .def("init", &Limiter::init)
        .def("set_params", &Limiter::set_params)
        .def("reset", &Limiter::reset)
        .def("process", &Limiter::process)
        .def("process_array", [](Limiter &self, py::array_t<float> data)
             {
            auto buf = data.mutable_unchecked<1>();
            for (py::ssize_t i = 0; i < buf.shape(0); i++) {
                buf(i) = self.process(buf(i));
            } });
}
