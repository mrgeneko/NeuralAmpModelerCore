#pragma once

// ParametricWaveNet: a WaveNet variant conditioned on external knob values via FiLM.
//
// ARCHITECTURE OVERVIEW
// =====================
// The model always uses the A2 shape: one layer array of exactly 23 layers, with
// channel count 3 (nano) or 8 (standard) read from config["layers"]. The only
// structural difference from a standard A2 model is that each layer contains an
// active activation_pre_film FiLM module whose condition input comes from knob
// values — not from the audio signal.
//
// Per-layer computation:
//   audio → rechannel(1→C) → dilated_conv(C→C) ──┐
//   audio → input_mixin(1→C) ────────────────────── sum → FiLM(knobs) → leaky_relu → l1x1 → residual
//                                                         ↑
//                                                    knobs (6 floats, constant per buffer)
//
// TWO-CONDITION DESIGN
// ====================
// The upstream WaveNet architecture uses a single "condition" matrix for BOTH
// input_mixin (audio, 1×N) and all FiLM modules (knobs, num_params×N). Since
// these have different row counts, they cannot share a matrix. The solution:
//
//   _condition_output  (1 × N)          — audio; goes to input_mixin in each layer
//   _film_condition    (num_params × N) — knobs; goes to FiLM modules via _get_film_condition()
//
// WaveNet::process() calls _get_film_condition() and passes the result as the
// film_condition pointer to LayerArray::Process → Layer::Process → FiLM::Process_.
//
// WEIGHT LAYOUT (must match training code exactly)
// =================================================
// Per layer (23 layers total), weights are consumed in this order by Layer::set_weights_:
//   conv.weight  conv.bias  mixin.weight  l1x1.weight  l1x1.bias
//   film.net.weight  film.net.bias
//
// After all layers, consumed by LayerArray::set_weights_ and WaveNet::set_weights_:
//   head_rechannel.weight  head_rechannel.bias  head_scale
//
// No special weight loading code is needed — this layout matches the existing
// Layer::set_weights_ order (conv → mixin → l1x1 → activation_pre_film).
//
// THREAD SAFETY
// =============
// SetKnobValues() is called from the UI/main thread; _process_condition() reads
// the knob values on the audio thread. Communication uses std::atomic<float> per
// knob with relaxed ordering — lock-free on all Apple Silicon targets. A torn read
// of a single float during a concurrent write produces at worst a one-buffer glitch.

#include "model.h"
#include "../model_config.h"
#include "a2_fast.h"
#include "json.hpp"

#include <array>
#include <atomic>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace nam
{
namespace wavenet
{

/// \brief Per-knob metadata from the "parameters" array in the parametric JSON block.
struct ParametricParam
{
  std::string name;
  float min_val;
  float max_val;
  float default_val;
};

/// \brief WaveNet conditioned on external knob values via per-layer FiLM modules.
class ParametricWaveNet : public WaveNet
{
public:
  /// Maximum knob count this implementation supports.
  static constexpr int kMaxParams = 16;

  /// \param in_channels        Always 1 (mono guitar input).
  /// \param layer_array_params One entry, A2-shaped: condition_size=1 (audio for input_mixin),
  ///                           film_condition_size=num_params (knob count for FiLM).
  /// \param head_scale         Scalar applied to the final head output.
  /// \param weights            Full weight vector consumed on construction.
  /// \param expected_sample_rate  Sample rate the model was trained at (-1 if unknown).
  /// \param num_params         Number of knobs (= condition_size from the parametric block).
  /// \param param_defs         Per-knob metadata (name/min/max/default).
  ParametricWaveNet(int in_channels, const std::vector<LayerArrayParams>& layer_array_params,
                    float head_scale, std::vector<float> weights, double expected_sample_rate,
                    int num_params, std::vector<ParametricParam> param_defs);

  /// \brief Update knob values. May be called from any thread while process() runs.
  void SetKnobValues(const std::vector<float>& values);

  const std::vector<ParametricParam>& GetParameterDefs() const { return _param_defs; }
  int GetNumParams() const { return _num_params; }

  // Public override so ContainerModel and prewarm() both go through the DC blocker.
  void process(NAM_SAMPLE** inputs, NAM_SAMPLE** outputs, int num_frames) override;

  // Recomputes DC blocker coefficient for the new sample rate and resets blocker state.
  // prewarm() called after Reset() naturally drives the blocker to steady state.
  void Reset(double sampleRate, int maxBufferSize) override;

protected:
  // Returns 1 so that WaveNet::SetMaxBufferSize sizes _condition_input and _condition_output to 1×N.
  // _film_condition is resized separately in our SetMaxBufferSize override.
  int _get_condition_dim() const override { return 1; }

  // After WaveNet::SetMaxBufferSize (which sizes _condition_input/_condition_output to 1×N),
  // resize _film_condition to num_params×N.
  void SetMaxBufferSize(int maxBufferSize) override;

  // Step 1: call WaveNet::_process_condition to copy audio → _condition_output (for input_mixin).
  // Step 2: broadcast current knob values → _film_condition (for FiLM modules).
  void _process_condition(int num_frames) override;

  // Returns &_film_condition so WaveNet::process() passes knob values to every LayerArray::Process.
  const Eigen::MatrixXf* _get_film_condition() const override { return &_film_condition; }

private:
  int _num_params;
  std::vector<ParametricParam> _param_defs;

  // Knob condition matrix: _num_params rows × maxBufferSize columns.
  // Each column is identical — knobs are constant across all frames in a buffer.
  Eigen::MatrixXf _film_condition;

  // Atomic storage for knob values. UI thread writes, audio thread reads.
  std::array<std::atomic<float>, kMaxParams> _knob_atoms{};

  // DC blocker state (first-order IIR high-pass, fc ≈ 20 Hz).
  // The FiLM shift terms produce a significant DC offset at any trained knob position
  // (~17 dBFS below clipping). This blocker removes it without affecting audible
  // guitar content. prewarm() calls process() via virtual dispatch, so the blocker
  // converges to steady state before real audio flows — no click on first buffer.
  float _dc_alpha = 0.99738f; // 1 - 2π·20/48000; recomputed in Reset()
  float _dc_x_prev = 0.0f;
  float _dc_y_prev = 0.0f;
};

/// \brief ModelConfig produced by create_parametric_config(), used by the registry to
///        defer construction until weights and sample rate are known.
struct ParametricWaveNetConfig : public ModelConfig
{
  int in_channels;
  std::vector<LayerArrayParams> layer_array_params;
  float head_scale;
  int num_params;
  std::vector<ParametricParam> param_defs;

  ParametricWaveNetConfig() = default;
  ParametricWaveNetConfig(ParametricWaveNetConfig&&) = default;
  ParametricWaveNetConfig& operator=(ParametricWaveNetConfig&&) = default;
  ParametricWaveNetConfig(const ParametricWaveNetConfig&) = delete;
  ParametricWaveNetConfig& operator=(const ParametricWaveNetConfig&) = delete;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

/// \brief Factory registered under "ParametricWaveNet" in ConfigParserRegistry.
///
/// Called by SlimmableContainer (and any other dispatcher) when it encounters a
/// submodel with architecture == "ParametricWaveNet". Reads:
///   config["layers"]           — int, channel count (3 = nano, 8 = standard)
///   config["head_scale"]       — float
///   config["parametric"]       — object:
///     ["condition_size"]       — int, number of knobs
///     ["parameters"]           — array of {name, min, max, default}  (keys "min"/"max")
std::unique_ptr<ModelConfig> create_parametric_config(const nlohmann::json& config, double sampleRate);

} // namespace wavenet
} // namespace nam
