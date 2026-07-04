// Tests for ParametricWaveNet — the knob-conditioned WaveNet variant.
//
// Coverage:
//   1. Config parsing: create_parametric_config() JSON → ParametricWaveNetConfig
//   2. Construction: ParametricWaveNet with custom (non-A2) tiny topology
//   3. Processing: finite output, zero input, different buffer sizes
//   4. SetKnobValues: different knob values produce different outputs
//   5. Accessor methods: GetNumParams, GetParameterDefs
//   6. Registry: "ParametricWaveNet" registered in ConfigParserRegistry
//   7. End-to-end factory: create_parametric_config → create() → process()
//   8. Standard WaveNet regression: film_condition=nullptr path unaffected

#include <cassert>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "NAM/model_config.h"
#include "NAM/wavenet/a2_fast.h"
#include "NAM/wavenet/model.h"
#include "NAM/wavenet/parametric_wavenet.h"

namespace test_parametric_wavenet
{

// =============================================================================
// Helpers for building a tiny ParametricWaveNet (NOT the full A2 topology)
// =============================================================================
//
// Configuration used throughout these helpers:
//   C=1 channel, 1 layer, kernel_size=1, dilation=1
//   condition_size=1 (audio), film_condition_size=num_params (knobs)
//   layer1x1 active (groups=1), head1x1 inactive
//   head_kernel_size=1, head_bias=false
//   Only activation_pre_film active (shift=true, groups=1)
//
// This keeps the weight count small and the math easy to reason about.

static nam::wavenet::LayerArrayParams make_tiny_params(const int num_params)
{
  const nam::wavenet::_FiLMParams off{false, false};
  const nam::wavenet::_FiLMParams activation_pre_film{true, true, 1}; // shift=true → scale+shift

  const auto leaky = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::LeakyReLU);

  return nam::wavenet::LayerArrayParams(
    /*input_size=*/       1,
    /*condition_size=*/   1,
    /*head_size=*/        1,
    /*head_kernel_size=*/ 1,
    /*channels=*/         1,
    /*bottleneck=*/       1,
    /*kernel_sizes=*/     {1},
    /*dilations=*/        {1},
    /*activation_configs=*/ {leaky},
    /*gating_modes=*/     {nam::wavenet::GatingMode::NONE},
    /*head_bias=*/        false,
    /*groups_input=*/     1,
    /*groups_input_mixin=*/ 1,
    nam::wavenet::Layer1x1Params{true, 1},
    nam::wavenet::Head1x1Params{false, 1, 1},
    /*secondary_activation_configs=*/ {nam::activations::ActivationConfig{}},
    off, off, off, off,
    activation_pre_film, // THE active FiLM; condition_dim = num_params
    off, off, off,
    /*film_condition_size=*/ num_params);
}

// Weight count for the tiny config above.
//
// Layout (Layer::set_weights_ order):
//   rechannel    : Conv1x1(1, 1, no bias)          = 1
//   layer[0]     : conv  Conv1D(1,1,K=1,bias)      = 1*1*1 + 1 = 2
//                  mixin Conv1x1(1,1,no bias)       = 1
//                  l1x1  Conv1x1(1,1,bias)          = 1*1 + 1 = 2
//                  film  Conv1x1(np,2,bias=true)    = np*2 + 2
//   head_rechannel: Conv1D(1,1,K=1,no bias)         = 1
//   head_scale:                                      = 1
static int count_tiny_weights(const int num_params)
{
  const int film_weights = num_params * 2 + 2; // Conv1x1(np, 2*1=2, bias)
  return 1 + (2 + 1 + 2 + film_weights) + 1 + 1;
}

static std::unique_ptr<nam::wavenet::ParametricWaveNet> make_tiny_pwn(const int num_params,
                                                                       const float weight_value = 0.5f)
{
  const int n = count_tiny_weights(num_params);
  std::vector<float> weights(n, weight_value);

  std::vector<nam::wavenet::LayerArrayParams> laps;
  laps.push_back(make_tiny_params(num_params));

  std::vector<nam::wavenet::ParametricParam> param_defs;
  for (int i = 0; i < num_params; ++i)
    param_defs.push_back({"knob" + std::to_string(i), 0.0f, 1.0f, 0.5f});

  return std::make_unique<nam::wavenet::ParametricWaveNet>(
    /*in_channels=*/ 1, laps, /*head_scale=*/ 0.5f, std::move(weights), /*expected_sample_rate=*/ 48000.0,
    num_params, std::move(param_defs));
}

// Process num_buffers of buffer_size frames; return the last output frame value.
static std::vector<NAM_SAMPLE> process_and_capture(nam::wavenet::ParametricWaveNet* pwn, int buffer_size,
                                                    float input_val)
{
  pwn->Reset(48000.0, buffer_size);
  std::vector<NAM_SAMPLE> input(buffer_size, input_val);
  std::vector<NAM_SAMPLE> output(buffer_size, 0.0f);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = output.data();
  pwn->process(&in_ptr, &out_ptr, buffer_size);
  return output;
}

// =============================================================================
// 1. Config parsing
// =============================================================================

void test_config_parse_valid_json()
{
  nlohmann::json cfg;
  cfg["layers"] = 3; // nano channel count
  cfg["head_scale"] = 1.0f;
  cfg["parametric"]["condition_size"] = 2;
  cfg["parametric"]["parameters"] = {{{"name", "drive"}, {"minimum", 0.0f}, {"maximum", 1.0f}, {"default", 0.5f}},
                                      {{"name", "tone"}, {"minimum", 0.0f}, {"maximum", 1.0f}, {"default", 0.5f}}};

  auto config = nam::wavenet::create_parametric_config(cfg, 48000.0);
  assert(config != nullptr);
  auto* pwn_cfg = dynamic_cast<nam::wavenet::ParametricWaveNetConfig*>(config.get());
  assert(pwn_cfg != nullptr);
  assert(pwn_cfg->num_params == 2);
  assert(pwn_cfg->param_defs.size() == 2);
  assert(pwn_cfg->param_defs[0].name == "drive");
  assert(pwn_cfg->param_defs[1].name == "tone");
  assert(pwn_cfg->param_defs[0].min_val == 0.0f);
  assert(pwn_cfg->param_defs[0].max_val == 1.0f);
  assert(pwn_cfg->param_defs[0].default_val == 0.5f);
  assert(pwn_cfg->in_channels == 1);
  assert(pwn_cfg->head_scale == 1.0f);
}

void test_config_parse_missing_layers_throws()
{
  nlohmann::json cfg;
  cfg["head_scale"] = 1.0f;
  cfg["parametric"]["condition_size"] = 2;

  bool threw = false;
  try
  {
    nam::wavenet::create_parametric_config(cfg, 48000.0);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

void test_config_parse_invalid_channel_count_throws()
{
  // Only 3 (nano) and 8 (standard) are valid
  nlohmann::json cfg;
  cfg["layers"] = 5; // invalid
  cfg["head_scale"] = 1.0f;
  cfg["parametric"]["condition_size"] = 2;

  bool threw = false;
  try
  {
    nam::wavenet::create_parametric_config(cfg, 48000.0);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

void test_config_parse_missing_head_scale_throws()
{
  nlohmann::json cfg;
  cfg["layers"] = 3;
  cfg["parametric"]["condition_size"] = 2;

  bool threw = false;
  try
  {
    nam::wavenet::create_parametric_config(cfg, 48000.0);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

void test_config_parse_missing_parametric_block_throws()
{
  nlohmann::json cfg;
  cfg["layers"] = 3;
  cfg["head_scale"] = 1.0f;

  bool threw = false;
  try
  {
    nam::wavenet::create_parametric_config(cfg, 48000.0);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

void test_config_parse_missing_condition_size_throws()
{
  nlohmann::json cfg;
  cfg["layers"] = 3;
  cfg["head_scale"] = 1.0f;
  cfg["parametric"]["parameters"] = nlohmann::json::array();

  bool threw = false;
  try
  {
    nam::wavenet::create_parametric_config(cfg, 48000.0);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

void test_config_parse_condition_size_too_large_throws()
{
  nlohmann::json cfg;
  cfg["layers"] = 3;
  cfg["head_scale"] = 1.0f;
  cfg["parametric"]["condition_size"] = nam::wavenet::ParametricWaveNet::kMaxParams + 1;

  bool threw = false;
  try
  {
    nam::wavenet::create_parametric_config(cfg, 48000.0);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

void test_config_parse_standard_channel_count()
{
  // 8 (standard) must also parse correctly
  nlohmann::json cfg;
  cfg["layers"] = 8;
  cfg["head_scale"] = 0.5f;
  cfg["parametric"]["condition_size"] = 1;

  auto config = nam::wavenet::create_parametric_config(cfg, 48000.0);
  assert(config != nullptr);
  auto* pwn_cfg = dynamic_cast<nam::wavenet::ParametricWaveNetConfig*>(config.get());
  assert(pwn_cfg != nullptr);
  // Verify channel count propagated: the one LayerArrayParams should have channels==8
  assert(pwn_cfg->layer_array_params.size() == 1);
  assert(pwn_cfg->layer_array_params[0].channels == 8);
}

void test_config_parse_param_defs_optional()
{
  // "parameters" array is optional — missing it should not throw
  nlohmann::json cfg;
  cfg["layers"] = 3;
  cfg["head_scale"] = 1.0f;
  cfg["parametric"]["condition_size"] = 3;
  // no "parameters" key

  auto config = nam::wavenet::create_parametric_config(cfg, 48000.0);
  assert(config != nullptr);
  auto* pwn_cfg = dynamic_cast<nam::wavenet::ParametricWaveNetConfig*>(config.get());
  assert(pwn_cfg != nullptr);
  assert(pwn_cfg->num_params == 3);
  assert(pwn_cfg->param_defs.empty()); // no defs provided
}

// =============================================================================
// 2. Construction
// =============================================================================

void test_construct_with_one_param()
{
  auto pwn = make_tiny_pwn(1);
  assert(pwn != nullptr);
  assert(pwn->GetNumParams() == 1);
}

void test_construct_with_multiple_params()
{
  auto pwn = make_tiny_pwn(6);
  assert(pwn != nullptr);
  assert(pwn->GetNumParams() == 6);
}

void test_get_parameter_defs()
{
  const int num_params = 3;
  auto pwn = make_tiny_pwn(num_params);
  const auto& defs = pwn->GetParameterDefs();
  assert((int)defs.size() == num_params);
  for (int i = 0; i < num_params; ++i)
  {
    assert(defs[i].name == "knob" + std::to_string(i));
    assert(defs[i].min_val == 0.0f);
    assert(defs[i].max_val == 1.0f);
    assert(defs[i].default_val == 0.5f);
  }
}

// =============================================================================
// 3. Processing: output finiteness and zero input
// =============================================================================

void test_processes_finite_output()
{
  auto pwn = make_tiny_pwn(2);
  const auto out = process_and_capture(pwn.get(), 64, 0.5f);
  for (auto v : out)
    assert(std::isfinite(v));
}

void test_processes_zero_input_finite()
{
  auto pwn = make_tiny_pwn(2);
  const auto out = process_and_capture(pwn.get(), 64, 0.0f);
  for (auto v : out)
    assert(std::isfinite(v));
}

void test_different_buffer_sizes()
{
  for (int buf : {16, 32, 64, 128, 256})
  {
    auto pwn = make_tiny_pwn(2);
    const auto out = process_and_capture(pwn.get(), buf, 0.3f);
    assert((int)out.size() == buf);
    for (auto v : out)
      assert(std::isfinite(v));
  }
}

// =============================================================================
// 4. SetKnobValues: different knob values produce measurably different outputs
// =============================================================================

void test_set_knob_values_changes_output()
{
  // Two separate instances so they start from identical zero state
  auto pwn_lo = make_tiny_pwn(2, 0.5f);
  auto pwn_hi = make_tiny_pwn(2, 0.5f);

  // Knobs near 0 vs near 1 — the active FiLM applies scale+shift = knob*0.5+knob*0.5+0.5 to _z
  pwn_lo->SetKnobValues({0.1f, 0.1f});
  pwn_hi->SetKnobValues({0.9f, 0.9f});

  const int buffer_size = 64;
  const float input_val = 0.5f;

  const auto out_lo = process_and_capture(pwn_lo.get(), buffer_size, input_val);
  const auto out_hi = process_and_capture(pwn_hi.get(), buffer_size, input_val);

  // FiLM scale+shift differ between lo and hi knobs → outputs must diverge
  bool any_different = false;
  for (int i = 0; i < buffer_size; ++i)
  {
    if (std::abs(out_lo[i] - out_hi[i]) > 1e-6f)
    {
      any_different = true;
      break;
    }
  }
  assert(any_different);
}

void test_set_knob_values_excess_values_no_crash()
{
  // Passing more values than num_params must not crash or corrupt state
  auto pwn = make_tiny_pwn(2);
  pwn->SetKnobValues({0.1f, 0.2f, 0.3f, 0.4f, 0.5f}); // 5 > num_params=2 → only first 2 used
  const auto out = process_and_capture(pwn.get(), 64, 0.5f);
  for (auto v : out)
    assert(std::isfinite(v));
}

void test_set_knob_values_empty_no_crash()
{
  auto pwn = make_tiny_pwn(2);
  pwn->SetKnobValues({}); // no values → existing values (0.5 default) unchanged
  const auto out = process_and_capture(pwn.get(), 64, 0.5f);
  for (auto v : out)
    assert(std::isfinite(v));
}

void test_default_knobs_midpoint()
{
  // Default knob value is 0.5; applying a full-range SetKnobValues(all 0.5) should produce
  // the same output as no SetKnobValues call (since constructor initializes to 0.5).
  auto pwn_default = make_tiny_pwn(3);
  auto pwn_explicit = make_tiny_pwn(3);
  pwn_explicit->SetKnobValues({0.5f, 0.5f, 0.5f});

  const auto out_default = process_and_capture(pwn_default.get(), 64, 0.3f);
  const auto out_explicit = process_and_capture(pwn_explicit.get(), 64, 0.3f);

  for (size_t i = 0; i < out_default.size(); ++i)
    assert(std::abs(out_default[i] - out_explicit[i]) < 1e-6f);
}

// =============================================================================
// 5. Registry
// =============================================================================

void test_registry_has_parametric_wavenet()
{
  // The auto-registration at namespace scope in parametric_wavenet.cpp registers
  // "ParametricWaveNet" before main() runs.
  assert(nam::ConfigParserRegistry::instance().has("ParametricWaveNet"));
}

// =============================================================================
// 6. End-to-end factory: create_parametric_config → create() → process()
//
// Uses A2 nano (C=3, 23 layers) with num_params=1 and zero weights.
// This exercises the full A2 topology assembly code in create_parametric_config.
//
// Weight count derivation for C=3, num_params=1 A2 nano:
//   kKernelSizes = 21×K=6 + 2×K=15
//   Per K=6 layer: conv(3*3*6+3=57) + mixin(3) + l1x1(3*3+3=12) + film(1*6+6=12) = 84
//   Per K=15 layer: conv(3*3*15+3=138) + mixin(3) + l1x1(12) + film(12) = 165
//   All 23 layers:  21*84 + 2*165 = 1764 + 330 = 2094
//   rechannel: Conv1x1(1,3,no bias) = 3
//   head_rechannel: Conv1D(3,1,K=16,bias=true) = 3*16+1 = 49
//   head_scale: 1
//   Total: 3 + 2094 + 49 + 1 = 2147
// =============================================================================

static constexpr int kA2NanoWeightCount_1param = 2147;

void test_factory_creates_and_processes_nano()
{
  nlohmann::json cfg;
  cfg["layers"] = 3;
  cfg["head_scale"] = 1.0f;
  cfg["parametric"]["condition_size"] = 1;
  cfg["parametric"]["parameters"] = {{{"name", "drive"}, {"minimum", 0.0f}, {"maximum", 1.0f}, {"default", 0.5f}}};

  auto config = nam::wavenet::create_parametric_config(cfg, 48000.0);
  assert(config != nullptr);

  std::vector<float> weights(kA2NanoWeightCount_1param, 0.1f); // small non-zero weights
  auto dsp = config->create(std::move(weights), 48000.0);
  assert(dsp != nullptr);

  auto* pwn = dynamic_cast<nam::wavenet::ParametricWaveNet*>(dsp.get());
  assert(pwn != nullptr);
  assert(pwn->GetNumParams() == 1);

  // Process audio and verify finite output
  const int buffer_size = 64;
  pwn->Reset(48000.0, buffer_size);
  std::vector<NAM_SAMPLE> input(buffer_size, 0.1f);
  std::vector<NAM_SAMPLE> output(buffer_size, 0.0f);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = output.data();
  pwn->process(&in_ptr, &out_ptr, buffer_size);

  for (auto v : output)
    assert(std::isfinite(v));
}

void test_factory_nano_knobs_change_output()
{
  nlohmann::json cfg;
  cfg["layers"] = 3;
  cfg["head_scale"] = 1.0f;
  cfg["parametric"]["condition_size"] = 1;

  auto config_lo = nam::wavenet::create_parametric_config(cfg, 48000.0);
  auto config_hi = nam::wavenet::create_parametric_config(cfg, 48000.0);

  std::vector<float> weights_lo(kA2NanoWeightCount_1param, 0.1f);
  std::vector<float> weights_hi(kA2NanoWeightCount_1param, 0.1f);

  auto dsp_lo = config_lo->create(std::move(weights_lo), 48000.0);
  auto dsp_hi = config_hi->create(std::move(weights_hi), 48000.0);

  auto* pwn_lo = dynamic_cast<nam::wavenet::ParametricWaveNet*>(dsp_lo.get());
  auto* pwn_hi = dynamic_cast<nam::wavenet::ParametricWaveNet*>(dsp_hi.get());
  assert(pwn_lo && pwn_hi);

  pwn_lo->SetKnobValues({0.0f});
  pwn_hi->SetKnobValues({1.0f});

  const int buffer_size = 64;
  pwn_lo->Reset(48000.0, buffer_size);
  pwn_hi->Reset(48000.0, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.1f);
  std::vector<NAM_SAMPLE> out_lo(buffer_size), out_hi(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* p;

  p = out_lo.data(); pwn_lo->process(&in_ptr, &p, buffer_size);
  p = out_hi.data(); pwn_hi->process(&in_ptr, &p, buffer_size);

  bool any_different = false;
  for (int i = 0; i < buffer_size; ++i)
    if (std::abs(out_lo[i] - out_hi[i]) > 1e-6f) { any_different = true; break; }
  assert(any_different);
}

// =============================================================================
// 7. Standard WaveNet regression: film_condition=nullptr path unaffected
//
// Builds the same tiny 1-layer WaveNet WITHOUT activation_pre_film active, using the
// standard WaveNet class directly. Verifies it still produces finite output after the
// film_condition=nullptr fallback logic was added to Layer::Process/LayerArray::Process.
// =============================================================================

void test_standard_wavenet_unaffected()
{
  const auto relu = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::LeakyReLU);
  const nam::wavenet::_FiLMParams off{false, false};
  nam::wavenet::LayerArrayParams lap(
    1, 1, 1, 1, 1, 1, {1}, {1}, {relu}, {nam::wavenet::GatingMode::NONE}, false, 1, 1,
    nam::wavenet::Layer1x1Params{true, 1}, nam::wavenet::Head1x1Params{false, 1, 1}, {nam::activations::ActivationConfig{}},
    off, off, off, off, off, off, off, off);

  // Weight count for standard WaveNet (no FiLM): 1 + (2+1+2) + 1 + 1 = 8
  std::vector<float> weights(8, 0.5f);

  std::vector<nam::wavenet::LayerArrayParams> laps;
  laps.push_back(std::move(lap));

  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    1, laps, 0.5f, /*with_head=*/false, std::nullopt, weights, nullptr, 48000.0);

  const int buffer_size = 64;
  wavenet->Reset(48000.0, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.4f);
  std::vector<NAM_SAMPLE> output(buffer_size, 0.0f);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = output.data();
  wavenet->process(&in_ptr, &out_ptr, buffer_size);

  for (auto v : output)
    assert(std::isfinite(v));
}

// Test: DC blocker reduces the model's steady-state DC during sustained silence.
// We verify that after many silent frames the output is MUCH smaller than it was
// at the very start (before the blocker had time to converge). This proves the
// blocker is working without requiring an absolute threshold that depends on the
// arbitrary weight values of the synthetic test model.
void test_dc_blocker_removes_dc_at_silence()
{
  const int np = 2;
  auto pwn = make_tiny_pwn(np, 1.0f);
  const int buf = 64;
  pwn->Reset(48000.0, buf);
  // Do NOT call prewarm here so we measure from a clean ring-buffer state.

  // First pass: the blocker starts at zero and sees the model's full DC output.
  std::vector<NAM_SAMPLE> silence(buf, 0.0f);
  std::vector<NAM_SAMPLE> out_first(buf, 0.0f);
  {
    NAM_SAMPLE* sil_ptr = silence.data();
    NAM_SAMPLE* out_ptr = out_first.data();
    pwn->process(&sil_ptr, &out_ptr, buf);
  }
  float peak_first = 0.0f;
  for (auto v : out_first) peak_first = std::max(peak_first, std::abs((float)v));

  // Run many more silent passes until the blocker has had >> 5× time-constant worth of
  // samples (time-constant = 1/(2π·20) · 48000 ≈ 382 samples; use 6000 to be safe).
  std::vector<NAM_SAMPLE> out_last(buf, 0.0f);
  const int total_passes = 6000 / buf + 1;
  for (int p = 0; p < total_passes; ++p)
  {
    NAM_SAMPLE* sil_ptr = silence.data();
    NAM_SAMPLE* out_ptr = out_last.data();
    pwn->process(&sil_ptr, &out_ptr, buf);
  }
  float peak_last = 0.0f;
  for (auto v : out_last) peak_last = std::max(peak_last, std::abs((float)v));

  // The blocker must have attenuated DC by at least 99% (factor of 100).
  assert(peak_first > 0.0f); // model must produce something initially
  assert(peak_last < peak_first * 0.01f);
}

// Test: DC blocker does not kill AC signal — output still changes when input changes.
// With silence the blocker converges to ~0; then a sudden non-zero input must
// produce a measurably non-zero output (proving the AC path is intact).
void test_dc_blocker_passes_ac_signal()
{
  const int np = 2;
  auto pwn = make_tiny_pwn(np, 1.0f);
  const int buf = 64;
  pwn->Reset(48000.0, buf);
  pwn->prewarm();

  // Run a non-zero input buffer.
  std::vector<NAM_SAMPLE> input(buf, 0.3f);
  std::vector<NAM_SAMPLE> out_signal(buf, 0.0f);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = out_signal.data();
  pwn->process(&in_ptr, &out_ptr, buf);

  // Run a silence buffer.
  std::vector<NAM_SAMPLE> silence(buf, 0.0f);
  std::vector<NAM_SAMPLE> out_silence(buf, 0.0f);
  NAM_SAMPLE* sil_ptr = silence.data();
  NAM_SAMPLE* out_sil_ptr = out_silence.data();
  pwn->process(&sil_ptr, &out_sil_ptr, buf);

  // Confirm the two outputs differ — AC signal is not zeroed by the blocker.
  float diff = 0.0f;
  for (int i = 0; i < buf; ++i)
    diff += std::abs((float)out_signal[i] - (float)out_silence[i]);
  assert(diff > 0.001f); // outputs should be measurably different
}

} // namespace test_parametric_wavenet
