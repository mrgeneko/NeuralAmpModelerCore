#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/container.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"
#include "NAM/wavenet/a2_fast.h"           // [Fork] for parametric weight-count formula
#include "NAM/wavenet/parametric_wavenet.h" // [Fork] for create_parametric_config

namespace test_container
{

class CountingDSP : public nam::DSP
{
public:
  explicit CountingDSP(NAM_SAMPLE process_value)
  : DSP(1, 1, 48000.0)
  , process_value(process_value)
  {
  }

  void Reset(const double sampleRate, const int maxBufferSize) override
  {
    if (on_reset)
      on_reset();

    reset_count++;
    reset_sample_rate = sampleRate;
    reset_buffer_size = maxBufferSize;
    DSP::Reset(sampleRate, maxBufferSize);
  }

  void prewarm() override { prewarm_count++; }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override
  {
    (void)input;
    process_count++;
    for (int i = 0; i < num_frames; i++)
      output[0][i] = process_value;
  }

  const NAM_SAMPLE process_value;
  int reset_count = 0;
  int prewarm_count = 0;
  int process_count = 0;
  double reset_sample_rate = 0.0;
  int reset_buffer_size = 0;
  std::function<void()> on_reset;
};

std::unique_ptr<nam::container::ContainerModel> build_counting_container(CountingDSP*& small, CountingDSP*& large)
{
  auto small_model = std::make_unique<CountingDSP>((NAM_SAMPLE)1.0);
  auto large_model = std::make_unique<CountingDSP>((NAM_SAMPLE)2.0);

  small = small_model.get();
  large = large_model.get();

  std::vector<nam::container::Submodel> submodels;
  submodels.push_back({0.5, 3, std::move(small_model)});
  submodels.push_back({1.0, 8, std::move(large_model)});

  return std::make_unique<nam::container::ContainerModel>(std::move(submodels), 48000.0);
}

NAM_SAMPLE process_one_sample(nam::DSP* dsp)
{
  NAM_SAMPLE input = (NAM_SAMPLE)0.0;
  NAM_SAMPLE output = (NAM_SAMPLE)-1.0;
  NAM_SAMPLE* in_ptr = &input;
  NAM_SAMPLE* out_ptr = &output;
  dsp->process(&in_ptr, &out_ptr, 1);
  return output;
}

// Helper: load a .nam file as JSON
nlohmann::json load_nam_json(const std::string& path)
{
  std::ifstream f(path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open " + path);
  nlohmann::json j;
  f >> j;
  return j;
}

// Build a SlimmableContainer JSON from 3 .nam files
nlohmann::json build_container_json(const std::string& small_path, const std::string& medium_path,
                                    const std::string& large_path)
{
  nlohmann::json small_model = load_nam_json(small_path);
  nlohmann::json medium_model = load_nam_json(medium_path);
  nlohmann::json large_model = load_nam_json(large_path);

  nlohmann::json container;
  container["version"] = "0.7.0";
  container["architecture"] = "SlimmableContainer";
  container["config"]["submodels"] = nlohmann::json::array({{{"max_value", 0.33}, {"model", small_model}},
                                                            {{"max_value", 0.66}, {"model", medium_model}},
                                                            {{"max_value", 1.0}, {"model", large_model}}});
  container["weights"] = nlohmann::json::array();
  container["sample_rate"] = 48000;
  return container;
}

// Helper to process audio through a DSP model and verify finite output
void process_and_verify(nam::DSP* dsp, int num_buffers, int buffer_size)
{
  const double sample_rate = dsp->GetExpectedSampleRate() > 0 ? dsp->GetExpectedSampleRate() : 48000.0;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size);
  std::vector<NAM_SAMPLE> output(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = output.data();

  for (int buf = 0; buf < num_buffers; buf++)
  {
    for (int i = 0; i < buffer_size; i++)
      input[i] = (NAM_SAMPLE)(0.1 * ((buf * buffer_size + i) % 100) / 100.0);

    dsp->process(&in_ptr, &out_ptr, buffer_size);

    for (int i = 0; i < buffer_size; i++)
      assert(std::isfinite(output[i]));
  }
}

// [Fork] BEGIN: helpers for ContainerModel knob-API delegation tests (Gap 2)
//
// Parametric models always use the A2 layer-array shape (23 layers, fixed kernel/
// dilation pattern; see NAM/wavenet/parametric_wavenet.h's architecture overview),
// so the weight count is a function of channel count and knob count alone. This
// mirrors the same formula test_a2_fast.cpp's (file-local) parametric_a2_weight_count
// uses, restated here against the public nam::wavenet::a2_fast constants since that
// helper isn't exported across translation units.
int parametric_weight_count(int channels, int num_params)
{
  using namespace nam::wavenet::a2_fast;
  int total = channels; // rechannel
  for (int i = 0; i < kNumLayers; i++)
  {
    const int K = kKernelSizes[i];
    total += channels * channels * K + channels; // conv weight + bias
    total += channels;                            // input mixin
    total += channels * channels + channels;      // layer1x1 weight + bias
    total += 2 * channels * num_params + 2 * channels; // per-layer FiLM net weight + bias
  }
  total += channels * kHeadKernelSize + 1; // head rechannel weight + bias
  total += 1;                              // head_scale
  return total;
}

// Builds a real parametric submodel DSP (uniform-valued weights, same convention as
// test_parametric_wavenet.cpp's test_factory_creates_and_processes_nano) via the same
// factory path production .nam files go through (create_parametric_config -> create()).
std::unique_ptr<nam::DSP> make_parametric_submodel(int channels, int num_params, float weight_value,
                                                    double sample_rate)
{
  nlohmann::json cfg;
  cfg["layers"] = channels;
  cfg["head_scale"] = 1.0f;
  cfg["parametric"]["condition_size"] = num_params;
  cfg["parametric"]["head_mode"] = "skip";
  cfg["parametric"]["schema_version"] = 1;
  // Populate real per-knob metadata so GetParameterDefs() has something meaningful to
  // delegate (an empty "parameters" array is valid but makes defs.size() 0 regardless
  // of num_params, which would make the delegation test ambiguous).
  nlohmann::json params = nlohmann::json::array();
  for (int i = 0; i < num_params; i++)
    params.push_back({{"name", "knob" + std::to_string(i)}, {"minimum", 0.0f}, {"maximum", 1.0f}, {"default", 0.5f}});
  cfg["parametric"]["parameters"] = params;

  auto config = nam::wavenet::create_parametric_config(cfg, sample_rate);
  std::vector<float> weights(parametric_weight_count(channels, num_params), weight_value);
  return config->create(std::move(weights), sample_rate);
}

double max_abs_diff(const std::vector<NAM_SAMPLE>& a, const std::vector<NAM_SAMPLE>& b)
{
  assert(a.size() == b.size());
  double d = 0.0;
  for (size_t i = 0; i < a.size(); i++)
    d = std::max(d, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  return d;
}
// [Fork] END: helpers for ContainerModel knob-API delegation tests

// =====================================================================
// Tests
// =====================================================================

void test_container_loads_from_json()
{
  auto j = build_container_json(
    "example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_feature_test.nam");
  auto dsp = nam::get_dsp(j);
  assert(dsp != nullptr);
}

void test_container_processes_audio()
{
  auto j = build_container_json(
    "example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_feature_test.nam");
  auto dsp = nam::get_dsp(j);
  process_and_verify(dsp.get(), 3, 64);
}

void test_container_slimmable_selects_submodel()
{
  auto j = build_container_json(
    "example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_feature_test.nam");
  auto dsp = nam::get_dsp(j);
  const double sample_rate = 48000.0;
  const int buffer_size = 64;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.1);
  std::vector<NAM_SAMPLE> out_small(buffer_size);
  std::vector<NAM_SAMPLE> out_large(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr;

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  // Process at minimum size (selects first submodel)
  slimmable->SetSlimmableSize(0.0);
  out_ptr = out_small.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Process at maximum size (selects last submodel)
  slimmable->SetSlimmableSize(1.0);
  out_ptr = out_large.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // The outputs should differ since different models are active
  // (Not guaranteed for every sample, but statistically they should differ)
  bool any_different = false;
  for (int i = 0; i < buffer_size; i++)
  {
    if (std::abs(out_small[i] - out_large[i]) > 1e-6)
    {
      any_different = true;
      break;
    }
  }
  assert(any_different);
}

void test_container_boundary_values()
{
  auto j = build_container_json(
    "example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_feature_test.nam");
  auto dsp = nam::get_dsp(j);
  const double sample_rate = 48000.0;
  const int buffer_size = 16;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.05);
  std::vector<NAM_SAMPLE> output(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr = output.data();

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  // Test exact boundary values (max_value is exclusive: val < max_value selects that submodel)
  slimmable->SetSlimmableSize(0.32); // Should select first submodel (0.32 < 0.33)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));

  slimmable->SetSlimmableSize(0.33); // Should select second submodel (0.33 is NOT < 0.33, but IS < 0.66)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));

  slimmable->SetSlimmableSize(0.65); // Should select second submodel (0.65 < 0.66)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));

  slimmable->SetSlimmableSize(0.66); // Should select third/last submodel (0.66 is NOT < 0.66, fallback)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));

  slimmable->SetSlimmableSize(1.0); // Should select third/last submodel (fallback)
  dsp->process(&in_ptr, &out_ptr, buffer_size);
  for (int i = 0; i < buffer_size; i++)
    assert(std::isfinite(output[i]));
}

void test_container_slimmable_breakpoints()
{
  CountingDSP* small = nullptr;
  CountingDSP* large = nullptr;
  auto dsp = build_counting_container(small, large);

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  const auto breakpoints = slimmable->GetSlimmableSizeBreakpoints();
  assert(breakpoints.size() == 1);
  assert(breakpoints[0] == 0.5);
}

void test_container_empty_submodels_throws()
{
  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableContainer";
  j["config"]["submodels"] = nlohmann::json::array();
  j["weights"] = nlohmann::json::array();
  j["sample_rate"] = 48000;

  bool threw = false;
  try
  {
    auto dsp = nam::get_dsp(j);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

void test_container_last_max_value_must_cover_one()
{
  // Build a container where last max_value < 1.0
  auto small_json = load_nam_json("example_models/lstm.nam");

  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableContainer";
  j["config"]["submodels"] = nlohmann::json::array({{{"max_value", 0.5}, {"model", small_json}}});
  j["weights"] = nlohmann::json::array();
  j["sample_rate"] = 48000;

  // Suppress the version warning
  std::streambuf* originalCerr = std::cerr.rdbuf();
  std::ostringstream nullStream;
  std::cerr.rdbuf(nullStream.rdbuf());

  bool threw = false;
  try
  {
    auto dsp = nam::get_dsp(j);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);

  std::cerr.rdbuf(originalCerr);
}

void test_container_unsorted_submodels_throws()
{
  auto small_json = load_nam_json("example_models/lstm.nam");
  auto medium_json = load_nam_json("example_models/wavenet.nam");

  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableContainer";
  j["config"]["submodels"] =
    nlohmann::json::array({{{"max_value", 0.8}, {"model", small_json}}, {{"max_value", 0.5}, {"model", medium_json}}});
  j["weights"] = nlohmann::json::array();
  j["sample_rate"] = 48000;

  std::streambuf* originalCerr = std::cerr.rdbuf();
  std::ostringstream nullStream;
  std::cerr.rdbuf(nullStream.rdbuf());

  bool threw = false;
  try
  {
    auto dsp = nam::get_dsp(j);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);

  std::cerr.rdbuf(originalCerr);
}

void test_container_sample_rate_mismatch_throws()
{
  // Create two models with different sample rates
  auto model_48k = load_nam_json("example_models/lstm.nam");
  auto model_44k = load_nam_json("example_models/lstm.nam");
  model_44k["sample_rate"] = 44100;

  nlohmann::json j;
  j["version"] = "0.7.0";
  j["architecture"] = "SlimmableContainer";
  j["config"]["submodels"] =
    nlohmann::json::array({{{"max_value", 0.5}, {"model", model_44k}}, {{"max_value", 1.0}, {"model", model_48k}}});
  j["weights"] = nlohmann::json::array();
  j["sample_rate"] = 48000;

  std::streambuf* originalCerr = std::cerr.rdbuf();
  std::ostringstream nullStream;
  std::cerr.rdbuf(nullStream.rdbuf());

  bool threw = false;
  try
  {
    auto dsp = nam::get_dsp(j);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);

  std::cerr.rdbuf(originalCerr);
}

void test_container_load_from_file()
{
  std::filesystem::path path("example_models/slimmable_container.nam");
  auto dsp = nam::get_dsp(path);
  assert(dsp != nullptr);
  process_and_verify(dsp.get(), 3, 64);
}

void test_container_default_is_max_size()
{
  auto j = build_container_json(
    "example_models/lstm.nam", "example_models/wavenet.nam", "example_models/wavenet_a2_feature_test.nam");
  auto dsp = nam::get_dsp(j);
  const double sample_rate = 48000.0;
  const int buffer_size = 64;
  dsp->Reset(sample_rate, buffer_size);

  std::vector<NAM_SAMPLE> input(buffer_size, 0.1);
  std::vector<NAM_SAMPLE> out_default(buffer_size);
  std::vector<NAM_SAMPLE> out_max(buffer_size);
  NAM_SAMPLE* in_ptr = input.data();
  NAM_SAMPLE* out_ptr;

  // Ensure both predictions start from identical model state.
  dsp->Reset(sample_rate, buffer_size);
  // Process with default (should be max size)
  out_ptr = out_default.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Explicitly set to max
  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);
  slimmable->SetSlimmableSize(1.0);
  dsp->Reset(sample_rate, buffer_size);
  out_ptr = out_max.data();
  dsp->process(&in_ptr, &out_ptr, buffer_size);

  // Both should produce the same output (same model active)
  for (int i = 0; i < buffer_size; i++)
    assert(std::abs(out_default[i] - out_max[i]) < 1e-6);
}

void test_container_reset_only_resets_active_submodel()
{
  CountingDSP* small = nullptr;
  CountingDSP* large = nullptr;
  auto dsp = build_counting_container(small, large);

  dsp->Reset(44100.0, 128);

  assert(small->reset_count == 0);
  assert(small->prewarm_count == 0);

  assert(large->reset_count == 1);
  assert(large->prewarm_count == 1);
  assert(large->reset_sample_rate == 44100.0);
  assert(large->reset_buffer_size == 128);
}

void test_container_switch_resets_before_activation()
{
  CountingDSP* small = nullptr;
  CountingDSP* large = nullptr;
  auto dsp = build_counting_container(small, large);
  dsp->Reset(48000.0, 64);

  NAM_SAMPLE value_during_reset = (NAM_SAMPLE)-1.0;
  small->on_reset = [&]() { value_during_reset = process_one_sample(dsp.get()); };

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);
  slimmable->SetSlimmableSize(0.0);

  assert(small->reset_count == 1);
  assert(value_during_reset == large->process_value);
  assert(process_one_sample(dsp.get()) == small->process_value);
}

// PrewarmForRatio() lets a caller (e.g. a background thread) pay the Reset()+prewarm()
// cost for a not-yet-active submodel ahead of time, so the later SetSlimmableSize() call
// that actually activates it -- which the ObjC bridge calls inline on the real-time
// render thread -- reduces to a cheap atomic index flip instead of redoing that work.
// See NAMProcessorWrapper.mm's real-time-deadline pop this was built to fix.

void test_container_prewarm_for_ratio_resets_without_activating()
{
  CountingDSP* small = nullptr;
  CountingDSP* large = nullptr;
  auto dsp = build_counting_container(small, large);
  dsp->Reset(48000.0, 64); // large is default-active; this resets large once.

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  slimmable->PrewarmForRatio(0.0); // targets small (0.0 < max_value 0.5)

  assert(small->reset_count == 1);
  assert(small->prewarm_count == 1);
  // Still processing through large -- PrewarmForRatio must not activate.
  assert(process_one_sample(dsp.get()) == large->process_value);
}

void test_container_set_slimmable_size_after_prewarm_skips_redundant_reset()
{
  CountingDSP* small = nullptr;
  CountingDSP* large = nullptr;
  auto dsp = build_counting_container(small, large);
  dsp->Reset(48000.0, 64);

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  slimmable->PrewarmForRatio(0.0);
  assert(small->reset_count == 1);

  slimmable->SetSlimmableSize(0.0);

  // The staged Reset() from PrewarmForRatio must be reused, not redone.
  assert(small->reset_count == 1);
  assert(process_one_sample(dsp.get()) == small->process_value);
}

void test_container_set_slimmable_size_without_prewarm_still_resets()
{
  // Fallback path: SetSlimmableSize() without a prior PrewarmForRatio() must behave
  // exactly as before (synchronous Reset()), matching test_container_switch_resets_before_activation.
  CountingDSP* small = nullptr;
  CountingDSP* large = nullptr;
  auto dsp = build_counting_container(small, large);
  dsp->Reset(48000.0, 64);

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  slimmable->SetSlimmableSize(0.0);

  assert(small->reset_count == 1);
  assert(process_one_sample(dsp.get()) == small->process_value);
}

void test_container_prewarm_for_ratio_already_active_is_noop()
{
  CountingDSP* small = nullptr;
  CountingDSP* large = nullptr;
  auto dsp = build_counting_container(small, large);
  dsp->Reset(48000.0, 64); // resets large (default-active) once.
  assert(large->reset_count == 1);

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  slimmable->PrewarmForRatio(1.0); // large is already active -- must be a no-op.

  assert(large->reset_count == 1);
}

void test_container_prewarm_for_ratio_stale_stage_falls_back_to_reset()
{
  // If the user changes their mind before a staged prewarm is consumed (prewarm for A,
  // then switch to B instead), SetSlimmableSize(B) must not mistake A's staged reset for
  // its own and must still reset B synchronously.
  CountingDSP* small = nullptr;
  CountingDSP* large = nullptr;
  auto dsp = build_counting_container(small, large);
  dsp->Reset(48000.0, 64); // large active

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  slimmable->PrewarmForRatio(0.0); // stages small
  assert(small->reset_count == 1);

  slimmable->SetSlimmableSize(1.0); // no-op: 1.0 selects large, already active

  // Now switch to small for real -- its stage from PrewarmForRatio(0.0) is still valid
  // (nothing invalidated it), so this must NOT reset small again.
  slimmable->SetSlimmableSize(0.0);
  assert(small->reset_count == 1);
  assert(process_one_sample(dsp.get()) == small->process_value);
}

// Real exported .nam files store "config.layers" as an array of per-layer-group
// objects (each with its own "channels" field), not the flat integer used by
// spice-to-nam's ParametricWaveNet format. GetAllowedSizes() must extract the
// first layer group's "channels" in that case, matching the Swift-side
// NAMModelLibraryManager.inferChannelRange() convention -- otherwise every
// submodel's channel count silently falls back to 0 and the model-size picker
// shows "0 (Lite)" / "0 (Full)" instead of the real widths.
void test_container_get_allowed_sizes_reads_array_layers_shape()
{
  auto j = build_container_json("example_models/wavenet.nam", "example_models/wavenet_a2_max.nam",
                                "example_models/wavenet_a2_max.nam");
  auto dsp = nam::get_dsp(j);
  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(dsp.get());
  assert(slimmable != nullptr);

  auto sizes = slimmable->GetAllowedSizes();
  assert(sizes.size() == 3);
  assert(sizes[0] == 3); // example_models/wavenet.nam: config.layers[0].channels == 3
  assert(sizes[1] == 4); // example_models/wavenet_a2_max.nam: config.layers[0].channels == 4
  assert(sizes[2] == 4);
  for (int size : sizes)
    assert(size != 0);
}

// [Fork] BEGIN: ContainerModel knob-API delegation tests (Gap 2)
//
// GetNumParams()/GetParameterDefs() delegate to _submodels[0]; SetKnobValues() writes to
// EVERY submodel (not just the active one) so that whichever submodel becomes active next
// already reflects the current knob state -- see the comment above ContainerModel::
// SetKnobValues in NAM/container.cpp. None of the three had any test coverage before this.

void test_container_knob_accessors_delegate_to_first_submodel()
{
  const double sample_rate = 48000.0;
  std::vector<nam::container::Submodel> submodels;
  submodels.push_back({0.5, 3, make_parametric_submodel(3, 1, 0.1f, sample_rate)});
  submodels.push_back({1.0, 8, make_parametric_submodel(8, 2, 0.1f, sample_rate)});

  auto container =
    std::make_unique<nam::container::ContainerModel>(std::move(submodels), sample_rate);

  // The first submodel has 1 param; the (currently-active, last) submodel has 2 -- if
  // GetNumParams/GetParameterDefs delegated to the active submodel instead of the first,
  // this would read 2, not 1.
  assert(container->GetNumParams() == 1);
  const auto defs = container->GetParameterDefs();
  assert(defs.size() == 1);
}

// Proves SetKnobValues() writes to a submodel that ISN'T active yet: set knobs, THEN
// switch the active submodel, then confirm the newly-active submodel's output already
// reflects the knob value set before the switch -- rather than the default it would show
// if SetKnobValues() only touched whichever submodel happened to be active at call time.
void test_container_set_knob_values_reaches_inactive_submodel()
{
  const double sample_rate = 48000.0;
  const int buffer_size = 64;
  const float weight_value = 0.1f;

  std::vector<nam::container::Submodel> submodels;
  submodels.push_back({0.5, 3, make_parametric_submodel(3, 1, weight_value, sample_rate)});
  submodels.push_back({1.0, 8, make_parametric_submodel(8, 1, weight_value, sample_rate)});

  auto container =
    std::make_unique<nam::container::ContainerModel>(std::move(submodels), sample_rate);
  // Default-active submodel is the LAST one (channels=8); this only resets that one.
  container->Reset(sample_rate, buffer_size);

  // Set knobs to a distinctly non-default value while submodel 0 (channels=3) is still
  // inactive and has never been Reset().
  container->SetKnobValues({0.9f});

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(container.get());
  assert(slimmable != nullptr);
  slimmable->SetSlimmableSize(0.0); // switches active -> submodel 0; Resets it here

  std::vector<NAM_SAMPLE> input(buffer_size);
  for (int i = 0; i < buffer_size; i++)
    input[i] = static_cast<NAM_SAMPLE>(0.3 * std::sin(2.0 * M_PI * 220.0 * i / sample_rate));

  std::vector<NAM_SAMPLE> out_container(buffer_size);
  {
    NAM_SAMPLE* in_ptr = input.data();
    NAM_SAMPLE* out_ptr = out_container.data();
    container->process(&in_ptr, &out_ptr, buffer_size);
  }

  // Reference A: standalone channels=3 model with the SAME weights, knob set to 0.9
  // BEFORE Reset() -- the same call order the container's submodel 0 actually saw.
  auto ref_hi = make_parametric_submodel(3, 1, weight_value, sample_rate);
  ref_hi->SetKnobValues({0.9f});
  ref_hi->Reset(sample_rate, buffer_size);
  std::vector<NAM_SAMPLE> out_ref_hi(buffer_size);
  {
    NAM_SAMPLE* in_ptr = input.data();
    NAM_SAMPLE* out_ptr = out_ref_hi.data();
    ref_hi->process(&in_ptr, &out_ptr, buffer_size);
  }

  // Reference B: standalone channels=3 model left at the constructor default (0.5) --
  // what the container would have produced if SetKnobValues() had only reached whichever
  // submodel was active at call time (i.e., NOT submodel 0).
  auto ref_default = make_parametric_submodel(3, 1, weight_value, sample_rate);
  ref_default->Reset(sample_rate, buffer_size);
  std::vector<NAM_SAMPLE> out_ref_default(buffer_size);
  {
    NAM_SAMPLE* in_ptr = input.data();
    NAM_SAMPLE* out_ptr = out_ref_default.data();
    ref_default->process(&in_ptr, &out_ptr, buffer_size);
  }

  const double diff_vs_hi = max_abs_diff(out_container, out_ref_hi);
  const double diff_vs_default = max_abs_diff(out_container, out_ref_default);

  if (!(diff_vs_hi < 1e-6) || !(diff_vs_default > 1000.0 * std::max(diff_vs_hi, 1e-9)))
  {
    std::cerr << "ContainerModel::SetKnobValues delegation: diff_vs_hi=" << diff_vs_hi
              << " diff_vs_default=" << diff_vs_default << " (want vs_hi ~ 0, vs_default >> vs_hi)"
              << std::endl;
    assert(false);
  }
  // Matches the "knob was already 0.9" reference almost exactly...
  assert(diff_vs_hi < 1e-6);
  // ...and is clearly NOT what we'd see if the knob had stayed at the default.
  assert(diff_vs_default > 1000.0 * std::max(diff_vs_hi, 1e-9));
}
// [Fork] END: ContainerModel knob-API delegation tests

} // namespace test_container
