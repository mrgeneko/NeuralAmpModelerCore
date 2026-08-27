// Numerical verification for the A2 fast-path WaveNet:
// for a config that matches the A2 shape, the fast path must produce the same
// output as the generic WaveNet on the same input and weights.
//
// Built only when NAM_ENABLE_A2_FAST is defined at compile time.

#if defined(NAM_ENABLE_A2_FAST)

  #include <algorithm>
  #include <cassert>
  #include <cmath>
  #include <cstdint>
  #include <iostream>
  #include <memory>
  #include <random>
  #include <string>
  #include <utility>
  #include <vector>

  #include "json.hpp"

  #include "NAM/dsp.h"
  #include "NAM/wavenet/a2_fast.h"
  #include "NAM/wavenet/model.h"
  #include "NAM/wavenet/parametric_wavenet.h" // [Fork] for the parametric fast-path tests

  #include "allocation_tracking.h"

namespace test_a2_fast
{
namespace
{

// Build a JSON config with the A2 shape, parameterized by channel count
// (3 = A2 nano, 8 = A2 standard). Follows the real .nam schema so both the
// strict detector and the generic parser accept it.
nlohmann::json build_a2_config(int channels)
{
  using nlohmann::json;

  json activation = json::array();
  json gating_mode = json::array();
  json secondary = json::array();
  for (int i = 0; i < nam::wavenet::a2_fast::kNumLayers; i++)
  {
    activation.push_back({{"type", "LeakyReLU"}, {"negative_slope", nam::wavenet::a2_fast::kLeakySlope}});
    gating_mode.push_back("none");
    secondary.push_back(nullptr);
  }

  json kernel_sizes = json::array();
  json dilations = json::array();
  for (int i = 0; i < nam::wavenet::a2_fast::kNumLayers; i++)
  {
    kernel_sizes.push_back(nam::wavenet::a2_fast::kKernelSizes[i]);
    dilations.push_back(nam::wavenet::a2_fast::kDilations[i]);
  }

  json film_inactive = {{"active", false}, {"shift", true}, {"groups", 1}};

  json layer;
  layer["input_size"] = 1;
  layer["condition_size"] = 1;
  layer["channels"] = channels;
  layer["bottleneck"] = channels;
  layer["kernel_sizes"] = kernel_sizes;
  layer["dilations"] = dilations;
  layer["activation"] = activation;
  layer["gating_mode"] = gating_mode;
  layer["secondary_activation"] = secondary;
  layer["head"] = {{"out_channels", 1}, {"kernel_size", nam::wavenet::a2_fast::kHeadKernelSize}, {"bias", true}};
  layer["head1x1"] = {{"active", false}, {"out_channels", 1}, {"groups", 1}};
  layer["layer1x1"] = {{"active", true}, {"groups", 1}};
  layer["conv_pre_film"] = film_inactive;
  layer["conv_post_film"] = film_inactive;
  layer["input_mixin_pre_film"] = film_inactive;
  layer["input_mixin_post_film"] = film_inactive;
  layer["activation_pre_film"] = film_inactive;
  layer["activation_post_film"] = film_inactive;
  layer["layer1x1_post_film"] = film_inactive;
  layer["head1x1_post_film"] = film_inactive;
  layer["groups_input"] = 1;
  layer["groups_input_mixin"] = 1;

  json config;
  config["layers"] = json::array({layer});
  config["head_scale"] = 0.01f;
  return config;
}

// Weight count for the A2 layer array with the given channel count.
// Must match the order and sizes expected by both the generic WaveNet parser
// and A2FastModel::_load_weights.
int a2_weight_count(int channels)
{
  const int bn = channels;
  int total = /*rechannel*/ channels;
  for (int i = 0; i < nam::wavenet::a2_fast::kNumLayers; i++)
  {
    const int K = nam::wavenet::a2_fast::kKernelSizes[i];
    total += bn * channels * K + bn; // conv1d weights + bias
    total += bn; // input mixin (no bias)
    total += channels * bn + channels; // layer1x1 + bias
  }
  total += channels * nam::wavenet::a2_fast::kHeadKernelSize + 1; // head rechannel + bias
  total += 1; // trailing head_scale (read by WaveNet::set_weights_)
  return total;
}

std::vector<float> make_deterministic_weights(int count, uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
  std::vector<float> w(count);
  for (auto& x : w)
    x = dist(rng);
  return w;
}

std::vector<NAM_SAMPLE> make_test_input(int num_frames, double sample_rate)
{
  std::vector<NAM_SAMPLE> in(num_frames);
  // Two-tone signal so the network sees varied content.
  for (int i = 0; i < num_frames; i++)
  {
    const double t = static_cast<double>(i) / sample_rate;
    in[i] = static_cast<NAM_SAMPLE>(0.25 * std::sin(2.0 * M_PI * 220.0 * t) + 0.10 * std::sin(2.0 * M_PI * 1230.0 * t));
  }
  return in;
}

std::vector<NAM_SAMPLE> run_dsp(nam::DSP& dsp, const std::vector<NAM_SAMPLE>& input, int block_size)
{
  // Reset also prewarms.
  dsp.Reset(48000.0, block_size);
  std::vector<NAM_SAMPLE> out(input.size(), static_cast<NAM_SAMPLE>(0));
  int pos = 0;
  const int total = static_cast<int>(input.size());
  while (pos < total)
  {
    const int n = std::min(block_size, total - pos);
    const NAM_SAMPLE* in_ptr = input.data() + pos;
    NAM_SAMPLE* out_ptr = out.data() + pos;
    const NAM_SAMPLE* in_arr[] = {in_ptr};
    NAM_SAMPLE* out_arr[] = {out_ptr};
    dsp.process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, n);
    pos += n;
  }
  return out;
}

void compare(const std::vector<NAM_SAMPLE>& a, const std::vector<NAM_SAMPLE>& b, int channels, int block_size,
             double tol)
{
  assert(a.size() == b.size());
  double max_diff = 0.0;
  int max_i = 0;
  for (size_t i = 0; i < a.size(); i++)
  {
    const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (d > max_diff)
    {
      max_diff = d;
      max_i = static_cast<int>(i);
    }
  }
  if (!(max_diff < tol))
  {
    std::cerr << "A2FastModel<" << channels << "> diverges from generic WaveNet "
              << "(block=" << block_size << "): max |diff| = " << max_diff << " at i=" << max_i
              << " (generic=" << a[max_i] << ", fast=" << b[max_i] << ")" << std::endl;
    assert(false);
  }
}

// [Fork] BEGIN: parametric fast-path numerical verification helpers
// Build a compact ParametricWaveNet config (config["layers"] is an int, not an array).
nlohmann::json build_parametric_a2_config(int channels, int num_params)
{
  // [Fork] Parametric models are SKIP-ONLY; an untagged/residual model is rejected
  // on load, so every test config must declare the tag.
  nlohmann::json cfg;
  cfg["layers"] = channels;
  cfg["head_scale"] = 0.01f;
  cfg["parametric"]["condition_size"] = num_params;
  cfg["parametric"]["schema_version"] = 1;
  cfg["parametric"]["head_mode"] = "skip";
  nlohmann::json params = nlohmann::json::array();
  for (int i = 0; i < num_params; i++)
    params.push_back({{"name", "knob" + std::to_string(i)}, {"minimum", 0.0f}, {"maximum", 1.0f}, {"default", 0.5f}});
  cfg["parametric"]["parameters"] = params;
  return cfg;
}

// Weight count for the ParametricWaveNet A2 layer array: same as a2_weight_count() plus
// a per-layer FiLM net (num_params -> 2*channels, with bias), per the weight layout
// documented in parametric_wavenet.h. Verified to reproduce the known-correct 2147
// (channels=3, num_params=1) constant used elsewhere in the test suite.
int parametric_a2_weight_count(int channels, int num_params)
{
  int total = a2_weight_count(channels);
  // a2_weight_count already counts rechannel + per-layer conv/mixin/l1x1 + head + head_scale;
  // add the per-layer FiLM net (weight + bias) on top.
  total += nam::wavenet::a2_fast::kNumLayers * (2 * channels * num_params + 2 * channels);
  return total;
}

void compare_parametric(const std::vector<NAM_SAMPLE>& a, const std::vector<NAM_SAMPLE>& b, int channels,
                         int block_size, double tol)
{
  assert(a.size() == b.size());
  double max_diff = 0.0;
  int max_i = 0;
  for (size_t i = 0; i < a.size(); i++)
  {
    const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (d > max_diff)
    {
      max_diff = d;
      max_i = static_cast<int>(i);
    }
  }
  if (!(max_diff < tol))
  {
    std::cerr << "ParametricA2FastModel<" << channels << "> diverges from generic ParametricWaveNet "
              << "(block=" << block_size << "): max |diff| = " << max_diff << " at i=" << max_i
              << " (generic=" << a[max_i] << ", fast=" << b[max_i] << ")" << std::endl;
    assert(false);
  }
}
// [Fork] END: parametric fast-path numerical verification helpers

// [Fork] END: LoRA fast-path numerical verification helpers

} // namespace

void test_detector_matches_nano()
{
  auto cfg = build_a2_config(3);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 3);
}

void test_detector_matches_standard()
{
  auto cfg = build_a2_config(8);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 8);
}

void test_detector_accepts_nonstandard_head_scale()
{
  auto cfg = build_a2_config(8);
  cfg["head_scale"] = 0.0042f;
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 8);
}

void test_detector_rejects_wrong_channels()
{
  // 10 is outside the supported fast-path set {3, 4, 5, 6, 7, 8, 9}.
  auto cfg = build_a2_config(3);
  cfg["layers"][0]["channels"] = 10;
  cfg["layers"][0]["bottleneck"] = 10;
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_detector_matches_medium()
{
  auto cfg = build_a2_config(4);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 4);
}

// [Fork] BEGIN: widths 5, 6, 7
void test_detector_matches_width5()
{
  auto cfg = build_a2_config(5);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 5);
}

void test_detector_matches_width6()
{
  auto cfg = build_a2_config(6);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 6);
}

void test_detector_matches_width7()
{
  auto cfg = build_a2_config(7);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 7);
}
// [Fork] END: widths 5, 6, 7

// [Fork] BEGIN: width 9
void test_detector_matches_width9()
{
  auto cfg = build_a2_config(9);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 9);
}
// [Fork] END: width 9

void test_parametric_detector_matches_supported_widths()
{
  for (int channels : {3, 4, 5, 6, 7, 8, 9})
  {
    nlohmann::json cfg;
    cfg["layers"] = channels;
    cfg["head_scale"] = 1.0f;
    cfg["parametric"]["condition_size"] = 1;
    int ch = 0, np = 0;
    assert(nam::wavenet::a2_fast::is_parametric_a2_shape(cfg, &ch, &np));
    assert(ch == channels);
    assert(np == 1);
  }

  // 10 is outside the supported fast-path set {3, 4, 5, 6, 7, 8, 9}.
  nlohmann::json cfg;
  cfg["layers"] = 10;
  cfg["head_scale"] = 1.0f;
  cfg["parametric"]["condition_size"] = 1;
  assert(!nam::wavenet::a2_fast::is_parametric_a2_shape(cfg, nullptr, nullptr));
}

void test_parametric_detector_accepts_explicit_film_type()
{
  // "type":"film" (what every LoRA-off export now declares explicitly) must still be
  // accepted -- the gate is an exclusion for "film+lora" (or any other non-"film" value),
  // not a requirement that the key be absent.
  for (int channels : {3, 4, 5, 6, 7, 8})
  {
    nlohmann::json cfg;
    cfg["layers"] = channels;
    cfg["head_scale"] = 1.0f;
    cfg["parametric"]["condition_size"] = 1;
    cfg["parametric"]["type"] = "film";
    int ch = 0, np = 0;
    assert(nam::wavenet::a2_fast::is_parametric_a2_shape(cfg, &ch, &np));
    assert(ch == channels);
  }
}

void test_detector_rejects_wrong_kernel_sizes()
{
  auto cfg = build_a2_config(8);
  cfg["layers"][0]["kernel_sizes"][0] = 7; // tweak first entry
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_detector_rejects_wrong_activation()
{
  auto cfg = build_a2_config(8);
  cfg["layers"][0]["activation"][0] = {{"type", "Tanh"}};
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_detector_rejects_gating()
{
  auto cfg = build_a2_config(3);
  cfg["layers"][0]["gating_mode"][0] = "gated";
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

// A condition DSP routes the conditioning signal through a nested model; the fast
// path has no such stage. The nested model holds its own weights, so the parent
// weight stream is unchanged and only the detector can catch this -- otherwise the
// fast path would silently produce different audio than the generic WaveNet.
void test_detector_rejects_condition_dsp()
{
  auto cfg = build_a2_config(8);
  cfg["condition_dsp"] = {{"version", "0.5.0"}, {"architecture", "Linear"},
                          {"config", nlohmann::json::object()}, {"weights", nlohmann::json::array()}};
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

// Legacy boolean `gated` (pre-gating_mode schema) maps to GATED layers in the
// generic parser, which the fast path does not implement.
void test_detector_rejects_legacy_gated()
{
  auto cfg = build_a2_config(3);
  cfg["layers"][0]["gated"] = true;
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_matches_generic(int channels)
{
  const auto cfg = build_a2_config(channels);
  const int weight_count = a2_weight_count(channels);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0xA2FA500u + channels);

  // Fast path: build through the explicit A2 factory.
  auto fast_cfg = nam::wavenet::a2_fast::create_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_fast = weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), 48000.0);

  // Generic path: call parse_config_json directly so the dispatcher's A2 shortcut
  // doesn't kick in. This gives us the reference WaveNet against the same config.
  auto generic_cfg = nam::wavenet::parse_config_json(cfg, 48000.0);
  std::vector<float> w_gen = weights;
  auto generic_dsp = generic_cfg.create(std::move(w_gen), 48000.0);

  // Exercise with two block sizes to catch any off-by-one in the ring-buffer rewind.
  const int total = 2048;
  const auto input = make_test_input(total, 48000.0);
  for (int block : {64, 256})
  {
    const auto out_generic = run_dsp(*generic_dsp, input, block);
    const auto out_fast = run_dsp(*fast_dsp, input, block);
    // 1e-5 is what nam2c_verify.py uses as its byte-exactness threshold; allow a little
    // slack because the generic path sums through Eigen and may reorder FMAs.
    compare(out_generic, out_fast, channels, block, /*tol=*/5e-5);
  }
}

void test_matches_generic_nano()
{
  test_matches_generic(3);
}

void test_matches_generic_medium()
{
  test_matches_generic(4);
}

// [Fork] BEGIN: widths 5, 6, 7
void test_matches_generic_width5()
{
  test_matches_generic(5);
}

void test_matches_generic_width6()
{
  test_matches_generic(6);
}

void test_matches_generic_width7()
{
  test_matches_generic(7);
}
// [Fork] END: widths 5, 6, 7

void test_matches_generic_standard()
{
  test_matches_generic(8);
}

// [Fork] BEGIN: width 9
void test_matches_generic_width9()
{
  test_matches_generic(9);
}
// [Fork] END: width 9

// [Fork] BEGIN: parametric fast-path numerical verification (mirrors
// test_matches_generic above, but for ParametricA2FastModel vs. the generic
// ParametricWaveNet reference exposed via create_parametric_config_generic()).
void test_parametric_matches_generic(int channels, int num_params)
{
  const auto cfg = build_parametric_a2_config(channels, num_params);
  const int weight_count = parametric_a2_weight_count(channels, num_params);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0xFA2FA50u + channels * 100 + num_params);

  // Fast path: build through the explicit parametric A2 factory.
  auto fast_cfg = nam::wavenet::a2_fast::create_parametric_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_fast = weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), 48000.0);

  // Generic path: call create_parametric_config_generic() directly so the fast-path
  // dispatch doesn't kick in. This gives the reference ParametricWaveNet for the same config.
  auto generic_cfg = nam::wavenet::create_parametric_config_generic(cfg, 48000.0);
  std::vector<float> w_gen = weights;
  auto generic_dsp = generic_cfg->create(std::move(w_gen), 48000.0);

  // Both models default every knob to 0.5 on construction; exercise a non-default knob
  // setting too so FiLM conditioning is actually tested, not just its zero/default case.
  std::vector<float> knobs(num_params);
  for (int i = 0; i < num_params; i++)
    knobs[i] = 0.2f + 0.6f * (static_cast<float>(i) / std::max(1, num_params));
  fast_dsp->SetKnobValues(knobs);
  generic_dsp->SetKnobValues(knobs);

  const int total = 2048;
  const auto input = make_test_input(total, 48000.0);
  for (int block : {64, 256})
  {
    const auto out_generic = run_dsp(*generic_dsp, input, block);
    const auto out_fast = run_dsp(*fast_dsp, input, block);
    compare_parametric(out_generic, out_fast, channels, block, /*tol=*/5e-5);
  }
}

// [Fork] BEGIN: SKIP-head parametric fast-path verification (migration 5b).
//
// The fast path accumulates _head_sum inside hand-unrolled per-width SIMD kernels; the
// generic path gets its skip accumulation from LayerArray (use_final_layer_output_for_head
// = false), which is long-standing, well-tested code. Comparing the two is what proves the
// SIMD accumulate is correct -- separately for every width we ship.
void test_parametric_skip_matches_generic(int channels, int num_params)
{
  auto cfg = build_parametric_a2_config(channels, num_params);
  cfg["parametric"]["head_mode"] = "skip";

  const int weight_count = parametric_a2_weight_count(channels, num_params);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0x5C1B0000u + channels * 100 + num_params);

  auto fast_cfg = nam::wavenet::a2_fast::create_parametric_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_fast = weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), 48000.0);

  auto generic_cfg = nam::wavenet::create_parametric_config_generic(cfg, 48000.0);
  std::vector<float> w_gen = weights;
  auto generic_dsp = generic_cfg->create(std::move(w_gen), 48000.0);

  // Exercise a non-default knob setting so FiLM is genuinely active, not just its default.
  std::vector<float> knobs(num_params);
  for (int i = 0; i < num_params; i++)
    knobs[i] = 0.2f + 0.6f * (static_cast<float>(i) / std::max(1, num_params));
  fast_dsp->SetKnobValues(knobs);
  generic_dsp->SetKnobValues(knobs);

  const int total = 2048;
  const auto input = make_test_input(total, 48000.0);
  for (int block : {64, 256})
  {
    const auto out_generic = run_dsp(*generic_dsp, input, block);
    const auto out_fast = run_dsp(*fast_dsp, input, block);
    compare_parametric(out_generic, out_fast, channels, block, /*tol=*/5e-5);
  }
}

void test_parametric_skip_matches_generic_nano()     { test_parametric_skip_matches_generic(3, 1); }
void test_parametric_skip_matches_generic_medium()   { test_parametric_skip_matches_generic(4, 1); }
void test_parametric_skip_matches_generic_width5()   { test_parametric_skip_matches_generic(5, 2); }
void test_parametric_skip_matches_generic_width6()   { test_parametric_skip_matches_generic(6, 2); }
void test_parametric_skip_matches_generic_width7()   { test_parametric_skip_matches_generic(7, 2); }
void test_parametric_skip_matches_generic_standard() { test_parametric_skip_matches_generic(8, 2); }
void test_parametric_skip_matches_generic_width9()   { test_parametric_skip_matches_generic(9, 2); }

// [Fork] END: SKIP-head parametric fast-path verification

void test_parametric_matches_generic_nano()
{
  test_parametric_matches_generic(3, 1);
}

void test_parametric_matches_generic_medium()
{
  test_parametric_matches_generic(4, 1);
}

// [Fork] BEGIN: widths 5, 6, 7
void test_parametric_matches_generic_width5()
{
  test_parametric_matches_generic(5, 1);
}

void test_parametric_matches_generic_width6()
{
  test_parametric_matches_generic(6, 1);
}

void test_parametric_matches_generic_width7()
{
  test_parametric_matches_generic(7, 1);
}
// [Fork] END: widths 5, 6, 7

void test_parametric_matches_generic_standard()
{
  test_parametric_matches_generic(8, 2);
}

// [Fork] BEGIN: width 9
void test_parametric_matches_generic_width9()
{
  test_parametric_matches_generic(9, 2);
}
// [Fork] END: width 9
// [Fork] END: parametric fast-path numerical verification

// The fast path must report the same prewarm count as the generic WaveNet it
// replaces; otherwise Reset() warms the two by a different number of samples and
// their first post-Reset output can diverge (regression guard for the A2 prewarm
// off-by-one). Builds both from the identical config via the same dual path as
// test_matches_generic.
void test_prewarm_matches_generic(int channels)
{
  const auto cfg = build_a2_config(channels);
  const int weight_count = a2_weight_count(channels);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0xA2FA500u + channels);

  auto fast_cfg = nam::wavenet::a2_fast::create_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_fast = weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), 48000.0);

  auto generic_cfg = nam::wavenet::parse_config_json(cfg, 48000.0);
  std::vector<float> w_gen = weights;
  auto generic_dsp = generic_cfg.create(std::move(w_gen), 48000.0);

  assert(fast_dsp->GetPrewarmSamples() == generic_dsp->GetPrewarmSamples());
}

void test_prewarm_matches_generic_nano()
{
  test_prewarm_matches_generic(3);
}

void test_prewarm_matches_generic_standard()
{
  test_prewarm_matches_generic(8);
}

// Real-time safety: once the DSP has been Reset (buffers sized, prewarmed),
// subsequent process() calls must not allocate or free heap memory. Uses the
// same allocation-tracking infrastructure as the generic WaveNet RT-safety
// tests (tools/test/allocation_tracking.{h,cpp}) — overridden malloc/free and
// global new/delete increment counters while tracking is enabled.
void test_process_realtime_safe(int channels)
{
  const auto cfg = build_a2_config(channels);
  const int weight_count = a2_weight_count(channels);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0xA2FA500u + channels);

  auto fast_cfg = nam::wavenet::a2_fast::create_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_fast = weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), 48000.0);

  // Exercise several block sizes all within a single pre-sized state so the
  // internal "num_frames > max_buffer_size" guard in process() never fires
  // (which would legitimately reallocate).
  const int max_buffer = 256;
  fast_dsp->Reset(48000.0, max_buffer);

  const int total = 4 * max_buffer;
  const auto input = make_test_input(total, 48000.0);
  std::vector<NAM_SAMPLE> output(total, 0.0);

  // Warm up caches / any lazy init with one untracked pass.
  {
    const NAM_SAMPLE* in = input.data();
    NAM_SAMPLE* out = output.data();
    const NAM_SAMPLE* in_arr[] = {in};
    NAM_SAMPLE* out_arr[] = {out};
    fast_dsp->process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, max_buffer);
  }

  for (int block : {1, 32, 64, 128, 256})
  {
    std::string test_name = "A2FastModel<" + std::to_string(channels) + ">::process block=" + std::to_string(block);
    allocation_tracking::run_allocation_test_no_allocations(
      nullptr,
      [&]() {
        int pos = 0;
        while (pos + block <= total)
        {
          const NAM_SAMPLE* in = input.data() + pos;
          NAM_SAMPLE* out = output.data() + pos;
          const NAM_SAMPLE* in_arr[] = {in};
          NAM_SAMPLE* out_arr[] = {out};
          fast_dsp->process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, block);
          pos += block;
        }
      },
      nullptr, test_name.c_str());
  }
}

void test_process_realtime_safe_nano()
{
  test_process_realtime_safe(3);
}

void test_process_realtime_safe_medium()
{
  test_process_realtime_safe(4);
}

// [Fork] BEGIN: widths 5, 6, 7
void test_process_realtime_safe_width5()
{
  test_process_realtime_safe(5);
}

void test_process_realtime_safe_width6()
{
  test_process_realtime_safe(6);
}

void test_process_realtime_safe_width7()
{
  test_process_realtime_safe(7);
}
// [Fork] END: widths 5, 6, 7

void test_process_realtime_safe_standard()
{
  test_process_realtime_safe(8);
}

// [Fork] BEGIN: width 9
void test_process_realtime_safe_width9()
{
  test_process_realtime_safe(9);
}
// [Fork] END: width 9

// [Fork] BEGIN: per-buffer knob smoothing convergence test (Gap 1).
//
// ParametricA2FastModel::_recompute_film one-pole smooths knob targets toward the
// atomic value SetKnobValues() writes (see a2_fast.cpp ~line 2160), specifically to
// avoid zipper noise on a sudden UI knob move. Prior to this test, that smoothing had
// zero coverage: nothing proved it actually ramps rather than snapping instantly (or,
// conversely, that it eventually gets there instead of stalling partway).
//
// Strategy: run one model through a knob jump (0.5 -> 1.0) and sample immediately
// (one buffer later) and long after (many buffers later, past the smoother's time
// constant). Compare both samples against a second, separately-constructed reference
// model whose knobs were set to 1.0 before its first process() call (so by the time
// IT has run for the same number of buffers, its own smoother has also fully
// converged to 1.0 -- there is no way to observe a "no-transition-at-all" reference,
// since the smoother always starts at the constructor's default 0.5). Both models
// see byte-identical input content on every buffer, so once each has converged, their
// internal FIR/ring-buffer state is also driven by the same steady periodic input and
// the two outputs must agree closely.
void test_parametric_knob_smoothing_converges()
{
  const int channels = 3;
  const int num_params = 1;
  const double sample_rate = 48000.0;
  const int block = 256;

  const auto cfg = build_parametric_a2_config(channels, num_params);
  const int weight_count = parametric_a2_weight_count(channels, num_params);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0x50007E5u);

  // Fixed input content, replayed identically every buffer.
  const auto input = make_test_input(block, sample_rate);

  auto run_buffers = [&](nam::DSP& dsp, int n) {
    std::vector<NAM_SAMPLE> out(block, static_cast<NAM_SAMPLE>(0));
    NAM_SAMPLE* in_ptr = const_cast<NAM_SAMPLE*>(input.data());
    NAM_SAMPLE* out_ptr = out.data();
    NAM_SAMPLE* in_arr[] = {in_ptr};
    NAM_SAMPLE* out_arr[] = {out_ptr};
    for (int i = 0; i < n; i++)
      dsp.process(in_arr, out_arr, block);
    return out;
  };

  // Model A: starts at the default (0.5), jumps to 1.0, sampled right after the jump
  // and again long after.
  auto cfg_a = nam::wavenet::a2_fast::create_parametric_a2_fast_config(cfg, sample_rate);
  std::vector<float> w_a = weights;
  auto dsp_a = cfg_a->create(std::move(w_a), sample_rate);
  dsp_a->Reset(sample_rate, block);
  run_buffers(*dsp_a, 2); // let it "settle" at the default -- no-op, already there
  dsp_a->SetKnobValues({1.0f});
  const auto out_immediate = run_buffers(*dsp_a, 1); // ONE buffer immediately after the jump

  // Enough buffers for (1 - coeff)^n to be negligible, derived from the same formula
  // SetMaxBufferSize() uses (a2_fast.cpp ~line 2205), plus a healthy margin.
  const float smoothTimeSamples = static_cast<float>(sample_rate) * 0.100f;
  const float coeff = 1.0f - std::exp(-static_cast<float>(block) / smoothTimeSamples);
  const int n_converge = static_cast<int>(std::ceil(std::log(1e-6) / std::log(1.0 - coeff))) + 20;
  const auto out_settled = run_buffers(*dsp_a, n_converge); // MANY buffers later

  // Model B: reference -- knobs set to 1.0 before the very first process() call (no
  // jump to observe from this model's own perspective).
  auto cfg_b = nam::wavenet::a2_fast::create_parametric_a2_fast_config(cfg, sample_rate);
  std::vector<float> w_b = weights;
  auto dsp_b = cfg_b->create(std::move(w_b), sample_rate);
  dsp_b->Reset(sample_rate, block);
  dsp_b->SetKnobValues({1.0f});
  const auto out_reference = run_buffers(*dsp_b, 2 + 1 + n_converge);

  auto max_abs_diff = [](const std::vector<NAM_SAMPLE>& a, const std::vector<NAM_SAMPLE>& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++)
      d = std::max(d, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return d;
  };

  const double diff_immediate = max_abs_diff(out_immediate, out_reference);
  const double diff_settled = max_abs_diff(out_settled, out_reference);

  if (!(diff_settled < 1e-5) || !(diff_immediate > 100.0 * std::max(diff_settled, 1e-9)))
  {
    std::cerr << "ParametricA2FastModel knob smoothing: diff_immediate=" << diff_immediate
              << " diff_settled=" << diff_settled << " (want settled << immediate, settled ~ 0)" << std::endl;
    assert(false);
  }
  // The jump has barely begun (knob moved by only 1-(1-coeff) of the way in one
  // buffer): it must not already match the fully-settled reference.
  assert(diff_immediate > 100.0 * std::max(diff_settled, 1e-9));
  // Fully converged: must closely match the reference model's own converged state.
  assert(diff_settled < 1e-5);
}
// [Fork] END: per-buffer knob smoothing convergence test

// [Fork] BEGIN: knob smoothing recalibrates after an implicit (Reset()-less) buffer-size grow.
//
// process() has a defensive fallback (a2_fast.cpp ~line 3146): if num_frames ever exceeds what
// Reset() last sized buffers for, it calls SetMaxBufferSize(num_frames) directly, without an
// intervening Reset(). Bug: SetMaxBufferSize() didn't recompute _knob_smooth_coeff, so it stayed
// calibrated for whatever (smaller) buffer size Reset() was last called with -- meaning a knob
// jump would ramp far slower than the intended ~100ms once the buffer size grew, since each now-
// larger buffer covers more wall-clock time but still closes only the smaller-buffer-sized
// fraction of the gap.
//
// Strategy: mirrors test_parametric_knob_smoothing_converges above, but Model A reaches
// `large_block` via the implicit fallback (Reset() at `small_block`, then a single process()
// call at `large_block`) instead of via an explicit Reset() at that size. If the coefficient
// were still stale, Model A's knob jump would converge far slower than n_converge buffers
// (computed from the *correct*, large_block-calibrated coefficient) predicts, and diff_settled
// would fail to shrink to the reference's converged state within that budget.
void test_parametric_knob_smoothing_recalibrates_after_implicit_resize()
{
  const int channels = 3;
  const int num_params = 1;
  const double sample_rate = 48000.0;
  const int small_block = 64;
  const int large_block = 512;

  const auto cfg = build_parametric_a2_config(channels, num_params);
  const int weight_count = parametric_a2_weight_count(channels, num_params);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0x2E512E5u);

  // Long enough to cover a full large_block-sized call; small_block calls just read its prefix.
  const auto input = make_test_input(large_block, sample_rate);

  auto run_buffers = [&](nam::DSP& dsp, int block, int n) {
    std::vector<NAM_SAMPLE> out(block, static_cast<NAM_SAMPLE>(0));
    NAM_SAMPLE* in_ptr = const_cast<NAM_SAMPLE*>(input.data());
    NAM_SAMPLE* out_ptr = out.data();
    NAM_SAMPLE* in_arr[] = {in_ptr};
    NAM_SAMPLE* out_arr[] = {out_ptr};
    for (int i = 0; i < n; i++)
      dsp.process(in_arr, out_arr, block);
    return out;
  };

  // Model A: Reset() at small_block, settle at the default there, then jump the knob and
  // immediately call process() at large_block -- this single call both triggers the implicit
  // resize (num_frames=large_block > GetMaxBufferSize()=small_block) AND is the buffer sampled
  // right after the jump.
  auto cfg_a = nam::wavenet::a2_fast::create_parametric_a2_fast_config(cfg, sample_rate);
  std::vector<float> w_a = weights;
  auto dsp_a = cfg_a->create(std::move(w_a), sample_rate);
  dsp_a->Reset(sample_rate, small_block);
  run_buffers(*dsp_a, small_block, 2); // settle at default (0.5) at the small size
  dsp_a->SetKnobValues({1.0f});
  const auto out_immediate = run_buffers(*dsp_a, large_block, 1); // triggers the implicit resize

  // Buffers needed for (1 - coeff)^n to be negligible AT large_block -- the size that matters
  // once the implicit resize has happened. Uses the *correct* formula (as if recomputed for
  // large_block), which is exactly what should now be in effect after the fix.
  const float smoothTimeSamples = static_cast<float>(sample_rate) * 0.100f;
  const float coeff = 1.0f - std::exp(-static_cast<float>(large_block) / smoothTimeSamples);
  const int n_converge = static_cast<int>(std::ceil(std::log(1e-6) / std::log(1.0 - coeff))) + 20;
  const auto out_settled = run_buffers(*dsp_a, large_block, n_converge);

  // Model B: reference -- Reset() directly at large_block (never touches the fallback path),
  // knobs jumped to 1.0 before its first process() call. Its smoothing is unambiguously
  // calibrated for large_block from the start.
  auto cfg_b = nam::wavenet::a2_fast::create_parametric_a2_fast_config(cfg, sample_rate);
  std::vector<float> w_b = weights;
  auto dsp_b = cfg_b->create(std::move(w_b), sample_rate);
  dsp_b->Reset(sample_rate, large_block);
  dsp_b->SetKnobValues({1.0f});
  const auto out_reference = run_buffers(*dsp_b, large_block, 1 + n_converge);

  auto max_abs_diff = [](const std::vector<NAM_SAMPLE>& a, const std::vector<NAM_SAMPLE>& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size(); i++)
      d = std::max(d, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return d;
  };

  const double diff_immediate = max_abs_diff(out_immediate, out_reference);
  const double diff_settled = max_abs_diff(out_settled, out_reference);

  if (!(diff_settled < 1e-5) || !(diff_immediate > 100.0 * std::max(diff_settled, 1e-9)))
  {
    std::cerr << "ParametricA2FastModel knob smoothing after implicit resize: diff_immediate=" << diff_immediate
              << " diff_settled=" << diff_settled
              << " (want settled << immediate, settled ~ 0 -- if settled is large, the "
              << "coefficient is still stale from small_block after the implicit resize)" << std::endl;
    assert(false);
  }
  assert(diff_immediate > 100.0 * std::max(diff_settled, 1e-9));
  assert(diff_settled < 1e-5);
}
// [Fork] END: knob smoothing recalibrates after an implicit buffer-size grow

} // namespace test_a2_fast

#endif // NAM_ENABLE_A2_FAST
