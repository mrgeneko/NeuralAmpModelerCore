#include "parametric_wavenet.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "../registry.h"

namespace nam
{
namespace wavenet
{

// ParametricWaveNet ===========================================================

ParametricWaveNet::ParametricWaveNet(const int in_channels,
                                     const std::vector<LayerArrayParams>& layer_array_params,
                                     const float head_scale, std::vector<float> weights,
                                     const double expected_sample_rate, const int num_params,
                                     std::vector<ParametricParam> param_defs)
: WaveNet(in_channels, layer_array_params, head_scale,
          /*with_head=*/false, /*head_params=*/std::nullopt,
          std::move(weights), /*condition_dsp=*/nullptr, expected_sample_rate)
, _num_params(num_params)
, _param_defs(std::move(param_defs))
{
  // Safe midpoint default: every knob starts at 0.5 so the model produces audio
  // immediately without a SetKnobValues() call from the host.
  for (int i = 0; i < _num_params && i < kMaxParams; ++i)
    _knob_atoms[i].store(0.5f, std::memory_order_relaxed);
}

void ParametricWaveNet::SetKnobValues(const std::vector<float>& values)
{
  const int n = std::min(static_cast<int>(values.size()), _num_params);
  for (int i = 0; i < n; ++i)
    _knob_atoms[i].store(values[i], std::memory_order_relaxed);
}

void ParametricWaveNet::process(NAM_SAMPLE** inputs, NAM_SAMPLE** outputs, const int num_frames)
{
  WaveNet::process(inputs, outputs, num_frames);

  // The FiLM shift terms produce a significant DC offset at any trained knob position
  // (measured ~-17 dBFS from silence at default knobs=0.5). Remove it with a first-order
  // IIR high-pass (y[n] = x[n] - x[n-1] + α·y[n-1], fc≈20 Hz) without touching audible
  // guitar content (guitar fundamentals start at ~80 Hz).
  //
  // Because DSP::prewarm() calls this->process() via virtual dispatch, the blocker
  // converges to steady state during prewarm — there is no click on the first real buffer.
  NAM_SAMPLE* out = outputs[0];
  for (int i = 0; i < num_frames; ++i)
  {
    const float x = (float)out[i];
    const float y = x - _dc_x_prev + _dc_alpha * _dc_y_prev;
    _dc_x_prev = x;
    _dc_y_prev = y;
    out[i] = (NAM_SAMPLE)y;
  }
}

void ParametricWaveNet::Reset(const double sampleRate, const int maxBufferSize)
{
  // Reset the DC blocker BEFORE calling WaveNet::Reset().
  // WaveNet::Reset() calls prewarm() which calls this->process() via virtual dispatch.
  // If we reset the blocker state AFTER prewarm() runs, we discard the converged state
  // and the blocker starts cold on the first real audio buffer — producing a long
  // decaying-DC transient ("warbly" sound that takes several seconds to settle).
  constexpr float kPi = 3.14159265358979f;
  _dc_alpha = 1.0f - (2.0f * kPi * 20.0f / (float)sampleRate);
  _dc_x_prev = 0.0f;
  _dc_y_prev = 0.0f;

  WaveNet::Reset(sampleRate, maxBufferSize); // prewarm() runs here; blocker converges
}

void ParametricWaveNet::SetMaxBufferSize(const int maxBufferSize)
{
  // Base class sizes _condition_input and _condition_output to 1×maxBufferSize
  // (because _get_condition_dim() returns 1).
  WaveNet::SetMaxBufferSize(maxBufferSize);
  // _film_condition is the knob condition: separate matrix, num_params rows.
  _film_condition.resize(_num_params, maxBufferSize);
}

void ParametricWaveNet::_process_condition(const int num_frames)
{
  // Step 1: copy audio from _condition_input (1×N) → _condition_output (1×N).
  // _condition_output is what input_mixin reads in each layer — audio mixing unchanged.
  WaveNet::_process_condition(num_frames);

  // Step 2: broadcast the current knob values into _film_condition.
  // Knobs are constant across all frames in a buffer (no sample-level interpolation).
  // _film_condition is then returned by _get_film_condition() and routed to FiLM modules
  // by WaveNet::process() → LayerArray::Process → Layer::Process.
  for (int k = 0; k < _num_params; ++k)
  {
    const float val = _knob_atoms[k].load(std::memory_order_relaxed);
    _film_condition.row(k).head(num_frames).fill(val);
  }
}

// ParametricWaveNetConfig =====================================================

std::unique_ptr<DSP> ParametricWaveNetConfig::create(std::vector<float> weights, const double sampleRate)
{
  return std::make_unique<ParametricWaveNet>(
    in_channels, layer_array_params, head_scale, std::move(weights), sampleRate, num_params, param_defs);
}

// Config parser ===============================================================

std::unique_ptr<ModelConfig> create_parametric_config(const nlohmann::json& config, const double sampleRate)
{
  // --- Channel count ---
  // config["layers"] is an INTEGER (channel count: 3 = nano, 8 = standard), NOT a layer-array list.
  // The training code named this field "layers" (matching the WaveNet convention for the layer array list)
  // but in ParametricWaveNet it means channel width. Layer depth is always kNumLayers = 23.
  if (!config.contains("layers") || !config["layers"].is_number_integer())
    throw std::runtime_error("ParametricWaveNet: config['layers'] must be an integer (channel count 3 or 8)");
  const int channels = config["layers"].get<int>();
  if (channels != 3 && channels != 8)
  {
    std::ostringstream ss;
    ss << "ParametricWaveNet: 'layers' (channel count) must be 3 (nano) or 8 (standard), got " << channels;
    throw std::runtime_error(ss.str());
  }

  // --- Head scale ---
  if (!config.contains("head_scale"))
    throw std::runtime_error("ParametricWaveNet: missing 'head_scale'");
  const float head_scale = config["head_scale"].get<float>();

  // --- Parametric conditioning block ---
  if (!config.contains("parametric"))
    throw std::runtime_error("ParametricWaveNet: missing 'parametric' block");
  const auto& param_block = config["parametric"];

  if (!param_block.contains("condition_size"))
    throw std::runtime_error("ParametricWaveNet: parametric block missing 'condition_size'");
  const int num_params = param_block["condition_size"].get<int>();
  if (num_params <= 0 || num_params > ParametricWaveNet::kMaxParams)
  {
    std::ostringstream ss;
    ss << "ParametricWaveNet: condition_size " << num_params
       << " out of range [1, " << ParametricWaveNet::kMaxParams << "]";
    throw std::runtime_error(ss.str());
  }

  std::vector<ParametricParam> param_defs;
  if (param_block.contains("parameters"))
  {
    for (const auto& p : param_block["parameters"])
    {
      ParametricParam pd;
      pd.name = p.value("name", "");
      pd.min_val = p.value("min", p.value("minimum", 0.0f));
      pd.max_val = p.value("max", p.value("maximum", 1.0f));
      pd.default_val = p.value("default", 0.5f);
      param_defs.push_back(std::move(pd));
    }
  }

  // --- Build A2-topology LayerArrayParams ---
  //
  // This produces the same 23-layer shape as a standard A2 model, except:
  //   condition_size      = 1           (audio; sizes input_mixin weight as [C, 1])
  //   film_condition_size = num_params  (knobs; sizes FiLM net weight as [2C, num_params])
  //   activation_pre_film = active      (FiLM applied after conv+mixin sum, before leaky_relu)
  //
  // All topology constants (kernel sizes, dilations, head kernel size, leaky slope) come from
  // a2_fast.h to ensure exact agreement with training and the standard A2 fast path.
  activations::ActivationConfig leaky;
  leaky.type = activations::ActivationType::LeakyReLU;
  leaky.negative_slope = a2_fast::kLeakySlope; // 0.01 — must match training

  std::vector<int> kernel_sizes(a2_fast::kKernelSizes.begin(), a2_fast::kKernelSizes.end());
  std::vector<int> dilations(a2_fast::kDilations.begin(), a2_fast::kDilations.end());
  std::vector<activations::ActivationConfig> activation_configs(a2_fast::kNumLayers, leaky);
  std::vector<GatingMode> gating_modes(a2_fast::kNumLayers, GatingMode::NONE);
  std::vector<activations::ActivationConfig> secondary_activations(a2_fast::kNumLayers,
                                                                    activations::ActivationConfig{});

  const Layer1x1Params layer1x1{true, 1};          // active, groups=1 (residual path)
  const Head1x1Params  head1x1{false, channels, 1}; // inactive (no per-layer head1x1 in A2)
  const _FiLMParams off{false, false};
  // This is the only active FiLM: applied to _z (conv+mixin sum) before leaky_relu, conditioned
  // on *film_condition (the knob values), not on condition (the audio). shift=true means the net
  // outputs both scale and shift (weight shape: [2*channels, num_params]).
  const _FiLMParams activation_pre_film{true, true, 1};

  LayerArrayParams lap(
    /*input_size=*/       1,
    /*condition_size=*/   1,                      // audio → input_mixin
    /*head_size=*/        1,                      // mono output
    /*head_kernel_size=*/ a2_fast::kHeadKernelSize,
    /*channels=*/         channels,
    /*bottleneck=*/       channels,               // no bottleneck reduction in A2
    std::move(kernel_sizes),
    std::move(dilations),
    std::move(activation_configs),
    std::move(gating_modes),
    /*head_bias=*/        true,
    /*groups_input=*/     1,
    /*groups_input_mixin=*/ 1,
    layer1x1,
    head1x1,
    std::move(secondary_activations),
    off,                   // conv_pre_film
    off,                   // conv_post_film
    off,                   // input_mixin_pre_film
    off,                   // input_mixin_post_film
    activation_pre_film,   // THE FiLM: after conv+mixin, before activation; conditioned on knobs
    off,                   // activation_post_film
    off,                   // layer1x1_post_film
    off,                   // head1x1_post_film
    /*film_condition_size=*/ num_params,            // knob count; sets FiLM condition_dim
    /*use_final_layer_output_for_head=*/ true);    // Python ParametricA2 has no skip connections

  auto pc = std::make_unique<ParametricWaveNetConfig>();
  pc->in_channels = 1;
  pc->layer_array_params.push_back(std::move(lap));
  pc->head_scale = head_scale;
  pc->num_params = num_params;
  pc->param_defs = std::move(param_defs);
  return pc;
}

// Auto-registration ===========================================================

namespace
{
// Registers "ParametricWaveNet" in ConfigParserRegistry at program startup.
// SlimmableContainer calls get_dsp(submodel_json) for each submodel in a .param_best.nam file;
// get_dsp dispatches to this factory when the submodel's architecture == "ParametricWaveNet".
static ConfigParserHelper _register_ParametricWaveNet("ParametricWaveNet", create_parametric_config);
} // namespace

} // namespace wavenet
} // namespace nam
