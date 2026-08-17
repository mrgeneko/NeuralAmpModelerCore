#pragma once

// Specialized WaveNet fast path for the A2 standard (Channels=8) and
// A2 nano (Channels=3) models. Shares the same architecture shape; only
// the channel count differs.
// [Fork] BEGIN: widened to also cover intermediate Channels=4..7
// widths. Channels 3-7 all use hand-unrolled small-GEMV kernels (Eigen's
// GEMM path carries too much overhead at these small inner dimensions);
// Channels==8 uses a generic Eigen fixed-size-matrix path (see
// a2_fast.cpp). Every "3 or 8" reference below was widened to
// "3, 4, 5, 6, 7, or 8" accordingly.
// [Fork] END
//
// When NAM_ENABLE_A2_FAST is defined at build time, wavenet::create_config
// consults is_a2_shape() on every incoming WaveNet config and, on match,
// instantiates an A2FastModel<Channels> instead of the generic WaveNet.
//
// The baseline here is correct-but-unoptimized (plain column-major loops).
// Follow-up optimizations (unrolled GEMV, tap-major nest, factored
// per-kernel-size helpers) plug into the same class.

#if defined(NAM_ENABLE_A2_FAST)

  #include <array>
  #include <memory>

  #include "../model_config.h"
  #include "json.hpp"

namespace nam
{
namespace wavenet
{
namespace a2_fast
{

/// \brief Number of layers in an A2 layer array.
constexpr int kNumLayers = 23;
/// \brief Kernel size of the layer-array head rechannel convolution.
constexpr int kHeadKernelSize = 16;
/// \brief LeakyReLU negative-slope used by every layer.
constexpr float kLeakySlope = 0.01f;

/// \brief Per-layer kernel sizes (fixed pattern shared by A2 standard + nano).
inline constexpr std::array<int, kNumLayers> kKernelSizes = {
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};

/// \brief Per-layer dilations (fixed pattern shared by A2 standard + nano).
inline constexpr std::array<int, kNumLayers> kDilations = {
  1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239};

/// \brief Strict detector: returns true iff config matches the A2 shape.
/// \param config   The "config" sub-object from a .nam WaveNet entry.
/// \param channels Out-param set to 3 (A2 nano) or 8 (A2 standard) on match.
///                 [Fork] also matches/returns 4, 5, 6, 7 (intermediate widths).
/// \return true if every architectural knob matches the A2 signature exactly.
bool is_a2_shape(const nlohmann::json& config, int* channels);

/// \brief Build a ModelConfig that instantiates the A2 fast path.
/// \pre is_a2_shape(config, ...) returned true.
std::unique_ptr<ModelConfig> create_a2_fast_config(const nlohmann::json& config, double sampleRate);

// [Fork] Parametric A2 fast path — same ring-buffer SIMD architecture as A2FastModel
// extended with per-layer FiLM conditioning. is_parametric_a2_shape detects the compact JSON
// format; create_parametric_a2_fast_config returns the config that constructs the fast model.
// Called from parametric_wavenet.cpp::create_parametric_config when NAM_ENABLE_A2_FAST is set.
/// \brief Strict detector: returns true iff config matches the ParametricWaveNet A2 shape.
///
/// Unlike is_a2_shape, this accepts the compact ParametricWaveNet JSON format where
/// config["layers"] is an integer (channel count), not an array. Checks that channels
/// are in {3, 4, 5, 6, 7, 8} and that a valid "parametric" block is present. Also accepts
/// "film+lora" (schema_version 2, config["parametric"]["lora"]["rank"] present and positive)
/// -- see LoRA in NAM/lora.h for what the extra conditioning does.
/// \param config      The "config" sub-object from a ParametricWaveNet model entry.
/// \param channels    Out-param set to 3-8 on match.
/// \param num_params  Out-param set to condition_size on match.
/// \param lora_rank   Out-param set to the LoRA rank on match if the model has a LoRA
///                    block, or 0 if it's FiLM-only. May be nullptr if not needed.
/// \return true if the config can be dispatched to the parametric A2 fast path.
bool is_parametric_a2_shape(const nlohmann::json& config, int* channels, int* num_params,
                            int* lora_rank = nullptr);

/// \brief Build a ModelConfig that instantiates the parametric A2 fast path.
/// \pre is_parametric_a2_shape(config, ...) returned true.
std::unique_ptr<ModelConfig> create_parametric_a2_fast_config(const nlohmann::json& config, double sampleRate);

} // namespace a2_fast
} // namespace wavenet
} // namespace nam

#endif // NAM_ENABLE_A2_FAST
