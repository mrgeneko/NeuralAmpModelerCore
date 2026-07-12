#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "dsp.h"
#include "model_config.h"
#include "slimmable.h"

namespace nam
{
namespace container
{

struct Submodel
{
  double max_value;
  // [Fork] BEGIN: channels field for GetAllowedSizes()
  /// Representative model size (e.g. WaveNet channel count) for UI display via
  /// GetAllowedSizes(); 0 if the submodel's config didn't expose one.
  int channels = 0;
  // [Fork] END
  std::unique_ptr<DSP> model;
};

/// \brief A container model that holds multiple submodels at different sizes
///
/// SetSlimmableSize selects the active submodel based on the max_value thresholds.
/// Each submodel covers values up to (but not including) its max_value.
/// The last submodel is the fallback for values at or above the last threshold.
class ContainerModel : public DSP, public SlimmableModel
{
public:
  /// \brief Constructor
  /// \param submodels Vector of submodels sorted by max_value ascending
  /// \param expected_sample_rate Expected sample rate in Hz
  ContainerModel(std::vector<Submodel> submodels, const double expected_sample_rate);

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override;
  void prewarm() override;
  void Reset(const double sampleRate, const int maxBufferSize) override;
  void SetPrewarmOnReset(const bool prewarmOnReset) override;
  void SetSlimmableSize(const double val) override;
  void PrewarmForRatio(const double val) override; // [Fork]
  std::vector<int> GetAllowedSizes() const override; // [Fork]
  int GetPrewarmSamples() override;

  // [Fork] Propagate knob API through the container to its submodels.
  // All submodels in a parametric container share the same knob interface, so
  // GetNumParams/GetParameterDefs delegate to the first submodel. SetKnobValues
  // writes to ALL submodels so whichever becomes active already has the current state.
  void SetKnobValues(const std::vector<float>& values) override;
  int GetNumParams() const override;
  std::vector<DSPParamDef> GetParameterDefs() const override;

private:
  // [Fork] BEGIN: PrewarmForRatio() staging
  /// Index computation shared by SetSlimmableSize() and PrewarmForRatio().
  size_t _index_for_ratio(const double val) const;
  /// Index most recently Reset()+prewarmed by PrewarmForRatio(), or -1 if none is
  /// staged / the staged one was already consumed by SetActiveIndex.
  std::atomic<int> _prewarmed_index{-1};
  // [Fork] END

  std::vector<Submodel> _submodels;
  std::atomic<size_t> _active_index{0};
  std::mutex _slim_set_mutex;
};

// Config / registration

struct ContainerConfig : public ModelConfig
{
  nlohmann::json raw_config;
  double sample_rate;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override;
};

std::unique_ptr<ModelConfig> create_config(const nlohmann::json& config, double sampleRate);

} // namespace container
} // namespace nam
