#pragma once

#include <vector> // [Fork] for GetAllowedSizes()

namespace nam
{

/// \brief Interface for models that support dynamic size reduction
///
/// Models implementing this interface can reduce their computational cost
/// at the expense of quality. The interpretation of the size parameter is
/// model-specific (e.g., selecting a sub-model, pruning channels, etc.).
class SlimmableModel
{
public:
  virtual ~SlimmableModel() = default;

  /// \brief Set the slimmable size of the model
  /// \param val Value between 0.0 (minimum size) and 1.0 (maximum size)
  ///
  /// Thread-safe
  /// Not real-time safe
  virtual void SetSlimmableSize(const double val) = 0;

  // [Fork] BEGIN: PrewarmForRatio()
  /// \brief Prepare (Reset()+prewarm) the submodel that SetSlimmableSize(val) would
  /// activate, WITHOUT activating it.
  ///
  /// Call this from a background thread before calling SetSlimmableSize(val) with the
  /// same value on the render thread: implementations that support staging make the
  /// subsequent SetSlimmableSize(val) cheap (a pointer/index flip) instead of doing the
  /// expensive Reset()+prewarm() work inline. Default is a no-op for implementations
  /// that don't need it (e.g. already-cheap size changes).
  ///
  /// Thread-safe. Not real-time safe (this is the method meant to absorb that cost).
  virtual void PrewarmForRatio(const double val) {}
  // [Fork] END: PrewarmForRatio()

  // [Fork] BEGIN: GetAllowedSizes()
  /// \brief The discrete sizes a caller may select between, sorted ascending
  ///
  /// The interpretation is model-specific (e.g. channel counts). Hosts use
  /// this to build a UI with one option per entry rather than assuming a
  /// fixed two-way (min/max) choice. Empty when the model exposes no
  /// discrete size list (e.g. a purely continuous or single-array model).
  virtual std::vector<int> GetAllowedSizes() const { return {}; }
  // [Fork] END: GetAllowedSizes()
};

} // namespace nam
