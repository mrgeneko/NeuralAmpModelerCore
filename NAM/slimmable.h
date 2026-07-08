#pragma once

#include <vector>

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

  /// \brief The discrete sizes a caller may select between, sorted ascending
  ///
  /// The interpretation is model-specific (e.g. channel counts). Hosts use
  /// this to build a UI with one option per entry rather than assuming a
  /// fixed two-way (min/max) choice. Empty when the model exposes no
  /// discrete size list (e.g. a purely continuous or single-array model).
  virtual std::vector<int> GetAllowedSizes() const { return {}; }
};

} // namespace nam
