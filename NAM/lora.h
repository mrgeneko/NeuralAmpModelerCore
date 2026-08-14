#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <vector>

#include "compiler.h"
#include "dsp.h"

namespace nam
{
/// \brief Low-rank ("LoRA-style") knob-conditioned additive weight update
///
/// Given an input (channels x num_frames) and a condition (condition_dim x num_frames), compute:
///   A = net_A(condition) reshaped (channels x rank)   [row-major, matches Python .view()]
///   B = net_B(condition) reshaped (rank x channels)   [row-major, matches Python .view()]
///   output = A @ (B @ input)
///
/// Unlike FiLM (a diagonal per-channel affine rescale of a FIXED underlying computation), this
/// lets the conditioning genuinely change what the wrapped layer computes: conceptually
/// effective_weight = base_weight + A@B, applied here as two small matmuls on the same input
/// the wrapped layer (layer1x1) also consumes, without ever materializing the full
/// (channels x channels) delta. Must match param_train.py's `LoRA.delta()` bit-for-bit.
class LoRA
{
public:
  /// \brief Constructor
  /// \param condition_dim Size of the conditioning input
  /// \param channels Size of the input/output to be updated
  /// \param rank Bottleneck rank of the low-rank update
  LoRA(const int condition_dim, const int channels, const int rank)
  : _net_A(condition_dim, channels * rank, /*bias=*/true, /*groups=*/1)
  , _net_B(condition_dim, rank * channels, /*bias=*/true, /*groups=*/1)
  , _channels(channels)
  , _rank(rank)
  {
  }

  /// \brief Get the entire internal output buffer
  ///
  /// This is intended for internal wiring between layers; callers should treat
  /// the buffer as pre-allocated storage and only consider the first num_frames columns
  /// valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  Eigen::MatrixXf& GetOutput() { return _output; }
  const Eigen::MatrixXf& GetOutput() const { return _output; }

  /// \brief Resize buffers to handle maxBufferSize frames
  void SetMaxBufferSize(const int maxBufferSize)
  {
    _net_A.SetMaxBufferSize(maxBufferSize);
    _net_B.SetMaxBufferSize(maxBufferSize);
    _output.resize(_channels, maxBufferSize);
    // [Fork] Scratch for B @ input; preallocated so Process() never allocates.
    _bx.resize(_rank, maxBufferSize);
    _cached_condition.resize(get_condition_dim());
    _cache_valid = false;
  }

  /// \brief Set the parameters (weights) of this module
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  void set_weights_(std::vector<float>::iterator& weights)
  {
    _net_A.set_weights_(weights);
    _net_B.set_weights_(weights);
    _cache_valid = false; // [Fork] new weights change A/B for any condition
  }

  long get_condition_dim() const { return _net_A.get_in_channels(); }
  int get_channels() const { return _channels; }
  int get_rank() const { return _rank; }

  /// \brief Compute A @ (B @ input) into the internal output buffer; access via GetOutput().
  ///
  /// Uses Eigen::Ref to accept matrices and block expressions without creating temporaries
  /// (real-time safe).
  /// \param input Input matrix (channels x num_frames) -- the SAME input the wrapped layer
  ///   (e.g. layer1x1) also consumes.
  /// \param condition Conditioning matrix (condition_dim x num_frames)
  /// \param num_frames Number of frames to process
  void Process(const Eigen::Ref<const Eigen::MatrixXf>& input, const Eigen::Ref<const Eigen::MatrixXf>& condition,
               const int num_frames)
  {
    assert((int)input.rows() == _channels);
    assert(get_condition_dim() == condition.rows());
    assert(num_frames <= input.cols());
    assert(num_frames <= condition.cols());
    assert(num_frames <= _output.cols());

    // [Fork] Condition-constant cache (see film.h's Process() for the identical pattern and
    // rationale -- ParametricWaveNet broadcasts knob values identically to every frame, so
    // net_A/net_B only need to be evaluated once per block, not once per frame).
    const int cond_dim = (int)get_condition_dim();
    bool cond_constant = true;
    {
      const float* NAM_RESTRICT c0 = condition.data();
      const int cond_stride = (int)condition.outerStride();
      for (int f = 1; f < num_frames && cond_constant; f++)
      {
        const float* NAM_RESTRICT cf = c0 + f * cond_stride;
        for (int r = 0; r < cond_dim; r++)
          if (cf[r] != c0[r])
          {
            cond_constant = false;
            break;
          }
      }
    }

    if (cond_constant)
    {
      bool hit = _cache_valid;
      if (hit)
      {
        const float* NAM_RESTRICT c0 = condition.data();
        for (int r = 0; r < cond_dim; r++)
          if (c0[r] != _cached_condition[r])
          {
            hit = false;
            break;
          }
      }
      if (!hit)
      {
        _net_A.process_(condition.leftCols(1), 1); // one column, not num_frames
        _net_B.process_(condition.leftCols(1), 1);
        const float* NAM_RESTRICT c0 = condition.data();
        for (int r = 0; r < cond_dim; r++)
          _cached_condition[r] = c0[r];
        _cache_valid = true;
      }
      // Row-major reshape of the single computed column, matching Python's `.view(channels, rank)`
      // / `.view(rank, channels)` (PyTorch's default row-major/C-order layout) bit-for-bit -- a
      // single column of a column-major Eigen matrix is contiguous, so this Map is a pure
      // reinterpretation, not a copy.
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(
        _net_A.GetOutput().data(), _channels, _rank);
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B(
        _net_B.GetOutput().data(), _rank, _channels);
      _bx.leftCols(num_frames).noalias() = B * input.leftCols(num_frames);
      _output.leftCols(num_frames).noalias() = A * _bx.leftCols(num_frames);
    }
    else
    {
      _net_A.process_(condition, num_frames);
      _net_B.process_(condition, num_frames);
      _cache_valid = false; // block's A/B vary per frame; nothing single to cache
      for (int f = 0; f < num_frames; f++)
      {
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(
          _net_A.GetOutput().col(f).data(), _channels, _rank);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> B(
          _net_B.GetOutput().col(f).data(), _rank, _channels);
        _bx.col(f).noalias() = B * input.col(f);
        _output.col(f).noalias() = A * _bx.col(f);
      }
    }
  }

  /// \brief Compute A @ (B @ input) and add it into `input` in-place.
  void Process_(Eigen::Ref<Eigen::MatrixXf> input, const Eigen::Ref<const Eigen::MatrixXf>& condition,
                const int num_frames)
  {
    Process(input, condition, num_frames);
    input.leftCols(num_frames).noalias() += _output.leftCols(num_frames);
  }

private:
  Conv1x1 _net_A; // condition_dim -> channels*rank
  Conv1x1 _net_B; // condition_dim -> rank*channels
  int _channels;
  int _rank;
  Eigen::MatrixXf _output; // channels x maxBufferSize

  // [Fork] Condition-constant cache (see Process). Built once per distinct constant condition
  // and reused across blocks until the condition or the weights change.
  Eigen::MatrixXf _bx; // rank x maxBufferSize; scratch for B @ input, preallocated
  Eigen::VectorXf _cached_condition; // condition_dim; the condition the cache was built for
  bool _cache_valid = false;
};
} // namespace nam
