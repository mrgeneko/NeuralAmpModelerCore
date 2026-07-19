#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <vector>

#include "compiler.h"
#include "dsp.h"

namespace nam
{
/// \brief Feature-wise Linear Modulation (FiLM)
///
/// Given an input (input_dim x num_frames) and a condition (condition_dim x num_frames), compute:
///   scale, shift = Conv1x1(condition) split across channels (top/bottom half, respectively)
///   output = input * scale + shift  (elementwise)
///
/// FiLM applies per-channel scaling and optional shifting based on conditioning input,
/// allowing the model to adapt its behavior based on external signals.
class FiLM
{
public:
  /// \brief Constructor
  /// \param condition_dim Size of the conditioning input
  /// \param input_dim Size of the input to be modulated
  /// \param shift Whether to apply both scale and shift (true) or only scale (false)
  /// \param groups Number of groups for grouped convolution in the condition-to-scale-shift submodule (default: 1)
  FiLM(const int condition_dim, const int input_dim, const bool shift, const int groups = 1)
  : _cond_to_scale_shift(condition_dim, (shift ? 2 : 1) * input_dim, /*bias=*/true, groups)
  , _do_shift(shift)
  {
  }

  /// \brief Get the entire internal output buffer
  ///
  /// This is intended for internal wiring between layers; callers should treat
  /// the buffer as pre-allocated storage and only consider the first num_frames columns
  /// valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  /// \return Reference to the output buffer
  Eigen::MatrixXf& GetOutput() { return _output; }

  /// \brief Get the entire internal output buffer (const version)
  /// \return Const reference to the output buffer
  const Eigen::MatrixXf& GetOutput() const { return _output; }

  /// \brief Resize buffers to handle maxBufferSize frames
  /// \param maxBufferSize Maximum number of frames to process in a single call
  void SetMaxBufferSize(const int maxBufferSize)
  {
    _cond_to_scale_shift.SetMaxBufferSize(maxBufferSize);
    _output.resize(get_input_dim(), maxBufferSize);
    // [Fork] Preallocate the scale/shift cache buffers so Process() never allocates.
    _scale_shift_broadcast.resize(_cond_to_scale_shift.get_out_channels(), maxBufferSize);
    _cached_condition.resize(get_condition_dim());
    _cache_valid = false;
  }

  /// \brief Set the parameters (weights) of this module
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  void set_weights_(std::vector<float>::iterator& weights)
  {
    _cond_to_scale_shift.set_weights_(weights);
    _cache_valid = false; // [Fork] new weights change scale/shift for any condition
  }

  /// \brief Get the condition dimension
  /// \return Size of the conditioning input
  long get_condition_dim() const { return _cond_to_scale_shift.get_in_channels(); }

  /// \brief Get the input dimension
  /// \return Size of the input to be modulated
  long get_input_dim() const
  {
    return _do_shift ? (_cond_to_scale_shift.get_out_channels() / 2) : _cond_to_scale_shift.get_out_channels();
  }

  /// \brief Process input with conditioning
  ///
  /// Writes (input_dim x num_frames) into internal output buffer; access via GetOutput().
  /// Uses Eigen::Ref to accept matrices and block expressions without creating temporaries (real-time safe).
  /// \param input Input matrix (input_dim x num_frames)
  /// \param condition Conditioning matrix (condition_dim x num_frames)
  /// \param num_frames Number of frames to process
  void Process(const Eigen::Ref<const Eigen::MatrixXf>& input, const Eigen::Ref<const Eigen::MatrixXf>& condition,
               const int num_frames)
  {
    assert(get_input_dim() == input.rows());
    assert(get_condition_dim() == condition.rows());
    assert(num_frames <= input.cols());
    assert(num_frames <= condition.cols());
    assert(num_frames <= _output.cols());

    // [Fork] Scale/shift cache ----------------------------------------------------------
    // The condition->scale/shift Conv1x1 is the expensive part of FiLM. When the conditioning is
    // constant across the block -- the ParametricWaveNet case, where knob values are broadcast
    // identically to every frame -- that Conv1x1 yields the SAME column for all num_frames, so
    // recomputing it per frame was pure waste. Detect that, compute the column ONCE, broadcast it,
    // and skip even the single recompute when the condition is unchanged since the previous block.
    // Time-varying conditioning (standard/audio FiLM) takes the original per-frame path, and the
    // constant path is bit-identical to it (same Conv1x1 on the same column, replicated), so output
    // is numerically unchanged either way.
    const int cond_dim = (int)get_condition_dim();
    bool cond_constant = true;
    {
      const float* NAM_RESTRICT c0 = condition.data();
      const int cond_stride = (int)condition.outerStride();
      for (int f = 1; f < num_frames && cond_constant; f++)
      {
        const float* __restrict cf = c0 + f * cond_stride;
        for (int r = 0; r < cond_dim; r++)
          if (cf[r] != c0[r])
          {
            cond_constant = false;
            break;
          }
      }
    }

    const Eigen::MatrixXf* scale_shift_src;
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
        _cond_to_scale_shift.process_(condition.leftCols(1), 1); // one column, not num_frames
        // Broadcast the single column so the affine below (which reads one column per frame) is
        // unchanged; every column is bit-identical to what the per-frame Conv1x1 would produce.
        _scale_shift_broadcast.colwise() = _cond_to_scale_shift.GetOutput().col(0);
        const float* NAM_RESTRICT c0 = condition.data();
        for (int r = 0; r < cond_dim; r++)
          _cached_condition[r] = c0[r];
        _cache_valid = true;
      }
      scale_shift_src = &_scale_shift_broadcast;
    }
    else
    {
      _cond_to_scale_shift.process_(condition, num_frames);
      _cache_valid = false; // block's scale/shift vary per frame; nothing single to cache
      scale_shift_src = &_cond_to_scale_shift.GetOutput();
    }
    const auto& scale_shift = *scale_shift_src;

#ifdef NAM_USE_INLINE_GEMM
    // Optimized inline FiLM operation
    const int input_dim = (int)get_input_dim();
    const float* NAM_RESTRICT input_ptr = input.data();
    const float* NAM_RESTRICT scale_shift_ptr = scale_shift.data();
    float* NAM_RESTRICT output_ptr = _output.data();
    const int scale_shift_rows = (int)scale_shift.rows();
    // Use outerStride() instead of rows() to correctly handle non-contiguous
    // block expressions (e.g. topRows()) where outerStride > rows
    const int input_stride = (int)input.outerStride();

    if (_do_shift)
    {
      // scale = top input_dim rows, shift = bottom input_dim rows
      if (input_dim == 3)
      {
        for (int f = 0; f < num_frames; f++)
        {
          const float* NAM_RESTRICT in_col = input_ptr + f * input_stride;
          const float* NAM_RESTRICT scale_col = scale_shift_ptr + f * scale_shift_rows;
          const float* NAM_RESTRICT shift_col = scale_col + 3;
          float* NAM_RESTRICT out_col = output_ptr + f * 3;
          out_col[0] = in_col[0] * scale_col[0] + shift_col[0];
          out_col[1] = in_col[1] * scale_col[1] + shift_col[1];
          out_col[2] = in_col[2] * scale_col[2] + shift_col[2];
        }
      }
      else
      {
        for (int f = 0; f < num_frames; f++)
        {
          const float* NAM_RESTRICT in_col = input_ptr + f * input_stride;
          const float* NAM_RESTRICT scale_col = scale_shift_ptr + f * scale_shift_rows;
          const float* NAM_RESTRICT shift_col = scale_col + input_dim;
          float* NAM_RESTRICT out_col = output_ptr + f * input_dim;

          int i = 0;
          for (; i + 3 < input_dim; i += 4)
          {
            out_col[i] = in_col[i] * scale_col[i] + shift_col[i];
            out_col[i + 1] = in_col[i + 1] * scale_col[i + 1] + shift_col[i + 1];
            out_col[i + 2] = in_col[i + 2] * scale_col[i + 2] + shift_col[i + 2];
            out_col[i + 3] = in_col[i + 3] * scale_col[i + 3] + shift_col[i + 3];
          }
          for (; i < input_dim; i++)
          {
            out_col[i] = in_col[i] * scale_col[i] + shift_col[i];
          }
        }
      }
    }
    else
    {
      // scale only
      if (input_dim == 3)
      {
        for (int f = 0; f < num_frames; f++)
        {
          const float* NAM_RESTRICT in_col = input_ptr + f * input_stride;
          const float* NAM_RESTRICT scale_col = scale_shift_ptr + f * scale_shift_rows;
          float* NAM_RESTRICT out_col = output_ptr + f * 3;
          out_col[0] = in_col[0] * scale_col[0];
          out_col[1] = in_col[1] * scale_col[1];
          out_col[2] = in_col[2] * scale_col[2];
        }
      }
      else
      {
        for (int f = 0; f < num_frames; f++)
        {
          const float* NAM_RESTRICT in_col = input_ptr + f * input_stride;
          const float* NAM_RESTRICT scale_col = scale_shift_ptr + f * scale_shift_rows;
          float* NAM_RESTRICT out_col = output_ptr + f * input_dim;

          int i = 0;
          for (; i + 3 < input_dim; i += 4)
          {
            out_col[i] = in_col[i] * scale_col[i];
            out_col[i + 1] = in_col[i + 1] * scale_col[i + 1];
            out_col[i + 2] = in_col[i + 2] * scale_col[i + 2];
            out_col[i + 3] = in_col[i + 3] * scale_col[i + 3];
          }
          for (; i < input_dim; i++)
          {
            out_col[i] = in_col[i] * scale_col[i];
          }
        }
      }
    }
#else
    const auto scale = scale_shift.topRows(get_input_dim()).leftCols(num_frames);
    if (_do_shift)
    {
      // scale = top input_dim, shift = bottom input_dim
      const auto shift = scale_shift.bottomRows(get_input_dim()).leftCols(num_frames);
      _output.leftCols(num_frames).array() = input.leftCols(num_frames).array() * scale.array() + shift.array();
    }
    else
    {
      _output.leftCols(num_frames).array() = input.leftCols(num_frames).array() * scale.array();
    }
#endif
  }

  /// \brief Process input with conditioning (in-place)
  ///
  /// Uses Eigen::Ref to accept matrices and block expressions without creating temporaries (real-time safe).
  /// Modifies the input matrix directly.
  /// \param input Input matrix (input_dim x num_frames), will be modified in-place
  /// \param condition Conditioning matrix (condition_dim x num_frames)
  /// \param num_frames Number of frames to process
  void Process_(Eigen::Ref<Eigen::MatrixXf> input, const Eigen::Ref<const Eigen::MatrixXf>& condition,
                const int num_frames)
  {
    Process(input, condition, num_frames);
    input.leftCols(num_frames).noalias() = _output.leftCols(num_frames);
  }

private:
  Conv1x1 _cond_to_scale_shift; // condition_dim -> (shift ? 2 : 1) * input_dim
  Eigen::MatrixXf _output; // input_dim x maxBufferSize
  bool _do_shift;

  // [Fork] Scale/shift cache (see Process). Built once per distinct constant condition and
  // reused across blocks until the condition or the weights change.
  Eigen::MatrixXf _scale_shift_broadcast; // (shift?2:1)*input_dim x maxBufferSize; each col identical
  Eigen::VectorXf _cached_condition; // condition_dim; the condition the cache was built for
  bool _cache_valid = false;
};
} // namespace nam
