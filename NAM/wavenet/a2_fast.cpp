#if defined(NAM_ENABLE_A2_FAST)

  // Ring-buffer strategy:
  //   0 = linear memmove-rewind (variable worst-case latency, sporadic spikes)
  //   1 = pow2 + tail mirror (constant per-block work, branchless reads)
  // Controlled externally with -DNAM_A2_RING_MODE=0 for head-to-head comparison.
  #ifndef NAM_A2_RING_MODE
    #define NAM_A2_RING_MODE 1
  #endif

  #include "a2_fast.h"
  #include "parametric_wavenet.h"

  #include <algorithm>
  #include <array>
  #include <cmath>
  #include <cstddef>
  #include <cstring>
  #include <iterator>
  #include <memory>
  #include <sstream>
  #include <stdexcept>
  #include <string>
  #include <utility>
  #include <vector>

  #include <Eigen/Dense>

  #include "../dsp.h"

namespace nam
{
namespace wavenet
{
namespace a2_fast
{

namespace
{

// =============================================================================
// A2FastModel<Channels>
//
// Skeleton implementation: correct but not yet optimized.
//
// Architectural invariants (checked once by is_a2_shape before we get here):
//   - single layer array with 23 layers
//   - Bottleneck == Channels
//   - condition_size == input_size == out_channels == 1
//   - LeakyReLU(0.01) on every layer, no gating, no FiLM, no head1x1
//   - layer1x1 active (groups=1), head rechannel conv k=16 bias=true
//   - no post-stack head
//
// Weight storage: column-major per kernel tap. For a (out_ch × in_ch) matrix
// at tap k, element (row=i, col=j) lives at w[k][j * out_ch + i]. All 1×1
// and K×1 convolutions follow the same convention (with K = 1 for 1×1).
// =============================================================================
template <int Channels>
class A2FastModel : public DSP
{
public:
  static constexpr int kChannels = Channels;
  static constexpr int kBottleneck = Channels;
  static constexpr int kHeadIn = Channels;

  A2FastModel(std::vector<float> weights, double expected_sample_rate);
  ~A2FastModel() override = default;

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override;
  int GetPrewarmSamples() override { return _prewarm_samples; }

protected:
  void SetMaxBufferSize(int maxBufferSize) override;

private:
  struct Layer
  {
    int kernel_size = 0;
    int dilation = 0;
    int max_lookback = 0; // (kernel_size - 1) * dilation

    // Dilated conv (Channels -> Bottleneck), column-major per tap.
    // Flat size = kernel_size * Channels * Bottleneck.
    std::vector<float> conv_w;
    std::array<float, Channels> conv_b{};

    // Input mixin (cond_size=1 -> Bottleneck), no bias.
    std::array<float, Channels> mixin_w{};

    // layer1x1 (Bottleneck -> Channels), with bias. Column-major (Channels × Bottleneck).
    std::array<float, Channels * Channels> l1x1_w{};
    std::array<float, Channels> l1x1_b{};

    // Conv1D input history ring buffer, column-major (Channels rows).
    std::vector<float> history;
  #if NAM_A2_RING_MODE == 1
    // pow2 ring + tail mirror. Storage = (pow2_size + max_buffer_size) cols.
    // write_pos is kept in [0, pow2_size), reads use (pos & pow2_mask) and are
    // always contiguous because cols [pow2_size, pow2_size + max_buffer_size)
    // mirror cols [0, max_buffer_size).
    int pow2_size = 0;
    int pow2_mask = 0;
    int write_pos = 0;
  #else
    // Linear ring with sporadic memmove-rewind. history_cols = 2*max_lookback +
    // max_buffer_size; write_pos grows monotonically until rewind fires.
    int history_cols = 0;
    int write_pos = 0;
  #endif
  };

  std::array<Layer, kNumLayers> _layers;

  // Rechannel (input_size=1 -> Channels), no bias.
  std::array<float, Channels> _rechannel_w{};

  // Head rechannel (Bottleneck -> 1), kernel=16, bias. Column-major per tap.
  // At each tap, matrix is (1 × Channels) col-major -> Channels floats.
  std::array<std::array<float, Channels>, kHeadKernelSize> _head_w{};
  float _head_b = 0.0f;

  // Head scale is stored as the trailing float in the weights stream (the generic
  // WaveNet reads it the same way, overriding the JSON head_scale field).
  float _head_scale = 1.0f;

  // Head ring buffer (Channels rows, col-major). Same ring layout as per-layer.
  std::vector<float> _head_history;
  #if NAM_A2_RING_MODE == 1
  int _head_pow2_size = 0;
  int _head_pow2_mask = 0;
  int _head_write_pos = 0;
  #else
  int _head_history_cols = 0;
  int _head_write_pos = 0;
  #endif

  // Working buffers (all Channels rows, max_buffer_size cols, col-major).
  // Eigen::aligned_allocator guarantees 16-byte alignment so the Eigen::Map<..., Eigen::Aligned16>
  // views constructed over these in the Channels==8 path (below) can use Eigen's fast aligned
  // SIMD path instead of falling back to its unaligned-safe (and much slower) path. Plain
  // std::vector<float> has no such guarantee -- see docs/Architecture/NAM_PERFORMANCE_INVESTIGATION.md.
  std::vector<float, Eigen::aligned_allocator<float>> _layer_in; // current layer input / next layer input (in-place residual)
  std::vector<float, Eigen::aligned_allocator<float>> _head_sum; // accumulates activations across all layers
  std::vector<float, Eigen::aligned_allocator<float>> _z; // per-layer conv output accumulator (tap-major)
  std::vector<float> _cond; // float32 copy of the double NAM_SAMPLE input, reused each block
  std::vector<float> _head_out; // float32 head output before writing to NAM_SAMPLE

  int _prewarm_samples = 0;

  void _load_weights(std::vector<float>& weights);
  void _ring_write(Layer& L, int num_frames);
  void _head_ring_write(int num_frames);
  void _layer_forward(int layer_idx, const float* cond, int num_frames);
  void _head_forward(float* output, int num_frames);

  // Compile-time-specialized per-layer kernel. KernelSize is lifted to a
  // template parameter so clang can fully unroll the tap loop and schedule
  // FMAs across taps. For the A2 shape we only need K=6 and K=15.
  template <int KernelSize>
  void _layer_forward_k(Layer& L, const float* cond, int num_frames);
};

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------
template <int Channels>
A2FastModel<Channels>::A2FastModel(std::vector<float> weights, double expected_sample_rate)
: DSP(/*in_channels=*/1, /*out_channels=*/1, expected_sample_rate)
{
  for (int i = 0; i < kNumLayers; i++)
  {
    _layers[i].kernel_size = kKernelSizes[i];
    _layers[i].dilation = kDilations[i];
    _layers[i].max_lookback = (kKernelSizes[i] - 1) * kDilations[i];
    _layers[i].conv_w.assign(static_cast<size_t>(kKernelSizes[i]) * Channels * Channels, 0.0f);
  }

  _load_weights(weights);

  // Receptive field = 1 (the sample being produced) + sum of per-layer lookbacks +
  // (head kernel - 1). The leading 1 matches the generic WaveNet's prewarm count
  // (model.cpp: mPrewarmSamples starts at 1 when there's no condition DSP), so the
  // fast path warms up by exactly the same number of samples as the model it replaces.
  int prewarm = 1;
  for (int i = 0; i < kNumLayers; i++)
    prewarm += _layers[i].max_lookback;
  prewarm += kHeadKernelSize - 1;
  _prewarm_samples = prewarm;
}

// -----------------------------------------------------------------------------
// Weight loader
//
// Reproduces the generic path's weight-reading order exactly:
//   - LayerArray::set_weights_:
//       _rechannel (Conv1x1 1 -> Channels, no bias)
//       for each layer:
//           _conv (Conv1D Channels -> Bottleneck, K × C × B + B bias)
//           _input_mixin (Conv1x1 1 -> Bottleneck, no bias)
//           _layer1x1 (Conv1x1 Bottleneck -> Channels, with bias)
//       _head_rechannel (Conv1D Bottleneck -> 1, K=16, bias)
//
// Generic Conv1D loader order: for i in out_ch: for j in in_ch: for k in taps.
// Generic Conv1x1 loader order: for i in out_ch: for j in in_ch.
// We permute into column-major per-tap storage while reading.
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::_load_weights(std::vector<float>& weights)
{
  auto it = weights.begin();
  const auto end = weights.end();

  auto take = [&]() -> float {
    if (it == end)
      throw std::runtime_error("A2FastModel: weight stream exhausted");
    return *it++;
  };

  // Rechannel: 1 -> Channels, no bias. Read order: for i in Channels: for j in 1.
  for (int i = 0; i < Channels; i++)
    _rechannel_w[i] = take();

  for (int li = 0; li < kNumLayers; li++)
  {
    Layer& L = _layers[li];
    const int K = L.kernel_size;

    // Conv1D: Channels -> Bottleneck, kernel=K, bias.
    // Read order: for i in Bottleneck: for j in Channels: for k in K.
    // Store at conv_w[k * C * B + j * B + i] (col-major (B × C) per tap).
    for (int i = 0; i < Channels; i++) // row (out)
    {
      for (int j = 0; j < Channels; j++) // col (in)
      {
        for (int k = 0; k < K; k++)
        {
          L.conv_w[k * Channels * Channels + j * Channels + i] = take();
        }
      }
    }
    for (int i = 0; i < Channels; i++)
      L.conv_b[i] = take();

    // Input mixin: 1 -> Bottleneck, no bias. Read order: for i in Bottleneck: for j in 1.
    for (int i = 0; i < Channels; i++)
      L.mixin_w[i] = take();

    // layer1x1: Bottleneck -> Channels, with bias. Read order: for i in Channels: for j in Bottleneck.
    // Store at l1x1_w[j * Channels + i] (col-major Channels × Bottleneck).
    for (int i = 0; i < Channels; i++) // row (out = Channels)
    {
      for (int j = 0; j < Channels; j++) // col (in = Bottleneck)
      {
        L.l1x1_w[j * Channels + i] = take();
      }
    }
    for (int i = 0; i < Channels; i++)
      L.l1x1_b[i] = take();
  }

  // Head rechannel: Bottleneck -> 1, kernel=16, bias.
  // Read order: for i in 1: for j in Bottleneck: for k in 16.
  // Store at _head_w[k][j] (row=0 since out=1, column-major => just Channels floats per tap).
  for (int j = 0; j < Channels; j++)
  {
    for (int k = 0; k < kHeadKernelSize; k++)
    {
      _head_w[k][j] = take();
    }
  }
  _head_b = take();

  // Matches WaveNet::set_weights_: the last value in the stream is head_scale.
  _head_scale = take();

  if (it != end)
  {
    std::stringstream ss;
    ss << "A2FastModel: weight stream has " << std::distance(it, end) << " trailing bytes";
    throw std::runtime_error(ss.str());
  }
}

// -----------------------------------------------------------------------------
// Buffer sizing
// -----------------------------------------------------------------------------
namespace
{
// Smallest power of 2 >= v (v > 0).
int next_pow2(int v)
{
  int p = 1;
  while (p < v)
    p <<= 1;
  return p;
}
} // namespace

template <int Channels>
void A2FastModel<Channels>::SetMaxBufferSize(int maxBufferSize)
{
  DSP::SetMaxBufferSize(maxBufferSize);

  _layer_in.assign(static_cast<size_t>(Channels) * maxBufferSize, 0.0f);
  _head_sum.assign(static_cast<size_t>(Channels) * maxBufferSize, 0.0f);
  _z.assign(static_cast<size_t>(Channels) * maxBufferSize, 0.0f);
  _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
  _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);

  for (auto& L : _layers)
  {
  #if NAM_A2_RING_MODE == 1
    L.pow2_size = next_pow2(L.max_lookback + maxBufferSize);
    L.pow2_mask = L.pow2_size - 1;
    L.history.assign(static_cast<size_t>(Channels) * (L.pow2_size + maxBufferSize), 0.0f);
    L.write_pos = L.max_lookback;
  #else
    L.history_cols = 2 * L.max_lookback + maxBufferSize;
    L.history.assign(static_cast<size_t>(Channels) * L.history_cols, 0.0f);
    L.write_pos = L.max_lookback;
  #endif
  }

  const int head_lookback = kHeadKernelSize - 1;
  #if NAM_A2_RING_MODE == 1
  _head_pow2_size = next_pow2(head_lookback + maxBufferSize);
  _head_pow2_mask = _head_pow2_size - 1;
  _head_history.assign(static_cast<size_t>(Channels) * (_head_pow2_size + maxBufferSize), 0.0f);
  _head_write_pos = head_lookback;
  #else
  _head_history_cols = 2 * head_lookback + maxBufferSize;
  _head_history.assign(static_cast<size_t>(Channels) * _head_history_cols, 0.0f);
  _head_write_pos = head_lookback;
  #endif
}

// -----------------------------------------------------------------------------
// Ring-write helpers.
//   Mode 1: pow2 + tail mirror. Constant-time per block (one short memcpy
//   into the ring, one mirror refresh).
//   Mode 0: linear with periodic memmove rewind. When write_pos nears the
//   end of history, memmove the trailing max_lookback cols back to offset 0
//   and reset write_pos. That memmove is the jitter spike we're measuring.
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::_ring_write(Layer& L, int num_frames)
{
  #if NAM_A2_RING_MODE == 1
  const int mbs = GetMaxBufferSize();
  float* const hist = L.history.data();
  const float* const src = _layer_in.data();
  const int wp = L.write_pos;
  const int first = std::min(num_frames, L.pow2_size - wp);
  std::memcpy(hist + static_cast<size_t>(wp) * Channels, src, static_cast<size_t>(first) * Channels * sizeof(float));
  if (first < num_frames)
  {
    std::memcpy(hist, src + static_cast<size_t>(first) * Channels,
                static_cast<size_t>(num_frames - first) * Channels * sizeof(float));
  }
  std::memcpy(
    hist + static_cast<size_t>(L.pow2_size) * Channels, hist, static_cast<size_t>(mbs) * Channels * sizeof(float));
  L.write_pos = (wp + num_frames) & L.pow2_mask;
  #else
  if (L.write_pos + num_frames > L.history_cols)
  {
    const int keep = L.max_lookback;
    std::memmove(L.history.data(), L.history.data() + static_cast<size_t>(L.write_pos - keep) * Channels,
                 static_cast<size_t>(keep) * Channels * sizeof(float));
    L.write_pos = keep;
  }
  std::memcpy(L.history.data() + static_cast<size_t>(L.write_pos) * Channels, _layer_in.data(),
              static_cast<size_t>(num_frames) * Channels * sizeof(float));
  L.write_pos += num_frames;
  #endif
}

template <int Channels>
void A2FastModel<Channels>::_head_ring_write(int num_frames)
{
  #if NAM_A2_RING_MODE == 1
  const int mbs = GetMaxBufferSize();
  float* const hist = _head_history.data();
  const float* const src = _head_sum.data();
  const int wp = _head_write_pos;
  const int first = std::min(num_frames, _head_pow2_size - wp);
  std::memcpy(hist + static_cast<size_t>(wp) * Channels, src, static_cast<size_t>(first) * Channels * sizeof(float));
  if (first < num_frames)
  {
    std::memcpy(hist, src + static_cast<size_t>(first) * Channels,
                static_cast<size_t>(num_frames - first) * Channels * sizeof(float));
  }
  std::memcpy(
    hist + static_cast<size_t>(_head_pow2_size) * Channels, hist, static_cast<size_t>(mbs) * Channels * sizeof(float));
  _head_write_pos = (wp + num_frames) & _head_pow2_mask;
  #else
  const int keep = kHeadKernelSize - 1;
  if (_head_write_pos + num_frames > _head_history_cols)
  {
    std::memmove(_head_history.data(), _head_history.data() + static_cast<size_t>(_head_write_pos - keep) * Channels,
                 static_cast<size_t>(keep) * Channels * sizeof(float));
    _head_write_pos = keep;
  }
  std::memcpy(_head_history.data() + static_cast<size_t>(_head_write_pos) * Channels, _head_sum.data(),
              static_cast<size_t>(num_frames) * Channels * sizeof(float));
  _head_write_pos += num_frames;
  #endif
}

// -----------------------------------------------------------------------------
// Per-layer forward pass. Reads current _layer_in, writes back into _layer_in
// after applying dilated conv + mixin + LeakyReLU + layer1x1 residual, and
// accumulates activations into _head_sum.
// -----------------------------------------------------------------------------
// Compile-time-specialized per-layer kernel. KernelSize is a template param
// so the K tap loop + per-tap weight offsets become compile-time constants;
// clang fully unrolls and can schedule FMAs across taps. Called from the
// runtime dispatcher below for each A2 kernel size (6 and 15).
template <int Channels>
template <int KernelSize>
void A2FastModel<Channels>::_layer_forward_k(Layer& L, const float* cond, int num_frames)
{
  constexpr int K = KernelSize;
  const int D = L.dilation;
  // Physical ring position of this block's first frame, offset by `taps_back *
  // D` samples into the past. In pow2 mode the position is wrapped by mask and
  // reads spanning the wrap land in the tail mirror; in linear mode write_pos
  // is monotonic and arithmetic is plain.
  #if NAM_A2_RING_MODE == 1
  const int mask = L.pow2_mask;
  auto tap_base_phys = [&](int taps_back) { return (L.write_pos - num_frames - taps_back * D) & mask; };
  #else
  const int base = L.write_pos - num_frames;
  auto tap_base_phys = [&](int taps_back) { return base - taps_back * D; };
  #endif

  // Two conv strategies, dispatched at compile time on Channels:
  //
  //   - Channels <= 4 (A2 nano): full-block tap-major. The z accumulator lives
  //     in the heap buffer across all taps, and for each tap the inner f-loop
  //     iterates over all num_frames. This gives clang frame-level
  //     parallelism — it vectorizes across 4 frames at a time, which matters
  //     more than weight-reload cost when the b-loop (3 wide) can't saturate
  //     NEON lanes on its own.
  //
  //   - Channels >= 8 (A2 standard): frame-tiled tap-major with T=4. ztile
  //     stays in NEON registers across all K taps, amortizing weight loads
  //     over 4 frames — equivalent to what a GEMM kernel does. Weight reuse
  //     matters here because the b-loop (8 wide) already saturates SIMD, so
  //     frame-level parallelism gives no extra headroom. The 1x1 residual is
  //     also tiled over the same T=4 frames so W1x1 loads are amortized.

  if constexpr (Channels == 3)
  {
    // Inner 3x3 GEMV fully unrolled: all 9 weights lifted into named consts
    // before the frame loop, the c-reduction kept in scalar temps a0/a1/a2 so
    // the compiler keeps them in FP registers across the frame loop. Mirrors
    // the nam2c --fused structure.
    float* z = _z.data();

    // Tap 0: seed z with conv_b (saves the memset-to-zero pass) and fold in
    // the first tap's FMAs.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2];
      const float w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 3];
        float a0 = cb0 + w0 * src[0];
        float a1 = cb1 + w1 * src[0];
        float a2 = cb2 + w2 * src[0];
        a0 += w3 * src[1];
        a1 += w4 * src[1];
        a2 += w5 * src[1];
        a0 += w6 * src[2];
        a1 += w7 * src[2];
        a2 += w8 * src[2];
        float* zf = z + static_cast<size_t>(f) * 3;
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
      }
    }

    // Taps 1..K-2: accumulate into z with the same unrolled inner kernel.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 9];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2];
      const float w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 3];
        float* zf = z + static_cast<size_t>(f) * 3;
        float a0 = zf[0] + w0 * src[0];
        float a1 = zf[1] + w1 * src[0];
        float a2 = zf[2] + w2 * src[0];
        a0 += w3 * src[1];
        a1 += w4 * src[1];
        a2 += w5 * src[1];
        a0 += w6 * src[2];
        a1 += w7 * src[2];
        a2 += w8 * src[2];
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
      }
    }

    // Final tap (K-1, offset 0) fully inlined with the post-conv tail.
    // Everything runs on register-resident scalars:
    //   conv tap K-1 -> mixin -> LeakyReLU -> head_sum += -> layer1x1 residual.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 9];
    const int tap_base_last = tap_base_phys(0);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2];
    const float cw3 = wk_last[3], cw4 = wk_last[4], cw5 = wk_last[5];
    const float cw6 = wk_last[6], cw7 = wk_last[7], cw8 = wk_last[8];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2];
    // layer1x1 col-major: lw[b*3 + c] is weight from bottleneck b to output c.
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2];
    const float lw10 = L.l1x1_w[3], lw11 = L.l1x1_w[4], lw12 = L.l1x1_w[5];
    const float lw20 = L.l1x1_w[6], lw21 = L.l1x1_w[7], lw22 = L.l1x1_w[8];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 3];
      const float* zf_mem = z + static_cast<size_t>(f) * 3;
      // Final tap GEMV.
      float a0 = zf_mem[0] + cw0 * src[0];
      float a1 = zf_mem[1] + cw1 * src[0];
      float a2 = zf_mem[2] + cw2 * src[0];
      a0 += cw3 * src[1];
      a1 += cw4 * src[1];
      a2 += cw5 * src[1];
      a0 += cw6 * src[2];
      a1 += cw7 * src[2];
      a2 += cw8 * src[2];
      // Mixin + LeakyReLU.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      // Head sum accumulate.
      float* hsum = &_head_sum[static_cast<size_t>(f) * 3];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 3];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2;
    }
  }
  // [Fork] BEGIN: hand-unrolled Channels==4 kernel, mirrors the Channels==3 branch
  // above (same tap-major full-block strategy) widened from a 3x3 GEMV (9 weights/tap) to
  // a 4x4 GEMV (16 weights/tap, 4 accumulators). Weight indexing is the same column-major
  // convention: w[j*Channels + i] for output row i, input col j. Verified numerically
  // against the generic WaveNet reference by test_matches_generic_medium.
  else if constexpr (Channels == 4)
  {
    float* z = _z.data();

    // Tap 0: seed z with conv_b (saves the memset-to-zero pass) and fold in
    // the first tap's FMAs.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3];
      const float w4 = wk[4], w5 = wk[5], w6 = wk[6], w7 = wk[7];
      const float w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11];
      const float w12 = wk[12], w13 = wk[13], w14 = wk[14], w15 = wk[15];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2], cb3 = L.conv_b[3];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 4];
        float a0 = cb0 + w0 * src[0];
        float a1 = cb1 + w1 * src[0];
        float a2 = cb2 + w2 * src[0];
        float a3 = cb3 + w3 * src[0];
        a0 += w4 * src[1];
        a1 += w5 * src[1];
        a2 += w6 * src[1];
        a3 += w7 * src[1];
        a0 += w8 * src[2];
        a1 += w9 * src[2];
        a2 += w10 * src[2];
        a3 += w11 * src[2];
        a0 += w12 * src[3];
        a1 += w13 * src[3];
        a2 += w14 * src[3];
        a3 += w15 * src[3];
        float* zf = z + static_cast<size_t>(f) * 4;
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
        zf[3] = a3;
      }
    }

    // Taps 1..K-2: accumulate into z with the same unrolled inner kernel.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 16];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3];
      const float w4 = wk[4], w5 = wk[5], w6 = wk[6], w7 = wk[7];
      const float w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11];
      const float w12 = wk[12], w13 = wk[13], w14 = wk[14], w15 = wk[15];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 4];
        float* zf = z + static_cast<size_t>(f) * 4;
        float a0 = zf[0] + w0 * src[0];
        float a1 = zf[1] + w1 * src[0];
        float a2 = zf[2] + w2 * src[0];
        float a3 = zf[3] + w3 * src[0];
        a0 += w4 * src[1];
        a1 += w5 * src[1];
        a2 += w6 * src[1];
        a3 += w7 * src[1];
        a0 += w8 * src[2];
        a1 += w9 * src[2];
        a2 += w10 * src[2];
        a3 += w11 * src[2];
        a0 += w12 * src[3];
        a1 += w13 * src[3];
        a2 += w14 * src[3];
        a3 += w15 * src[3];
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
        zf[3] = a3;
      }
    }

    // Final tap (K-1, offset 0) fully inlined with the post-conv tail.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 16];
    const int tap_base_last = tap_base_phys(0);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2], cw3 = wk_last[3];
    const float cw4 = wk_last[4], cw5 = wk_last[5], cw6 = wk_last[6], cw7 = wk_last[7];
    const float cw8 = wk_last[8], cw9 = wk_last[9], cw10 = wk_last[10], cw11 = wk_last[11];
    const float cw12 = wk_last[12], cw13 = wk_last[13], cw14 = wk_last[14], cw15 = wk_last[15];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2], mw3 = L.mixin_w[3];
    // layer1x1 col-major: lw[b*4 + c] is weight from bottleneck b to output c.
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2], lw03 = L.l1x1_w[3];
    const float lw10 = L.l1x1_w[4], lw11 = L.l1x1_w[5], lw12 = L.l1x1_w[6], lw13 = L.l1x1_w[7];
    const float lw20 = L.l1x1_w[8], lw21 = L.l1x1_w[9], lw22 = L.l1x1_w[10], lw23 = L.l1x1_w[11];
    const float lw30 = L.l1x1_w[12], lw31 = L.l1x1_w[13], lw32 = L.l1x1_w[14], lw33 = L.l1x1_w[15];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2], lb3 = L.l1x1_b[3];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 4];
      const float* zf_mem = z + static_cast<size_t>(f) * 4;
      // Final tap GEMV.
      float a0 = zf_mem[0] + cw0 * src[0];
      float a1 = zf_mem[1] + cw1 * src[0];
      float a2 = zf_mem[2] + cw2 * src[0];
      float a3 = zf_mem[3] + cw3 * src[0];
      a0 += cw4 * src[1];
      a1 += cw5 * src[1];
      a2 += cw6 * src[1];
      a3 += cw7 * src[1];
      a0 += cw8 * src[2];
      a1 += cw9 * src[2];
      a2 += cw10 * src[2];
      a3 += cw11 * src[2];
      a0 += cw12 * src[3];
      a1 += cw13 * src[3];
      a2 += cw14 * src[3];
      a3 += cw15 * src[3];
      // Mixin + LeakyReLU.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a3 += mw3 * cf;
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      a3 = (a3 >= 0.0f) ? a3 : a3 * kLeakySlope;
      // Head sum accumulate.
      float* hsum = &_head_sum[static_cast<size_t>(f) * 4];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      hsum[3] += a3;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 4];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2 + lw30 * a3;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2 + lw31 * a3;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2 + lw32 * a3;
      lin[3] += lb3 + lw03 * a0 + lw13 * a1 + lw23 * a2 + lw33 * a3;
    }
  }
  // [Fork] END: hand-unrolled Channels==4 kernel
  else if constexpr (Channels == 5)
  {
    float* z = _z.data();

    // Tap 0: seed z with conv_b (saves the memset-to-zero pass) and fold in
    // the first tap's FMAs.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4];
      const float w5 = wk[5], w6 = wk[6], w7 = wk[7], w8 = wk[8], w9 = wk[9];
      const float w10 = wk[10], w11 = wk[11], w12 = wk[12], w13 = wk[13], w14 = wk[14];
      const float w15 = wk[15], w16 = wk[16], w17 = wk[17], w18 = wk[18], w19 = wk[19];
      const float w20 = wk[20], w21 = wk[21], w22 = wk[22], w23 = wk[23], w24 = wk[24];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2], cb3 = L.conv_b[3], cb4 = L.conv_b[4];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 5];
        float a0 = cb0 + w0 * src[0];
        float a1 = cb1 + w1 * src[0];
        float a2 = cb2 + w2 * src[0];
        float a3 = cb3 + w3 * src[0];
        float a4 = cb4 + w4 * src[0];
        a0 += w5 * src[1];
        a1 += w6 * src[1];
        a2 += w7 * src[1];
        a3 += w8 * src[1];
        a4 += w9 * src[1];
        a0 += w10 * src[2];
        a1 += w11 * src[2];
        a2 += w12 * src[2];
        a3 += w13 * src[2];
        a4 += w14 * src[2];
        a0 += w15 * src[3];
        a1 += w16 * src[3];
        a2 += w17 * src[3];
        a3 += w18 * src[3];
        a4 += w19 * src[3];
        a0 += w20 * src[4];
        a1 += w21 * src[4];
        a2 += w22 * src[4];
        a3 += w23 * src[4];
        a4 += w24 * src[4];
        float* zf = z + static_cast<size_t>(f) * 5;
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
        zf[3] = a3;
        zf[4] = a4;
      }
    }

    // Taps 1..K-2: accumulate into z with the same unrolled inner kernel.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 25];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4];
      const float w5 = wk[5], w6 = wk[6], w7 = wk[7], w8 = wk[8], w9 = wk[9];
      const float w10 = wk[10], w11 = wk[11], w12 = wk[12], w13 = wk[13], w14 = wk[14];
      const float w15 = wk[15], w16 = wk[16], w17 = wk[17], w18 = wk[18], w19 = wk[19];
      const float w20 = wk[20], w21 = wk[21], w22 = wk[22], w23 = wk[23], w24 = wk[24];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 5];
        float* zf = z + static_cast<size_t>(f) * 5;
        float a0 = zf[0] + w0 * src[0];
        float a1 = zf[1] + w1 * src[0];
        float a2 = zf[2] + w2 * src[0];
        float a3 = zf[3] + w3 * src[0];
        float a4 = zf[4] + w4 * src[0];
        a0 += w5 * src[1];
        a1 += w6 * src[1];
        a2 += w7 * src[1];
        a3 += w8 * src[1];
        a4 += w9 * src[1];
        a0 += w10 * src[2];
        a1 += w11 * src[2];
        a2 += w12 * src[2];
        a3 += w13 * src[2];
        a4 += w14 * src[2];
        a0 += w15 * src[3];
        a1 += w16 * src[3];
        a2 += w17 * src[3];
        a3 += w18 * src[3];
        a4 += w19 * src[3];
        a0 += w20 * src[4];
        a1 += w21 * src[4];
        a2 += w22 * src[4];
        a3 += w23 * src[4];
        a4 += w24 * src[4];
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
        zf[3] = a3;
        zf[4] = a4;
      }
    }

    // Final tap (K-1, offset 0) fully inlined with the post-conv tail.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 25];
    const int tap_base_last = tap_base_phys(0);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2], cw3 = wk_last[3], cw4 = wk_last[4];
    const float cw5 = wk_last[5], cw6 = wk_last[6], cw7 = wk_last[7], cw8 = wk_last[8], cw9 = wk_last[9];
    const float cw10 = wk_last[10], cw11 = wk_last[11], cw12 = wk_last[12], cw13 = wk_last[13], cw14 = wk_last[14];
    const float cw15 = wk_last[15], cw16 = wk_last[16], cw17 = wk_last[17], cw18 = wk_last[18], cw19 = wk_last[19];
    const float cw20 = wk_last[20], cw21 = wk_last[21], cw22 = wk_last[22], cw23 = wk_last[23], cw24 = wk_last[24];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2], mw3 = L.mixin_w[3], mw4 = L.mixin_w[4];
    // layer1x1 col-major: lw[b*5 + c] is weight from bottleneck b to output c.
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2], lw03 = L.l1x1_w[3], lw04 = L.l1x1_w[4];
    const float lw10 = L.l1x1_w[5], lw11 = L.l1x1_w[6], lw12 = L.l1x1_w[7], lw13 = L.l1x1_w[8], lw14 = L.l1x1_w[9];
    const float lw20 = L.l1x1_w[10], lw21 = L.l1x1_w[11], lw22 = L.l1x1_w[12], lw23 = L.l1x1_w[13], lw24 = L.l1x1_w[14];
    const float lw30 = L.l1x1_w[15], lw31 = L.l1x1_w[16], lw32 = L.l1x1_w[17], lw33 = L.l1x1_w[18], lw34 = L.l1x1_w[19];
    const float lw40 = L.l1x1_w[20], lw41 = L.l1x1_w[21], lw42 = L.l1x1_w[22], lw43 = L.l1x1_w[23], lw44 = L.l1x1_w[24];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2], lb3 = L.l1x1_b[3], lb4 = L.l1x1_b[4];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 5];
      const float* zf_mem = z + static_cast<size_t>(f) * 5;
      // Final tap GEMV.
      float a0 = zf_mem[0] + cw0 * src[0];
      float a1 = zf_mem[1] + cw1 * src[0];
      float a2 = zf_mem[2] + cw2 * src[0];
      float a3 = zf_mem[3] + cw3 * src[0];
      float a4 = zf_mem[4] + cw4 * src[0];
      a0 += cw5 * src[1];
      a1 += cw6 * src[1];
      a2 += cw7 * src[1];
      a3 += cw8 * src[1];
      a4 += cw9 * src[1];
      a0 += cw10 * src[2];
      a1 += cw11 * src[2];
      a2 += cw12 * src[2];
      a3 += cw13 * src[2];
      a4 += cw14 * src[2];
      a0 += cw15 * src[3];
      a1 += cw16 * src[3];
      a2 += cw17 * src[3];
      a3 += cw18 * src[3];
      a4 += cw19 * src[3];
      a0 += cw20 * src[4];
      a1 += cw21 * src[4];
      a2 += cw22 * src[4];
      a3 += cw23 * src[4];
      a4 += cw24 * src[4];
      // Mixin + LeakyReLU.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a3 += mw3 * cf;
      a4 += mw4 * cf;
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      a3 = (a3 >= 0.0f) ? a3 : a3 * kLeakySlope;
      a4 = (a4 >= 0.0f) ? a4 : a4 * kLeakySlope;
      // Head sum accumulate.
      float* hsum = &_head_sum[static_cast<size_t>(f) * 5];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      hsum[3] += a3;
      hsum[4] += a4;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 5];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2 + lw30 * a3 + lw40 * a4;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2 + lw31 * a3 + lw41 * a4;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2 + lw32 * a3 + lw42 * a4;
      lin[3] += lb3 + lw03 * a0 + lw13 * a1 + lw23 * a2 + lw33 * a3 + lw43 * a4;
      lin[4] += lb4 + lw04 * a0 + lw14 * a1 + lw24 * a2 + lw34 * a3 + lw44 * a4;
    }
  }
  // [Fork] END: hand-unrolled Channels==5 kernel
  else if constexpr (Channels == 6)
  {
    float* z = _z.data();

    // Tap 0: seed z with conv_b (saves the memset-to-zero pass) and fold in
    // the first tap's FMAs.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11];
      const float w12 = wk[12], w13 = wk[13], w14 = wk[14], w15 = wk[15], w16 = wk[16], w17 = wk[17];
      const float w18 = wk[18], w19 = wk[19], w20 = wk[20], w21 = wk[21], w22 = wk[22], w23 = wk[23];
      const float w24 = wk[24], w25 = wk[25], w26 = wk[26], w27 = wk[27], w28 = wk[28], w29 = wk[29];
      const float w30 = wk[30], w31 = wk[31], w32 = wk[32], w33 = wk[33], w34 = wk[34], w35 = wk[35];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2], cb3 = L.conv_b[3], cb4 = L.conv_b[4], cb5 = L.conv_b[5];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 6];
        float a0 = cb0 + w0 * src[0];
        float a1 = cb1 + w1 * src[0];
        float a2 = cb2 + w2 * src[0];
        float a3 = cb3 + w3 * src[0];
        float a4 = cb4 + w4 * src[0];
        float a5 = cb5 + w5 * src[0];
        a0 += w6 * src[1];
        a1 += w7 * src[1];
        a2 += w8 * src[1];
        a3 += w9 * src[1];
        a4 += w10 * src[1];
        a5 += w11 * src[1];
        a0 += w12 * src[2];
        a1 += w13 * src[2];
        a2 += w14 * src[2];
        a3 += w15 * src[2];
        a4 += w16 * src[2];
        a5 += w17 * src[2];
        a0 += w18 * src[3];
        a1 += w19 * src[3];
        a2 += w20 * src[3];
        a3 += w21 * src[3];
        a4 += w22 * src[3];
        a5 += w23 * src[3];
        a0 += w24 * src[4];
        a1 += w25 * src[4];
        a2 += w26 * src[4];
        a3 += w27 * src[4];
        a4 += w28 * src[4];
        a5 += w29 * src[4];
        a0 += w30 * src[5];
        a1 += w31 * src[5];
        a2 += w32 * src[5];
        a3 += w33 * src[5];
        a4 += w34 * src[5];
        a5 += w35 * src[5];
        float* zf = z + static_cast<size_t>(f) * 6;
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
        zf[3] = a3;
        zf[4] = a4;
        zf[5] = a5;
      }
    }

    // Taps 1..K-2: accumulate into z with the same unrolled inner kernel.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 36];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11];
      const float w12 = wk[12], w13 = wk[13], w14 = wk[14], w15 = wk[15], w16 = wk[16], w17 = wk[17];
      const float w18 = wk[18], w19 = wk[19], w20 = wk[20], w21 = wk[21], w22 = wk[22], w23 = wk[23];
      const float w24 = wk[24], w25 = wk[25], w26 = wk[26], w27 = wk[27], w28 = wk[28], w29 = wk[29];
      const float w30 = wk[30], w31 = wk[31], w32 = wk[32], w33 = wk[33], w34 = wk[34], w35 = wk[35];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 6];
        float* zf = z + static_cast<size_t>(f) * 6;
        float a0 = zf[0] + w0 * src[0];
        float a1 = zf[1] + w1 * src[0];
        float a2 = zf[2] + w2 * src[0];
        float a3 = zf[3] + w3 * src[0];
        float a4 = zf[4] + w4 * src[0];
        float a5 = zf[5] + w5 * src[0];
        a0 += w6 * src[1];
        a1 += w7 * src[1];
        a2 += w8 * src[1];
        a3 += w9 * src[1];
        a4 += w10 * src[1];
        a5 += w11 * src[1];
        a0 += w12 * src[2];
        a1 += w13 * src[2];
        a2 += w14 * src[2];
        a3 += w15 * src[2];
        a4 += w16 * src[2];
        a5 += w17 * src[2];
        a0 += w18 * src[3];
        a1 += w19 * src[3];
        a2 += w20 * src[3];
        a3 += w21 * src[3];
        a4 += w22 * src[3];
        a5 += w23 * src[3];
        a0 += w24 * src[4];
        a1 += w25 * src[4];
        a2 += w26 * src[4];
        a3 += w27 * src[4];
        a4 += w28 * src[4];
        a5 += w29 * src[4];
        a0 += w30 * src[5];
        a1 += w31 * src[5];
        a2 += w32 * src[5];
        a3 += w33 * src[5];
        a4 += w34 * src[5];
        a5 += w35 * src[5];
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
        zf[3] = a3;
        zf[4] = a4;
        zf[5] = a5;
      }
    }

    // Final tap (K-1, offset 0) fully inlined with the post-conv tail.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 36];
    const int tap_base_last = tap_base_phys(0);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2], cw3 = wk_last[3], cw4 = wk_last[4], cw5 = wk_last[5];
    const float cw6 = wk_last[6], cw7 = wk_last[7], cw8 = wk_last[8], cw9 = wk_last[9], cw10 = wk_last[10], cw11 = wk_last[11];
    const float cw12 = wk_last[12], cw13 = wk_last[13], cw14 = wk_last[14], cw15 = wk_last[15], cw16 = wk_last[16], cw17 = wk_last[17];
    const float cw18 = wk_last[18], cw19 = wk_last[19], cw20 = wk_last[20], cw21 = wk_last[21], cw22 = wk_last[22], cw23 = wk_last[23];
    const float cw24 = wk_last[24], cw25 = wk_last[25], cw26 = wk_last[26], cw27 = wk_last[27], cw28 = wk_last[28], cw29 = wk_last[29];
    const float cw30 = wk_last[30], cw31 = wk_last[31], cw32 = wk_last[32], cw33 = wk_last[33], cw34 = wk_last[34], cw35 = wk_last[35];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2], mw3 = L.mixin_w[3], mw4 = L.mixin_w[4], mw5 = L.mixin_w[5];
    // layer1x1 col-major: lw[b*6 + c] is weight from bottleneck b to output c.
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2], lw03 = L.l1x1_w[3], lw04 = L.l1x1_w[4], lw05 = L.l1x1_w[5];
    const float lw10 = L.l1x1_w[6], lw11 = L.l1x1_w[7], lw12 = L.l1x1_w[8], lw13 = L.l1x1_w[9], lw14 = L.l1x1_w[10], lw15 = L.l1x1_w[11];
    const float lw20 = L.l1x1_w[12], lw21 = L.l1x1_w[13], lw22 = L.l1x1_w[14], lw23 = L.l1x1_w[15], lw24 = L.l1x1_w[16], lw25 = L.l1x1_w[17];
    const float lw30 = L.l1x1_w[18], lw31 = L.l1x1_w[19], lw32 = L.l1x1_w[20], lw33 = L.l1x1_w[21], lw34 = L.l1x1_w[22], lw35 = L.l1x1_w[23];
    const float lw40 = L.l1x1_w[24], lw41 = L.l1x1_w[25], lw42 = L.l1x1_w[26], lw43 = L.l1x1_w[27], lw44 = L.l1x1_w[28], lw45 = L.l1x1_w[29];
    const float lw50 = L.l1x1_w[30], lw51 = L.l1x1_w[31], lw52 = L.l1x1_w[32], lw53 = L.l1x1_w[33], lw54 = L.l1x1_w[34], lw55 = L.l1x1_w[35];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2], lb3 = L.l1x1_b[3], lb4 = L.l1x1_b[4], lb5 = L.l1x1_b[5];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 6];
      const float* zf_mem = z + static_cast<size_t>(f) * 6;
      // Final tap GEMV.
      float a0 = zf_mem[0] + cw0 * src[0];
      float a1 = zf_mem[1] + cw1 * src[0];
      float a2 = zf_mem[2] + cw2 * src[0];
      float a3 = zf_mem[3] + cw3 * src[0];
      float a4 = zf_mem[4] + cw4 * src[0];
      float a5 = zf_mem[5] + cw5 * src[0];
      a0 += cw6 * src[1];
      a1 += cw7 * src[1];
      a2 += cw8 * src[1];
      a3 += cw9 * src[1];
      a4 += cw10 * src[1];
      a5 += cw11 * src[1];
      a0 += cw12 * src[2];
      a1 += cw13 * src[2];
      a2 += cw14 * src[2];
      a3 += cw15 * src[2];
      a4 += cw16 * src[2];
      a5 += cw17 * src[2];
      a0 += cw18 * src[3];
      a1 += cw19 * src[3];
      a2 += cw20 * src[3];
      a3 += cw21 * src[3];
      a4 += cw22 * src[3];
      a5 += cw23 * src[3];
      a0 += cw24 * src[4];
      a1 += cw25 * src[4];
      a2 += cw26 * src[4];
      a3 += cw27 * src[4];
      a4 += cw28 * src[4];
      a5 += cw29 * src[4];
      a0 += cw30 * src[5];
      a1 += cw31 * src[5];
      a2 += cw32 * src[5];
      a3 += cw33 * src[5];
      a4 += cw34 * src[5];
      a5 += cw35 * src[5];
      // Mixin + LeakyReLU.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a3 += mw3 * cf;
      a4 += mw4 * cf;
      a5 += mw5 * cf;
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      a3 = (a3 >= 0.0f) ? a3 : a3 * kLeakySlope;
      a4 = (a4 >= 0.0f) ? a4 : a4 * kLeakySlope;
      a5 = (a5 >= 0.0f) ? a5 : a5 * kLeakySlope;
      // Head sum accumulate.
      float* hsum = &_head_sum[static_cast<size_t>(f) * 6];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      hsum[3] += a3;
      hsum[4] += a4;
      hsum[5] += a5;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 6];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2 + lw30 * a3 + lw40 * a4 + lw50 * a5;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2 + lw31 * a3 + lw41 * a4 + lw51 * a5;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2 + lw32 * a3 + lw42 * a4 + lw52 * a5;
      lin[3] += lb3 + lw03 * a0 + lw13 * a1 + lw23 * a2 + lw33 * a3 + lw43 * a4 + lw53 * a5;
      lin[4] += lb4 + lw04 * a0 + lw14 * a1 + lw24 * a2 + lw34 * a3 + lw44 * a4 + lw54 * a5;
      lin[5] += lb5 + lw05 * a0 + lw15 * a1 + lw25 * a2 + lw35 * a3 + lw45 * a4 + lw55 * a5;
    }
  }
  // [Fork] END: hand-unrolled Channels==6 kernel
  else if constexpr (Channels == 7)
  {
    float* z = _z.data();

    // Tap 0: seed z with conv_b (saves the memset-to-zero pass) and fold in
    // the first tap's FMAs.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5], w6 = wk[6];
      const float w7 = wk[7], w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11], w12 = wk[12], w13 = wk[13];
      const float w14 = wk[14], w15 = wk[15], w16 = wk[16], w17 = wk[17], w18 = wk[18], w19 = wk[19], w20 = wk[20];
      const float w21 = wk[21], w22 = wk[22], w23 = wk[23], w24 = wk[24], w25 = wk[25], w26 = wk[26], w27 = wk[27];
      const float w28 = wk[28], w29 = wk[29], w30 = wk[30], w31 = wk[31], w32 = wk[32], w33 = wk[33], w34 = wk[34];
      const float w35 = wk[35], w36 = wk[36], w37 = wk[37], w38 = wk[38], w39 = wk[39], w40 = wk[40], w41 = wk[41];
      const float w42 = wk[42], w43 = wk[43], w44 = wk[44], w45 = wk[45], w46 = wk[46], w47 = wk[47], w48 = wk[48];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2], cb3 = L.conv_b[3], cb4 = L.conv_b[4], cb5 = L.conv_b[5], cb6 = L.conv_b[6];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 7];
        float a0 = cb0 + w0 * src[0];
        float a1 = cb1 + w1 * src[0];
        float a2 = cb2 + w2 * src[0];
        float a3 = cb3 + w3 * src[0];
        float a4 = cb4 + w4 * src[0];
        float a5 = cb5 + w5 * src[0];
        float a6 = cb6 + w6 * src[0];
        a0 += w7 * src[1];
        a1 += w8 * src[1];
        a2 += w9 * src[1];
        a3 += w10 * src[1];
        a4 += w11 * src[1];
        a5 += w12 * src[1];
        a6 += w13 * src[1];
        a0 += w14 * src[2];
        a1 += w15 * src[2];
        a2 += w16 * src[2];
        a3 += w17 * src[2];
        a4 += w18 * src[2];
        a5 += w19 * src[2];
        a6 += w20 * src[2];
        a0 += w21 * src[3];
        a1 += w22 * src[3];
        a2 += w23 * src[3];
        a3 += w24 * src[3];
        a4 += w25 * src[3];
        a5 += w26 * src[3];
        a6 += w27 * src[3];
        a0 += w28 * src[4];
        a1 += w29 * src[4];
        a2 += w30 * src[4];
        a3 += w31 * src[4];
        a4 += w32 * src[4];
        a5 += w33 * src[4];
        a6 += w34 * src[4];
        a0 += w35 * src[5];
        a1 += w36 * src[5];
        a2 += w37 * src[5];
        a3 += w38 * src[5];
        a4 += w39 * src[5];
        a5 += w40 * src[5];
        a6 += w41 * src[5];
        a0 += w42 * src[6];
        a1 += w43 * src[6];
        a2 += w44 * src[6];
        a3 += w45 * src[6];
        a4 += w46 * src[6];
        a5 += w47 * src[6];
        a6 += w48 * src[6];
        float* zf = z + static_cast<size_t>(f) * 7;
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
        zf[3] = a3;
        zf[4] = a4;
        zf[5] = a5;
        zf[6] = a6;
      }
    }

    // Taps 1..K-2: accumulate into z with the same unrolled inner kernel.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 49];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5], w6 = wk[6];
      const float w7 = wk[7], w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11], w12 = wk[12], w13 = wk[13];
      const float w14 = wk[14], w15 = wk[15], w16 = wk[16], w17 = wk[17], w18 = wk[18], w19 = wk[19], w20 = wk[20];
      const float w21 = wk[21], w22 = wk[22], w23 = wk[23], w24 = wk[24], w25 = wk[25], w26 = wk[26], w27 = wk[27];
      const float w28 = wk[28], w29 = wk[29], w30 = wk[30], w31 = wk[31], w32 = wk[32], w33 = wk[33], w34 = wk[34];
      const float w35 = wk[35], w36 = wk[36], w37 = wk[37], w38 = wk[38], w39 = wk[39], w40 = wk[40], w41 = wk[41];
      const float w42 = wk[42], w43 = wk[43], w44 = wk[44], w45 = wk[45], w46 = wk[46], w47 = wk[47], w48 = wk[48];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 7];
        float* zf = z + static_cast<size_t>(f) * 7;
        float a0 = zf[0] + w0 * src[0];
        float a1 = zf[1] + w1 * src[0];
        float a2 = zf[2] + w2 * src[0];
        float a3 = zf[3] + w3 * src[0];
        float a4 = zf[4] + w4 * src[0];
        float a5 = zf[5] + w5 * src[0];
        float a6 = zf[6] + w6 * src[0];
        a0 += w7 * src[1];
        a1 += w8 * src[1];
        a2 += w9 * src[1];
        a3 += w10 * src[1];
        a4 += w11 * src[1];
        a5 += w12 * src[1];
        a6 += w13 * src[1];
        a0 += w14 * src[2];
        a1 += w15 * src[2];
        a2 += w16 * src[2];
        a3 += w17 * src[2];
        a4 += w18 * src[2];
        a5 += w19 * src[2];
        a6 += w20 * src[2];
        a0 += w21 * src[3];
        a1 += w22 * src[3];
        a2 += w23 * src[3];
        a3 += w24 * src[3];
        a4 += w25 * src[3];
        a5 += w26 * src[3];
        a6 += w27 * src[3];
        a0 += w28 * src[4];
        a1 += w29 * src[4];
        a2 += w30 * src[4];
        a3 += w31 * src[4];
        a4 += w32 * src[4];
        a5 += w33 * src[4];
        a6 += w34 * src[4];
        a0 += w35 * src[5];
        a1 += w36 * src[5];
        a2 += w37 * src[5];
        a3 += w38 * src[5];
        a4 += w39 * src[5];
        a5 += w40 * src[5];
        a6 += w41 * src[5];
        a0 += w42 * src[6];
        a1 += w43 * src[6];
        a2 += w44 * src[6];
        a3 += w45 * src[6];
        a4 += w46 * src[6];
        a5 += w47 * src[6];
        a6 += w48 * src[6];
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
        zf[3] = a3;
        zf[4] = a4;
        zf[5] = a5;
        zf[6] = a6;
      }
    }

    // Final tap (K-1, offset 0) fully inlined with the post-conv tail.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 49];
    const int tap_base_last = tap_base_phys(0);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2], cw3 = wk_last[3], cw4 = wk_last[4], cw5 = wk_last[5], cw6 = wk_last[6];
    const float cw7 = wk_last[7], cw8 = wk_last[8], cw9 = wk_last[9], cw10 = wk_last[10], cw11 = wk_last[11], cw12 = wk_last[12], cw13 = wk_last[13];
    const float cw14 = wk_last[14], cw15 = wk_last[15], cw16 = wk_last[16], cw17 = wk_last[17], cw18 = wk_last[18], cw19 = wk_last[19], cw20 = wk_last[20];
    const float cw21 = wk_last[21], cw22 = wk_last[22], cw23 = wk_last[23], cw24 = wk_last[24], cw25 = wk_last[25], cw26 = wk_last[26], cw27 = wk_last[27];
    const float cw28 = wk_last[28], cw29 = wk_last[29], cw30 = wk_last[30], cw31 = wk_last[31], cw32 = wk_last[32], cw33 = wk_last[33], cw34 = wk_last[34];
    const float cw35 = wk_last[35], cw36 = wk_last[36], cw37 = wk_last[37], cw38 = wk_last[38], cw39 = wk_last[39], cw40 = wk_last[40], cw41 = wk_last[41];
    const float cw42 = wk_last[42], cw43 = wk_last[43], cw44 = wk_last[44], cw45 = wk_last[45], cw46 = wk_last[46], cw47 = wk_last[47], cw48 = wk_last[48];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2], mw3 = L.mixin_w[3], mw4 = L.mixin_w[4], mw5 = L.mixin_w[5], mw6 = L.mixin_w[6];
    // layer1x1 col-major: lw[b*7 + c] is weight from bottleneck b to output c.
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2], lw03 = L.l1x1_w[3], lw04 = L.l1x1_w[4], lw05 = L.l1x1_w[5], lw06 = L.l1x1_w[6];
    const float lw10 = L.l1x1_w[7], lw11 = L.l1x1_w[8], lw12 = L.l1x1_w[9], lw13 = L.l1x1_w[10], lw14 = L.l1x1_w[11], lw15 = L.l1x1_w[12], lw16 = L.l1x1_w[13];
    const float lw20 = L.l1x1_w[14], lw21 = L.l1x1_w[15], lw22 = L.l1x1_w[16], lw23 = L.l1x1_w[17], lw24 = L.l1x1_w[18], lw25 = L.l1x1_w[19], lw26 = L.l1x1_w[20];
    const float lw30 = L.l1x1_w[21], lw31 = L.l1x1_w[22], lw32 = L.l1x1_w[23], lw33 = L.l1x1_w[24], lw34 = L.l1x1_w[25], lw35 = L.l1x1_w[26], lw36 = L.l1x1_w[27];
    const float lw40 = L.l1x1_w[28], lw41 = L.l1x1_w[29], lw42 = L.l1x1_w[30], lw43 = L.l1x1_w[31], lw44 = L.l1x1_w[32], lw45 = L.l1x1_w[33], lw46 = L.l1x1_w[34];
    const float lw50 = L.l1x1_w[35], lw51 = L.l1x1_w[36], lw52 = L.l1x1_w[37], lw53 = L.l1x1_w[38], lw54 = L.l1x1_w[39], lw55 = L.l1x1_w[40], lw56 = L.l1x1_w[41];
    const float lw60 = L.l1x1_w[42], lw61 = L.l1x1_w[43], lw62 = L.l1x1_w[44], lw63 = L.l1x1_w[45], lw64 = L.l1x1_w[46], lw65 = L.l1x1_w[47], lw66 = L.l1x1_w[48];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2], lb3 = L.l1x1_b[3], lb4 = L.l1x1_b[4], lb5 = L.l1x1_b[5], lb6 = L.l1x1_b[6];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 7];
      const float* zf_mem = z + static_cast<size_t>(f) * 7;
      // Final tap GEMV.
      float a0 = zf_mem[0] + cw0 * src[0];
      float a1 = zf_mem[1] + cw1 * src[0];
      float a2 = zf_mem[2] + cw2 * src[0];
      float a3 = zf_mem[3] + cw3 * src[0];
      float a4 = zf_mem[4] + cw4 * src[0];
      float a5 = zf_mem[5] + cw5 * src[0];
      float a6 = zf_mem[6] + cw6 * src[0];
      a0 += cw7 * src[1];
      a1 += cw8 * src[1];
      a2 += cw9 * src[1];
      a3 += cw10 * src[1];
      a4 += cw11 * src[1];
      a5 += cw12 * src[1];
      a6 += cw13 * src[1];
      a0 += cw14 * src[2];
      a1 += cw15 * src[2];
      a2 += cw16 * src[2];
      a3 += cw17 * src[2];
      a4 += cw18 * src[2];
      a5 += cw19 * src[2];
      a6 += cw20 * src[2];
      a0 += cw21 * src[3];
      a1 += cw22 * src[3];
      a2 += cw23 * src[3];
      a3 += cw24 * src[3];
      a4 += cw25 * src[3];
      a5 += cw26 * src[3];
      a6 += cw27 * src[3];
      a0 += cw28 * src[4];
      a1 += cw29 * src[4];
      a2 += cw30 * src[4];
      a3 += cw31 * src[4];
      a4 += cw32 * src[4];
      a5 += cw33 * src[4];
      a6 += cw34 * src[4];
      a0 += cw35 * src[5];
      a1 += cw36 * src[5];
      a2 += cw37 * src[5];
      a3 += cw38 * src[5];
      a4 += cw39 * src[5];
      a5 += cw40 * src[5];
      a6 += cw41 * src[5];
      a0 += cw42 * src[6];
      a1 += cw43 * src[6];
      a2 += cw44 * src[6];
      a3 += cw45 * src[6];
      a4 += cw46 * src[6];
      a5 += cw47 * src[6];
      a6 += cw48 * src[6];
      // Mixin + LeakyReLU.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a3 += mw3 * cf;
      a4 += mw4 * cf;
      a5 += mw5 * cf;
      a6 += mw6 * cf;
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      a3 = (a3 >= 0.0f) ? a3 : a3 * kLeakySlope;
      a4 = (a4 >= 0.0f) ? a4 : a4 * kLeakySlope;
      a5 = (a5 >= 0.0f) ? a5 : a5 * kLeakySlope;
      a6 = (a6 >= 0.0f) ? a6 : a6 * kLeakySlope;
      // Head sum accumulate.
      float* hsum = &_head_sum[static_cast<size_t>(f) * 7];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      hsum[3] += a3;
      hsum[4] += a4;
      hsum[5] += a5;
      hsum[6] += a6;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 7];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2 + lw30 * a3 + lw40 * a4 + lw50 * a5 + lw60 * a6;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2 + lw31 * a3 + lw41 * a4 + lw51 * a5 + lw61 * a6;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2 + lw32 * a3 + lw42 * a4 + lw52 * a5 + lw62 * a6;
      lin[3] += lb3 + lw03 * a0 + lw13 * a1 + lw23 * a2 + lw33 * a3 + lw43 * a4 + lw53 * a5 + lw63 * a6;
      lin[4] += lb4 + lw04 * a0 + lw14 * a1 + lw24 * a2 + lw34 * a3 + lw44 * a4 + lw54 * a5 + lw64 * a6;
      lin[5] += lb5 + lw05 * a0 + lw15 * a1 + lw25 * a2 + lw35 * a3 + lw45 * a4 + lw55 * a5 + lw65 * a6;
      lin[6] += lb6 + lw06 * a0 + lw16 * a1 + lw26 * a2 + lw36 * a3 + lw46 * a4 + lw56 * a5 + lw66 * a6;
    }
  }
  // [Fork] END: hand-unrolled Channels==7 kernel

  else if constexpr (Channels == 8)
  {
    float* z = _z.data();

    // Tap 0: seed z with conv_b (saves the memset-to-zero pass) and fold in
    // the first tap's FMAs.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5], w6 = wk[6], w7 = wk[7];
      const float w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11], w12 = wk[12], w13 = wk[13], w14 = wk[14], w15 = wk[15];
      const float w16 = wk[16], w17 = wk[17], w18 = wk[18], w19 = wk[19], w20 = wk[20], w21 = wk[21], w22 = wk[22], w23 = wk[23];
      const float w24 = wk[24], w25 = wk[25], w26 = wk[26], w27 = wk[27], w28 = wk[28], w29 = wk[29], w30 = wk[30], w31 = wk[31];
      const float w32 = wk[32], w33 = wk[33], w34 = wk[34], w35 = wk[35], w36 = wk[36], w37 = wk[37], w38 = wk[38], w39 = wk[39];
      const float w40 = wk[40], w41 = wk[41], w42 = wk[42], w43 = wk[43], w44 = wk[44], w45 = wk[45], w46 = wk[46], w47 = wk[47];
      const float w48 = wk[48], w49 = wk[49], w50 = wk[50], w51 = wk[51], w52 = wk[52], w53 = wk[53], w54 = wk[54], w55 = wk[55];
      const float w56 = wk[56], w57 = wk[57], w58 = wk[58], w59 = wk[59], w60 = wk[60], w61 = wk[61], w62 = wk[62], w63 = wk[63];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2], cb3 = L.conv_b[3], cb4 = L.conv_b[4], cb5 = L.conv_b[5], cb6 = L.conv_b[6], cb7 = L.conv_b[7];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 8];
        float a0 = cb0 + w0 * src[0];
        float a1 = cb1 + w1 * src[0];
        float a2 = cb2 + w2 * src[0];
        float a3 = cb3 + w3 * src[0];
        float a4 = cb4 + w4 * src[0];
        float a5 = cb5 + w5 * src[0];
        float a6 = cb6 + w6 * src[0];
        float a7 = cb7 + w7 * src[0];
        a0 += w8 * src[1];
        a1 += w9 * src[1];
        a2 += w10 * src[1];
        a3 += w11 * src[1];
        a4 += w12 * src[1];
        a5 += w13 * src[1];
        a6 += w14 * src[1];
        a7 += w15 * src[1];
        a0 += w16 * src[2];
        a1 += w17 * src[2];
        a2 += w18 * src[2];
        a3 += w19 * src[2];
        a4 += w20 * src[2];
        a5 += w21 * src[2];
        a6 += w22 * src[2];
        a7 += w23 * src[2];
        a0 += w24 * src[3];
        a1 += w25 * src[3];
        a2 += w26 * src[3];
        a3 += w27 * src[3];
        a4 += w28 * src[3];
        a5 += w29 * src[3];
        a6 += w30 * src[3];
        a7 += w31 * src[3];
        a0 += w32 * src[4];
        a1 += w33 * src[4];
        a2 += w34 * src[4];
        a3 += w35 * src[4];
        a4 += w36 * src[4];
        a5 += w37 * src[4];
        a6 += w38 * src[4];
        a7 += w39 * src[4];
        a0 += w40 * src[5];
        a1 += w41 * src[5];
        a2 += w42 * src[5];
        a3 += w43 * src[5];
        a4 += w44 * src[5];
        a5 += w45 * src[5];
        a6 += w46 * src[5];
        a7 += w47 * src[5];
        a0 += w48 * src[6];
        a1 += w49 * src[6];
        a2 += w50 * src[6];
        a3 += w51 * src[6];
        a4 += w52 * src[6];
        a5 += w53 * src[6];
        a6 += w54 * src[6];
        a7 += w55 * src[6];
        a0 += w56 * src[7];
        a1 += w57 * src[7];
        a2 += w58 * src[7];
        a3 += w59 * src[7];
        a4 += w60 * src[7];
        a5 += w61 * src[7];
        a6 += w62 * src[7];
        a7 += w63 * src[7];
        float* zf = z + static_cast<size_t>(f) * 8;
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
        zf[3] = a3;
        zf[4] = a4;
        zf[5] = a5;
        zf[6] = a6;
        zf[7] = a7;
      }
    }

    // Taps 1..K-2: accumulate into z with the same unrolled inner kernel.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 64];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5], w6 = wk[6], w7 = wk[7];
      const float w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11], w12 = wk[12], w13 = wk[13], w14 = wk[14], w15 = wk[15];
      const float w16 = wk[16], w17 = wk[17], w18 = wk[18], w19 = wk[19], w20 = wk[20], w21 = wk[21], w22 = wk[22], w23 = wk[23];
      const float w24 = wk[24], w25 = wk[25], w26 = wk[26], w27 = wk[27], w28 = wk[28], w29 = wk[29], w30 = wk[30], w31 = wk[31];
      const float w32 = wk[32], w33 = wk[33], w34 = wk[34], w35 = wk[35], w36 = wk[36], w37 = wk[37], w38 = wk[38], w39 = wk[39];
      const float w40 = wk[40], w41 = wk[41], w42 = wk[42], w43 = wk[43], w44 = wk[44], w45 = wk[45], w46 = wk[46], w47 = wk[47];
      const float w48 = wk[48], w49 = wk[49], w50 = wk[50], w51 = wk[51], w52 = wk[52], w53 = wk[53], w54 = wk[54], w55 = wk[55];
      const float w56 = wk[56], w57 = wk[57], w58 = wk[58], w59 = wk[59], w60 = wk[60], w61 = wk[61], w62 = wk[62], w63 = wk[63];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 8];
        float* zf = z + static_cast<size_t>(f) * 8;
        float a0 = zf[0] + w0 * src[0];
        float a1 = zf[1] + w1 * src[0];
        float a2 = zf[2] + w2 * src[0];
        float a3 = zf[3] + w3 * src[0];
        float a4 = zf[4] + w4 * src[0];
        float a5 = zf[5] + w5 * src[0];
        float a6 = zf[6] + w6 * src[0];
        float a7 = zf[7] + w7 * src[0];
        a0 += w8 * src[1];
        a1 += w9 * src[1];
        a2 += w10 * src[1];
        a3 += w11 * src[1];
        a4 += w12 * src[1];
        a5 += w13 * src[1];
        a6 += w14 * src[1];
        a7 += w15 * src[1];
        a0 += w16 * src[2];
        a1 += w17 * src[2];
        a2 += w18 * src[2];
        a3 += w19 * src[2];
        a4 += w20 * src[2];
        a5 += w21 * src[2];
        a6 += w22 * src[2];
        a7 += w23 * src[2];
        a0 += w24 * src[3];
        a1 += w25 * src[3];
        a2 += w26 * src[3];
        a3 += w27 * src[3];
        a4 += w28 * src[3];
        a5 += w29 * src[3];
        a6 += w30 * src[3];
        a7 += w31 * src[3];
        a0 += w32 * src[4];
        a1 += w33 * src[4];
        a2 += w34 * src[4];
        a3 += w35 * src[4];
        a4 += w36 * src[4];
        a5 += w37 * src[4];
        a6 += w38 * src[4];
        a7 += w39 * src[4];
        a0 += w40 * src[5];
        a1 += w41 * src[5];
        a2 += w42 * src[5];
        a3 += w43 * src[5];
        a4 += w44 * src[5];
        a5 += w45 * src[5];
        a6 += w46 * src[5];
        a7 += w47 * src[5];
        a0 += w48 * src[6];
        a1 += w49 * src[6];
        a2 += w50 * src[6];
        a3 += w51 * src[6];
        a4 += w52 * src[6];
        a5 += w53 * src[6];
        a6 += w54 * src[6];
        a7 += w55 * src[6];
        a0 += w56 * src[7];
        a1 += w57 * src[7];
        a2 += w58 * src[7];
        a3 += w59 * src[7];
        a4 += w60 * src[7];
        a5 += w61 * src[7];
        a6 += w62 * src[7];
        a7 += w63 * src[7];
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
        zf[3] = a3;
        zf[4] = a4;
        zf[5] = a5;
        zf[6] = a6;
        zf[7] = a7;
      }
    }

    // Final tap (K-1, offset 0) fully inlined with the post-conv tail.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 64];
    const int tap_base_last = tap_base_phys(0);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2], cw3 = wk_last[3], cw4 = wk_last[4], cw5 = wk_last[5], cw6 = wk_last[6], cw7 = wk_last[7];
    const float cw8 = wk_last[8], cw9 = wk_last[9], cw10 = wk_last[10], cw11 = wk_last[11], cw12 = wk_last[12], cw13 = wk_last[13], cw14 = wk_last[14], cw15 = wk_last[15];
    const float cw16 = wk_last[16], cw17 = wk_last[17], cw18 = wk_last[18], cw19 = wk_last[19], cw20 = wk_last[20], cw21 = wk_last[21], cw22 = wk_last[22], cw23 = wk_last[23];
    const float cw24 = wk_last[24], cw25 = wk_last[25], cw26 = wk_last[26], cw27 = wk_last[27], cw28 = wk_last[28], cw29 = wk_last[29], cw30 = wk_last[30], cw31 = wk_last[31];
    const float cw32 = wk_last[32], cw33 = wk_last[33], cw34 = wk_last[34], cw35 = wk_last[35], cw36 = wk_last[36], cw37 = wk_last[37], cw38 = wk_last[38], cw39 = wk_last[39];
    const float cw40 = wk_last[40], cw41 = wk_last[41], cw42 = wk_last[42], cw43 = wk_last[43], cw44 = wk_last[44], cw45 = wk_last[45], cw46 = wk_last[46], cw47 = wk_last[47];
    const float cw48 = wk_last[48], cw49 = wk_last[49], cw50 = wk_last[50], cw51 = wk_last[51], cw52 = wk_last[52], cw53 = wk_last[53], cw54 = wk_last[54], cw55 = wk_last[55];
    const float cw56 = wk_last[56], cw57 = wk_last[57], cw58 = wk_last[58], cw59 = wk_last[59], cw60 = wk_last[60], cw61 = wk_last[61], cw62 = wk_last[62], cw63 = wk_last[63];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2], mw3 = L.mixin_w[3], mw4 = L.mixin_w[4], mw5 = L.mixin_w[5], mw6 = L.mixin_w[6], mw7 = L.mixin_w[7];
    // layer1x1 col-major: lw[b*8 + c] is weight from bottleneck b to output c.
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2], lw03 = L.l1x1_w[3], lw04 = L.l1x1_w[4], lw05 = L.l1x1_w[5], lw06 = L.l1x1_w[6], lw07 = L.l1x1_w[7];
    const float lw10 = L.l1x1_w[8], lw11 = L.l1x1_w[9], lw12 = L.l1x1_w[10], lw13 = L.l1x1_w[11], lw14 = L.l1x1_w[12], lw15 = L.l1x1_w[13], lw16 = L.l1x1_w[14], lw17 = L.l1x1_w[15];
    const float lw20 = L.l1x1_w[16], lw21 = L.l1x1_w[17], lw22 = L.l1x1_w[18], lw23 = L.l1x1_w[19], lw24 = L.l1x1_w[20], lw25 = L.l1x1_w[21], lw26 = L.l1x1_w[22], lw27 = L.l1x1_w[23];
    const float lw30 = L.l1x1_w[24], lw31 = L.l1x1_w[25], lw32 = L.l1x1_w[26], lw33 = L.l1x1_w[27], lw34 = L.l1x1_w[28], lw35 = L.l1x1_w[29], lw36 = L.l1x1_w[30], lw37 = L.l1x1_w[31];
    const float lw40 = L.l1x1_w[32], lw41 = L.l1x1_w[33], lw42 = L.l1x1_w[34], lw43 = L.l1x1_w[35], lw44 = L.l1x1_w[36], lw45 = L.l1x1_w[37], lw46 = L.l1x1_w[38], lw47 = L.l1x1_w[39];
    const float lw50 = L.l1x1_w[40], lw51 = L.l1x1_w[41], lw52 = L.l1x1_w[42], lw53 = L.l1x1_w[43], lw54 = L.l1x1_w[44], lw55 = L.l1x1_w[45], lw56 = L.l1x1_w[46], lw57 = L.l1x1_w[47];
    const float lw60 = L.l1x1_w[48], lw61 = L.l1x1_w[49], lw62 = L.l1x1_w[50], lw63 = L.l1x1_w[51], lw64 = L.l1x1_w[52], lw65 = L.l1x1_w[53], lw66 = L.l1x1_w[54], lw67 = L.l1x1_w[55];
    const float lw70 = L.l1x1_w[56], lw71 = L.l1x1_w[57], lw72 = L.l1x1_w[58], lw73 = L.l1x1_w[59], lw74 = L.l1x1_w[60], lw75 = L.l1x1_w[61], lw76 = L.l1x1_w[62], lw77 = L.l1x1_w[63];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2], lb3 = L.l1x1_b[3], lb4 = L.l1x1_b[4], lb5 = L.l1x1_b[5], lb6 = L.l1x1_b[6], lb7 = L.l1x1_b[7];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 8];
      const float* zf_mem = z + static_cast<size_t>(f) * 8;
      // Final tap GEMV.
      float a0 = zf_mem[0] + cw0 * src[0];
      float a1 = zf_mem[1] + cw1 * src[0];
      float a2 = zf_mem[2] + cw2 * src[0];
      float a3 = zf_mem[3] + cw3 * src[0];
      float a4 = zf_mem[4] + cw4 * src[0];
      float a5 = zf_mem[5] + cw5 * src[0];
      float a6 = zf_mem[6] + cw6 * src[0];
      float a7 = zf_mem[7] + cw7 * src[0];
      a0 += cw8 * src[1];
      a1 += cw9 * src[1];
      a2 += cw10 * src[1];
      a3 += cw11 * src[1];
      a4 += cw12 * src[1];
      a5 += cw13 * src[1];
      a6 += cw14 * src[1];
      a7 += cw15 * src[1];
      a0 += cw16 * src[2];
      a1 += cw17 * src[2];
      a2 += cw18 * src[2];
      a3 += cw19 * src[2];
      a4 += cw20 * src[2];
      a5 += cw21 * src[2];
      a6 += cw22 * src[2];
      a7 += cw23 * src[2];
      a0 += cw24 * src[3];
      a1 += cw25 * src[3];
      a2 += cw26 * src[3];
      a3 += cw27 * src[3];
      a4 += cw28 * src[3];
      a5 += cw29 * src[3];
      a6 += cw30 * src[3];
      a7 += cw31 * src[3];
      a0 += cw32 * src[4];
      a1 += cw33 * src[4];
      a2 += cw34 * src[4];
      a3 += cw35 * src[4];
      a4 += cw36 * src[4];
      a5 += cw37 * src[4];
      a6 += cw38 * src[4];
      a7 += cw39 * src[4];
      a0 += cw40 * src[5];
      a1 += cw41 * src[5];
      a2 += cw42 * src[5];
      a3 += cw43 * src[5];
      a4 += cw44 * src[5];
      a5 += cw45 * src[5];
      a6 += cw46 * src[5];
      a7 += cw47 * src[5];
      a0 += cw48 * src[6];
      a1 += cw49 * src[6];
      a2 += cw50 * src[6];
      a3 += cw51 * src[6];
      a4 += cw52 * src[6];
      a5 += cw53 * src[6];
      a6 += cw54 * src[6];
      a7 += cw55 * src[6];
      a0 += cw56 * src[7];
      a1 += cw57 * src[7];
      a2 += cw58 * src[7];
      a3 += cw59 * src[7];
      a4 += cw60 * src[7];
      a5 += cw61 * src[7];
      a6 += cw62 * src[7];
      a7 += cw63 * src[7];
      // Mixin + LeakyReLU.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a3 += mw3 * cf;
      a4 += mw4 * cf;
      a5 += mw5 * cf;
      a6 += mw6 * cf;
      a7 += mw7 * cf;
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      a3 = (a3 >= 0.0f) ? a3 : a3 * kLeakySlope;
      a4 = (a4 >= 0.0f) ? a4 : a4 * kLeakySlope;
      a5 = (a5 >= 0.0f) ? a5 : a5 * kLeakySlope;
      a6 = (a6 >= 0.0f) ? a6 : a6 * kLeakySlope;
      a7 = (a7 >= 0.0f) ? a7 : a7 * kLeakySlope;
      // Head sum accumulate.
      float* hsum = &_head_sum[static_cast<size_t>(f) * 8];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      hsum[3] += a3;
      hsum[4] += a4;
      hsum[5] += a5;
      hsum[6] += a6;
      hsum[7] += a7;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 8];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2 + lw30 * a3 + lw40 * a4 + lw50 * a5 + lw60 * a6 + lw70 * a7;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2 + lw31 * a3 + lw41 * a4 + lw51 * a5 + lw61 * a6 + lw71 * a7;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2 + lw32 * a3 + lw42 * a4 + lw52 * a5 + lw62 * a6 + lw72 * a7;
      lin[3] += lb3 + lw03 * a0 + lw13 * a1 + lw23 * a2 + lw33 * a3 + lw43 * a4 + lw53 * a5 + lw63 * a6 + lw73 * a7;
      lin[4] += lb4 + lw04 * a0 + lw14 * a1 + lw24 * a2 + lw34 * a3 + lw44 * a4 + lw54 * a5 + lw64 * a6 + lw74 * a7;
      lin[5] += lb5 + lw05 * a0 + lw15 * a1 + lw25 * a2 + lw35 * a3 + lw45 * a4 + lw55 * a5 + lw65 * a6 + lw75 * a7;
      lin[6] += lb6 + lw06 * a0 + lw16 * a1 + lw26 * a2 + lw36 * a3 + lw46 * a4 + lw56 * a5 + lw66 * a6 + lw76 * a7;
      lin[7] += lb7 + lw07 * a0 + lw17 * a1 + lw27 * a2 + lw37 * a3 + lw47 * a4 + lw57 * a5 + lw67 * a6 + lw77 * a7;
    }
  }
  // [Fork] END: hand-unrolled Channels==8 kernel

  else
  {
    // Use Eigen's tuned 8x8 × 8xN GEMM for the whole block at once. Unlike a
    // small-tile version, this hits Eigen's actual GEMM kernel (tuned for
    // inner dimensions of ~64) rather than its tiny-matrix fallback path.
    //
    // Compile-time improvements over the generic WaveNet path:
    //   - Channels and Bottleneck are template constants (no dynamic shape).
    //   - Per-layer buffers are pre-sized at SetMaxBufferSize; nothing resizes
    //     during process().
    //   - No FiLM / gating / head1x1 / grouped-conv branches.
    //   - No virtual dispatch / conditional on optional layer features.
    //   - All conv + post-conv ops operate on the full block — even the
    //     mixin, bias, activation, and 1x1 residual are Eigen block ops so
    //     they vectorize the same way the GEMMs do.
    using MatCC = Eigen::Matrix<float, Channels, Channels>;
    using MatCDyn = Eigen::Matrix<float, Channels, Eigen::Dynamic>;
    using VecC = Eigen::Matrix<float, Channels, 1>;
    using RowDyn = Eigen::Matrix<float, 1, Eigen::Dynamic>;

    Eigen::Map<const VecC> conv_b_vec(L.conv_b.data());
    Eigen::Map<const VecC> mixin_vec(L.mixin_w.data());
    Eigen::Map<const MatCC> l1x1_mat(L.l1x1_w.data());
    Eigen::Map<const VecC> l1x1_b_vec(L.l1x1_b.data());
    Eigen::Map<const RowDyn> cond_row(cond, 1, num_frames);

    // Aligned16: _z/_head_sum/_layer_in use Eigen::aligned_allocator (guaranteed 16-byte
    // aligned), so it's safe to tell Eigen explicitly rather than let it assume Unaligned
    // and take its conservative (much slower) SIMD path. See member declarations above.
    Eigen::Map<MatCDyn, Eigen::Aligned16> ztile(_z.data(), Channels, num_frames);
    Eigen::Map<MatCDyn, Eigen::Aligned16> hsum_block(_head_sum.data(), Channels, num_frames);
    Eigen::Map<MatCDyn, Eigen::Aligned16> lin_block(_layer_in.data(), Channels, num_frames);

    ztile.setZero();

    // Conv: one 8x8 × 8xN GEMM per tap.
    for (int k = 0; k < K; k++)
    {
      const int tap_base = tap_base_phys(K - 1 - k);
      Eigen::Map<const MatCC> W(&L.conv_w[static_cast<size_t>(k) * Channels * Channels]);
      Eigen::Map<const MatCDyn> input_block(&L.history[static_cast<size_t>(tap_base) * Channels], Channels, num_frames);
      ztile.noalias() += W * input_block;
    }

    // Post-conv: bias, mixin, LeakyReLU, head_sum, 1x1 residual — all block ops.
    ztile.colwise() += conv_b_vec;
    ztile.noalias() += mixin_vec * cond_row; // rank-1 outer product
    ztile = (ztile.array() < 0.0f).select(ztile.array() * kLeakySlope, ztile.array());
    hsum_block += ztile;
    lin_block.noalias() += l1x1_mat * ztile; // 8x8 × 8xN GEMM
    lin_block.colwise() += l1x1_b_vec;
  }
}

// Runtime dispatcher: selects the K-specialized kernel for this layer.
// For the A2 shape the detector only admits K in {6, 15}; any other value
// here means something passed the detector that shouldn't have.
template <int Channels>
void A2FastModel<Channels>::_layer_forward(int layer_idx, const float* cond, int num_frames)
{
  Layer& L = _layers[layer_idx];
  _ring_write(L, num_frames);
  switch (L.kernel_size)
  {
    case 6: _layer_forward_k<6>(L, cond, num_frames); break;
    case 15: _layer_forward_k<15>(L, cond, num_frames); break;
    default: throw std::runtime_error("A2FastModel: unexpected kernel_size " + std::to_string(L.kernel_size));
  }
}

// -----------------------------------------------------------------------------
// Head: K=16 dilation-1 conv from Channels to 1, plus bias + scale.
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::_head_forward(float* output, int num_frames)
{
  _head_ring_write(num_frames);
  #if NAM_A2_RING_MODE == 1
  const int mask = _head_pow2_mask;
  auto col_of = [&](int f, int k) { return (_head_write_pos - num_frames + f - (kHeadKernelSize - 1 - k)) & mask; };
  #else
  const int base = _head_write_pos - num_frames;
  auto col_of = [&](int f, int k) { return base + f - (kHeadKernelSize - 1 - k); };
  #endif

  for (int f = 0; f < num_frames; f++)
  {
    float y = _head_b;
    for (int k = 0; k < kHeadKernelSize; k++)
    {
      const int col = col_of(f, k);
      const float* src = &_head_history[static_cast<size_t>(col) * Channels];
      const float* wk = _head_w[k].data();
      for (int b = 0; b < Channels; b++)
        y += wk[b] * src[b];
    }
    output[f] = y * _head_scale;
  }
}

// -----------------------------------------------------------------------------
// DSP::process override
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames)
{
  if (num_frames > GetMaxBufferSize())
    SetMaxBufferSize(num_frames);

  const NAM_SAMPLE* in0 = input[0];
  NAM_SAMPLE* out0 = output[0];

  // Rechannel: layer_in[c, f] = _rechannel_w[c] * input[f] for c in Channels.
  // Also prepare float cond buffer (input copied to float for inner loops).
  float* cond = _cond.data();
  for (int f = 0; f < num_frames; f++)
  {
    const float x = static_cast<float>(in0[f]);
    cond[f] = x;
    float* lin = &_layer_in[static_cast<size_t>(f) * Channels];
    for (int c = 0; c < Channels; c++)
      lin[c] = _rechannel_w[c] * x;
  }

  // Zero head accumulator.
  std::memset(_head_sum.data(), 0, static_cast<size_t>(num_frames) * Channels * sizeof(float));

  for (int li = 0; li < kNumLayers; li++)
    _layer_forward(li, cond, num_frames);

  // Output.
  float* head_out = _head_out.data();
  _head_forward(head_out, num_frames);
  for (int f = 0; f < num_frames; f++)
    out0[f] = static_cast<NAM_SAMPLE>(head_out[f]);
}

// -----------------------------------------------------------------------------
// A2FastConfig — wraps the constructed DSP behind the ModelConfig interface.
// -----------------------------------------------------------------------------
struct A2FastConfig : public ModelConfig
{
  int channels = 0;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override
  {
    if (channels == 3)
      return std::make_unique<A2FastModel<3>>(std::move(weights), sampleRate);
    // [Fork] BEGIN: intermediate widths
    if (channels == 4)
      return std::make_unique<A2FastModel<4>>(std::move(weights), sampleRate);
    if (channels == 5)
      return std::make_unique<A2FastModel<5>>(std::move(weights), sampleRate);
    if (channels == 6)
      return std::make_unique<A2FastModel<6>>(std::move(weights), sampleRate);
    if (channels == 7)
      return std::make_unique<A2FastModel<7>>(std::move(weights), sampleRate);
    // [Fork] END
    if (channels == 8)
      return std::make_unique<A2FastModel<8>>(std::move(weights), sampleRate);
    throw std::runtime_error("A2FastConfig: unsupported channel count " + std::to_string(channels));
  }
};

// -----------------------------------------------------------------------------
// Detector helpers
// -----------------------------------------------------------------------------
bool close_to(float v, float target)
{
  return std::fabs(v - target) <= 1e-7f;
}

bool all_none_strings(const nlohmann::json& j)
{
  if (!j.is_array())
    return false;
  for (const auto& e : j)
  {
    if (!e.is_string() || e.get<std::string>() != "none")
      return false;
  }
  return true;
}

bool all_null(const nlohmann::json& j)
{
  if (!j.is_array())
    return false;
  for (const auto& e : j)
  {
    if (!e.is_null())
      return false;
  }
  return true;
}

bool film_inactive(const nlohmann::json& layer, const char* key)
{
  auto it = layer.find(key);
  if (it == layer.end() || it->is_null())
    return true;
  if (it->is_boolean())
    return !it->get<bool>();
  if (it->is_object())
    return !it->value("active", false);
  return false;
}

// =============================================================================
// [Fork] BEGIN: ParametricA2FastModel<Channels>
//
// Fast path for ParametricWaveNet, replacing the generic Eigen WaveNet path
// (~17× faster on Apple Silicon). Added by Fork — not in upstream NAM.
//
// Optimized fast path for ParametricWaveNet (A2 topology + per-layer FiLM).
//
// One structural difference from A2FastModel:
//   1. FiLM (gamma * x + beta) applied after conv+mixin, before LeakyReLU.
//      Gamma and beta are precomputed from knob atomics once per buffer.
//
// The head is fed the same way as A2FastModel: _head_sum accumulates every
// layer's post-activation output (standard WaveNet skip connections), matching
// how NAM's parametric training actually builds these models. (An earlier,
// now-removed residual-only head variant fed the head from just the final
// layer's output instead -- see the SKIP-ONLY refactor; this doc comment used
// to describe that removed behavior.)
//
// Everything else (ring buffers, weight loading skeleton, head conv) is
// identical to A2FastModel so the same throughput characteristics apply.
// =============================================================================
template <int Channels>
class ParametricA2FastModel : public DSP
{
public:
  static constexpr int kChannels = Channels;
  static constexpr int kMaxParams = ParametricWaveNet::kMaxParams;

  ParametricA2FastModel(std::vector<float> weights, double expected_sample_rate,
                        int num_params, std::vector<ParametricParam> param_defs);
  ~ParametricA2FastModel() override = default;

  // Thread-safe: may be called from UI thread while process() runs on audio thread.
  void SetKnobValues(const std::vector<float>& values) override;

  std::vector<DSPParamDef> GetParameterDefs() const override
  {
    std::vector<DSPParamDef> result;
    result.reserve(_param_defs.size());
    for (const auto& p : _param_defs)
      result.push_back({p.name, p.min_val, p.max_val, p.default_val, p.steps});
    return result;
  }
  int GetNumParams() const override { return _num_params; }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override;
  int GetPrewarmSamples() override { return _prewarm_samples; }
  void Reset(double sampleRate, int maxBufferSize) override;

protected:
  void SetMaxBufferSize(int maxBufferSize) override;

private:
  struct Layer
  {
    int kernel_size = 0;
    int dilation = 0;
    int max_lookback = 0;
    std::vector<float> conv_w; // K × Channels × Channels, col-major per tap
    std::array<float, Channels> conv_b{};
    std::array<float, Channels> mixin_w{};
    std::array<float, Channels * Channels> l1x1_w{};
    std::array<float, Channels> l1x1_b{};
    std::vector<float> history;
  #if NAM_A2_RING_MODE == 1
    int pow2_size = 0;
    int pow2_mask = 0;
    int write_pos = 0;
  #else
    int history_cols = 0;
    int write_pos = 0;
  #endif
  };

  std::array<Layer, kNumLayers> _layers;

  // Per-layer FiLM network weights.
  // _film_net_w[li]: (2*Channels rows × _num_params cols), row-major.
  // _film_net_b[li]: 2*Channels floats.
  struct FilmLayer
  {
    std::vector<float> net_w; // 2*Channels × _num_params, row-major
    std::vector<float> net_b; // 2*Channels
  };
  std::array<FilmLayer, kNumLayers> _film_layers;

  // Per-layer precomputed gamma/beta (rebuilt each buffer from _knob_atoms).
  std::array<std::array<float, Channels>, kNumLayers> _film_gamma{};
  std::array<std::array<float, Channels>, kNumLayers> _film_beta{};

  int _num_params = 0;
  std::vector<ParametricParam> _param_defs;
  std::array<std::atomic<float>, kMaxParams> _knob_atoms{};
  // [Fork] Per-buffer one-pole smoother — eliminates zipper noise when UI sliders move.
  // Audio thread only; no atomic needed. Initialized to default knob value (0.5).
  std::array<float, kMaxParams> _knob_smoothed{};
  float _knob_smooth_coeff{0.3f}; // updated in Reset(); default covers startup before first Reset

  // Rechannel (1 → Channels), no bias.
  std::array<float, Channels> _rechannel_w{};

  // Head conv (Channels → 1, kernel=16, bias). Col-major per tap.
  std::array<std::array<float, Channels>, kHeadKernelSize> _head_w{};
  float _head_b = 0.0f;
  float _head_scale = 1.0f;

  // Head ring buffer. Reads from _head_sum (the skip-connection accumulator),
  // not from _layer_in.
  std::vector<float> _head_history;
  #if NAM_A2_RING_MODE == 1
  int _head_pow2_size = 0;
  int _head_pow2_mask = 0;
  int _head_write_pos = 0;
  #else
  int _head_history_cols = 0;
  int _head_write_pos = 0;
  #endif

  // Working buffers (Channels × maxBufferSize, col-major).
  // Eigen::aligned_allocator guarantees 16-byte alignment for the Eigen::Map<..., Eigen::Aligned16>
  // views below -- see A2FastModel's identical comment above for why this matters.
  std::vector<float, Eigen::aligned_allocator<float>> _layer_in; // layer input / residual accumulator
  // Skip accumulator: sum of every layer's post-activation (fed to the head).
  std::vector<float, Eigen::aligned_allocator<float>> _head_sum;
  std::vector<float, Eigen::aligned_allocator<float>> _z;        // per-layer conv accumulator
  std::vector<float> _cond;     // float32 copy of input audio
  std::vector<float> _head_out; // float32 head output

  // First-order IIR high-pass DC blocker (fc ≈ 20 Hz).
  float _dc_alpha = 0.99738f; // 1 - 2π·20/48000; recomputed in Reset()
  float _dc_x_prev = 0.0f;
  float _dc_y_prev = 0.0f;

  int _prewarm_samples = 0;

  void _load_weights(std::vector<float>& weights);
  // Rebuild _film_gamma/_film_beta for all layers from _knob_atoms. Audio thread only.
  void _recompute_film();
  void _ring_write(Layer& L, int num_frames);
  // Write _head_sum (the skip-connection accumulator) into _head_history ring.
  void _head_ring_write(int num_frames);

  template <int KernelSize>
  void _layer_forward_k(Layer& L, int li, const float* cond, int num_frames);
  void _layer_forward(int li, const float* cond, int num_frames);
  void _head_forward(float* output, int num_frames);
};

// --- Construction ---
template <int Channels>
ParametricA2FastModel<Channels>::ParametricA2FastModel(std::vector<float> weights,
                                                        double expected_sample_rate,
                                                        int num_params,
                                                        std::vector<ParametricParam> param_defs)
: DSP(/*in_channels=*/1, /*out_channels=*/1, expected_sample_rate)
, _num_params(num_params)
, _param_defs(std::move(param_defs))
{
  for (int i = 0; i < kNumLayers; i++)
  {
    _layers[i].kernel_size = kKernelSizes[i];
    _layers[i].dilation = kDilations[i];
    _layers[i].max_lookback = (kKernelSizes[i] - 1) * kDilations[i];
    _layers[i].conv_w.assign(static_cast<size_t>(kKernelSizes[i]) * Channels * Channels, 0.0f);
    _film_layers[i].net_w.assign(static_cast<size_t>(2 * Channels) * num_params, 0.0f);
    _film_layers[i].net_b.assign(static_cast<size_t>(2 * Channels), 0.0f);
  }

  // Default knob values match ParametricWaveNet constructor.
  for (int k = 0; k < num_params && k < kMaxParams; k++) {
    _knob_atoms[k].store(0.5f, std::memory_order_relaxed);
    _knob_smoothed[k] = 0.5f; // smoother starts at the same default so there's no initial transient
  }

  _load_weights(weights);

  int prewarm = 0;
  for (int i = 0; i < kNumLayers; i++)
    prewarm += _layers[i].max_lookback;
  prewarm += kHeadKernelSize - 1;
  _prewarm_samples = prewarm;
}

// --- Weight loader ---
// Order (must match Layer::set_weights_ with activation_pre_film active):
//   rechannel.weight (C floats)
//   per layer × 23: conv.w + conv.b + mixin.w + l1x1.w + l1x1.b + film.net.w + film.net.b
//   head.w + head.b + head_scale
template <int Channels>
void ParametricA2FastModel<Channels>::_load_weights(std::vector<float>& weights)
{
  auto it = weights.begin();
  const auto end = weights.end();
  auto take = [&]() -> float {
    if (it == end)
      throw std::runtime_error("ParametricA2FastModel: weight stream exhausted");
    return *it++;
  };

  // Rechannel: 1 → Channels, no bias.
  for (int i = 0; i < Channels; i++)
    _rechannel_w[i] = take();

  for (int li = 0; li < kNumLayers; li++)
  {
    Layer& L = _layers[li];
    const int K = L.kernel_size;

    // Conv1D: Channels → Channels, kernel=K, bias. Col-major per tap.
    for (int i = 0; i < Channels; i++)
      for (int j = 0; j < Channels; j++)
        for (int k = 0; k < K; k++)
          L.conv_w[k * Channels * Channels + j * Channels + i] = take();
    for (int i = 0; i < Channels; i++)
      L.conv_b[i] = take();

    // Input mixin: 1 → Channels, no bias.
    for (int i = 0; i < Channels; i++)
      L.mixin_w[i] = take();

    // layer1x1: Channels → Channels, with bias. Col-major.
    for (int i = 0; i < Channels; i++)
      for (int j = 0; j < Channels; j++)
        L.l1x1_w[j * Channels + i] = take();
    for (int i = 0; i < Channels; i++)
      L.l1x1_b[i] = take();

    // FiLM net: Conv1x1(num_params → 2*Channels, bias).
    // Read order: for i in 2*Channels: for j in num_params.
    for (int i = 0; i < 2 * Channels; i++)
      for (int j = 0; j < _num_params; j++)
        _film_layers[li].net_w[i * _num_params + j] = take();
    for (int i = 0; i < 2 * Channels; i++)
      _film_layers[li].net_b[i] = take();
  }

  // Head rechannel: Channels → 1, kernel=16, bias. Col-major per tap.
  for (int j = 0; j < Channels; j++)
    for (int k = 0; k < kHeadKernelSize; k++)
      _head_w[k][j] = take();
  _head_b = take();
  _head_scale = take();

  if (it != end)
    throw std::runtime_error("ParametricA2FastModel: " + std::to_string(std::distance(it, end))
                             + " trailing weights");
}

// --- SetKnobValues ---
template <int Channels>
void ParametricA2FastModel<Channels>::SetKnobValues(const std::vector<float>& values)
{
  const int n = std::min(static_cast<int>(values.size()), _num_params);
  for (int k = 0; k < n; k++)
    _knob_atoms[k].store(values[k], std::memory_order_relaxed);
}

// --- _recompute_film: rebuild gamma/beta from knob atomics (audio thread, once per buffer) ---
template <int Channels>
void ParametricA2FastModel<Channels>::_recompute_film()
{
  // [Fork] One-pole smooth toward the atomic target each buffer.
  // Eliminates zipper noise when UI sliders move; adds ~20 ms of imperceptible lag.
  float knobs[kMaxParams];
  for (int j = 0; j < _num_params; j++) {
    float target = _knob_atoms[j].load(std::memory_order_relaxed);
    _knob_smoothed[j] += _knob_smooth_coeff * (target - _knob_smoothed[j]);
    knobs[j] = _knob_smoothed[j];
  }

  for (int li = 0; li < kNumLayers; li++)
  {
    const float* w = _film_layers[li].net_w.data(); // (2C × num_params) row-major
    const float* b = _film_layers[li].net_b.data();
    for (int i = 0; i < Channels; i++)
    {
      float g = b[i];
      float beta = b[Channels + i];
      for (int j = 0; j < _num_params; j++)
      {
        g    += w[i           * _num_params + j] * knobs[j];
        beta += w[(Channels + i) * _num_params + j] * knobs[j];
      }
      _film_gamma[li][i] = g;
      _film_beta[li][i]  = beta;
    }
  }
}

// --- Buffer sizing ---
template <int Channels>
void ParametricA2FastModel<Channels>::SetMaxBufferSize(int maxBufferSize)
{
  DSP::SetMaxBufferSize(maxBufferSize);

  _layer_in.assign(static_cast<size_t>(Channels) * maxBufferSize, 0.0f);
  _head_sum.assign(static_cast<size_t>(Channels) * maxBufferSize, 0.0f);
  _z.assign(static_cast<size_t>(Channels) * maxBufferSize, 0.0f);
  _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
  _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);

  for (auto& L : _layers)
  {
  #if NAM_A2_RING_MODE == 1
    L.pow2_size = next_pow2(L.max_lookback + maxBufferSize);
    L.pow2_mask = L.pow2_size - 1;
    L.history.assign(static_cast<size_t>(Channels) * (L.pow2_size + maxBufferSize), 0.0f);
    L.write_pos = L.max_lookback;
  #else
    L.history_cols = 2 * L.max_lookback + maxBufferSize;
    L.history.assign(static_cast<size_t>(Channels) * L.history_cols, 0.0f);
    L.write_pos = L.max_lookback;
  #endif
  }

  const int head_lookback = kHeadKernelSize - 1;
  #if NAM_A2_RING_MODE == 1
  _head_pow2_size = next_pow2(head_lookback + maxBufferSize);
  _head_pow2_mask = _head_pow2_size - 1;
  _head_history.assign(static_cast<size_t>(Channels) * (_head_pow2_size + maxBufferSize), 0.0f);
  _head_write_pos = head_lookback;
  #else
  _head_history_cols = 2 * head_lookback + maxBufferSize;
  _head_history.assign(static_cast<size_t>(Channels) * _head_history_cols, 0.0f);
  _head_write_pos = head_lookback;
  #endif
}

// --- Reset (recomputes DC blocker coefficient for new sample rate) ---
template <int Channels>
void ParametricA2FastModel<Channels>::Reset(double sampleRate, int maxBufferSize)
{
  _dc_alpha = 1.0f - 2.0f * static_cast<float>(M_PI) * 20.0f / static_cast<float>(sampleRate);
  _dc_x_prev = 0.0f;
  _dc_y_prev = 0.0f;
  // [Fork] Per-buffer smooth coeff: 1 - exp(-bufferSize / (sampleRate * smoothTime)).
  // 20 ms gives ~3-4 buffer transitions at 256-sample blocks — imperceptible lag, no zipper.
  const float smoothTimeSamples = static_cast<float>(sampleRate) * 0.020f;
  _knob_smooth_coeff = 1.0f - std::exp(-static_cast<float>(maxBufferSize) / smoothTimeSamples);
  DSP::Reset(sampleRate, maxBufferSize); // calls SetMaxBufferSize then prewarm()
}

// --- Ring write (identical to A2FastModel) ---
template <int Channels>
void ParametricA2FastModel<Channels>::_ring_write(Layer& L, int num_frames)
{
  #if NAM_A2_RING_MODE == 1
  const int mbs = GetMaxBufferSize();
  float* const hist = L.history.data();
  const float* const src = _layer_in.data();
  const int wp = L.write_pos;
  const int first = std::min(num_frames, L.pow2_size - wp);
  std::memcpy(hist + static_cast<size_t>(wp) * Channels, src, static_cast<size_t>(first) * Channels * sizeof(float));
  if (first < num_frames)
    std::memcpy(hist, src + static_cast<size_t>(first) * Channels,
                static_cast<size_t>(num_frames - first) * Channels * sizeof(float));
  std::memcpy(hist + static_cast<size_t>(L.pow2_size) * Channels, hist,
              static_cast<size_t>(mbs) * Channels * sizeof(float));
  L.write_pos = (wp + num_frames) & L.pow2_mask;
  #else
  if (L.write_pos + num_frames > L.history_cols)
  {
    const int keep = L.max_lookback;
    std::memmove(L.history.data(), L.history.data() + static_cast<size_t>(L.write_pos - keep) * Channels,
                 static_cast<size_t>(keep) * Channels * sizeof(float));
    L.write_pos = keep;
  }
  std::memcpy(L.history.data() + static_cast<size_t>(L.write_pos) * Channels, _layer_in.data(),
              static_cast<size_t>(num_frames) * Channels * sizeof(float));
  L.write_pos += num_frames;
  #endif
}

// --- Head ring write: source is _head_sum (the skip-connection accumulator) ---
template <int Channels>
void ParametricA2FastModel<Channels>::_head_ring_write(int num_frames)
{
  // Head input: the sum of every layer's post-activation (standard WaveNet skip connections).
  const float* const src = _head_sum.data();

  #if NAM_A2_RING_MODE == 1
  const int mbs = GetMaxBufferSize();
  float* const hist = _head_history.data();
  const int wp = _head_write_pos;
  const int first = std::min(num_frames, _head_pow2_size - wp);
  std::memcpy(hist + static_cast<size_t>(wp) * Channels, src, static_cast<size_t>(first) * Channels * sizeof(float));
  if (first < num_frames)
    std::memcpy(hist, src + static_cast<size_t>(first) * Channels,
                static_cast<size_t>(num_frames - first) * Channels * sizeof(float));
  std::memcpy(hist + static_cast<size_t>(_head_pow2_size) * Channels, hist,
              static_cast<size_t>(mbs) * Channels * sizeof(float));
  _head_write_pos = (wp + num_frames) & _head_pow2_mask;
  #else
  const int keep = kHeadKernelSize - 1;
  if (_head_write_pos + num_frames > _head_history_cols)
  {
    std::memmove(_head_history.data(),
                 _head_history.data() + static_cast<size_t>(_head_write_pos - keep) * Channels,
                 static_cast<size_t>(keep) * Channels * sizeof(float));
    _head_write_pos = keep;
  }
  std::memcpy(_head_history.data() + static_cast<size_t>(_head_write_pos) * Channels, src,
              static_cast<size_t>(num_frames) * Channels * sizeof(float));
  _head_write_pos += num_frames;
  #endif
}

// --- Per-layer forward: FiLM applied; head_sum accumulated (skip connections) ---
template <int Channels>
template <int KernelSize>
void ParametricA2FastModel<Channels>::_layer_forward_k(Layer& L, int li, const float* cond, int num_frames)
{
  constexpr int K = KernelSize;
  const int D = L.dilation;
  #if NAM_A2_RING_MODE == 1
  const int mask = L.pow2_mask;
  auto tap_base_phys = [&](int taps_back) { return (L.write_pos - num_frames - taps_back * D) & mask; };
  #else
  const int base = L.write_pos - num_frames;
  auto tap_base_phys = [&](int taps_back) { return base - taps_back * D; };
  #endif

  if constexpr (Channels == 3)
  {
    // Preload per-layer FiLM coefficients (constant across all frames in this buffer).
    const float g0 = _film_gamma[li][0], g1 = _film_gamma[li][1], g2 = _film_gamma[li][2];
    const float bt0 = _film_beta[li][0], bt1 = _film_beta[li][1], bt2 = _film_beta[li][2];

    float* z = _z.data();

    // Tap 0: seed z with conv_b.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0=wk[0],w1=wk[1],w2=wk[2],w3=wk[3],w4=wk[4],w5=wk[5],w6=wk[6],w7=wk[7],w8=wk[8];
      const float cb0=L.conv_b[0],cb1=L.conv_b[1],cb2=L.conv_b[2];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 3];
        float* zf = z + static_cast<size_t>(f) * 3;
        zf[0] = cb0 + w0*src[0] + w3*src[1] + w6*src[2];
        zf[1] = cb1 + w1*src[0] + w4*src[1] + w7*src[2];
        zf[2] = cb2 + w2*src[0] + w5*src[1] + w8*src[2];
      }
    }

    // Taps 1..K-2: accumulate.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 9];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0=wk[0],w1=wk[1],w2=wk[2],w3=wk[3],w4=wk[4],w5=wk[5],w6=wk[6],w7=wk[7],w8=wk[8];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 3];
        float* zf = z + static_cast<size_t>(f) * 3;
        zf[0] += w0*src[0] + w3*src[1] + w6*src[2];
        zf[1] += w1*src[0] + w4*src[1] + w7*src[2];
        zf[2] += w2*src[0] + w5*src[1] + w8*src[2];
      }
    }

    // Final tap + mixin + FiLM + LeakyReLU + [head_sum accumulate] + l1x1 residual.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 9];
    const int tap_base_last = tap_base_phys(0);
    const float cw0=wk_last[0],cw1=wk_last[1],cw2=wk_last[2];
    const float cw3=wk_last[3],cw4=wk_last[4],cw5=wk_last[5];
    const float cw6=wk_last[6],cw7=wk_last[7],cw8=wk_last[8];
    const float mw0=L.mixin_w[0],mw1=L.mixin_w[1],mw2=L.mixin_w[2];
    const float lw00=L.l1x1_w[0],lw01=L.l1x1_w[1],lw02=L.l1x1_w[2];
    const float lw10=L.l1x1_w[3],lw11=L.l1x1_w[4],lw12=L.l1x1_w[5];
    const float lw20=L.l1x1_w[6],lw21=L.l1x1_w[7],lw22=L.l1x1_w[8];
    const float lb0=L.l1x1_b[0],lb1=L.l1x1_b[1],lb2=L.l1x1_b[2];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 3];
      const float* zf = z + static_cast<size_t>(f) * 3;
      // Final tap GEMV.
      float a0 = zf[0] + cw0*src[0] + cw3*src[1] + cw6*src[2];
      float a1 = zf[1] + cw1*src[0] + cw4*src[1] + cw7*src[2];
      float a2 = zf[2] + cw2*src[0] + cw5*src[1] + cw8*src[2];
      // Mixin.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      // FiLM: gamma * (conv+mixin) + beta — constant per buffer, 6 FMAs.
      a0 = g0 * a0 + bt0;
      a1 = g1 * a1 + bt1;
      a2 = g2 * a2 + bt2;
      // LeakyReLU.
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      // Skip connections: accumulate this layer's post-activation into the head sum
      // (same site the static A2FastModel uses -- after FiLM+LeakyReLU, BEFORE the
      // l1x1 residual).
      float* __restrict__ hsum = &_head_sum[static_cast<size_t>(f) * 3];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 3];
      lin[0] += lb0 + lw00*a0 + lw10*a1 + lw20*a2;
      lin[1] += lb1 + lw01*a0 + lw11*a1 + lw21*a2;
      lin[2] += lb2 + lw02*a0 + lw12*a1 + lw22*a2;
    }
  }
  // [Fork] BEGIN: hand-unrolled Channels==4 kernel, mirrors the Channels==3 branch
  // above (adds FiLM gamma/beta vs. the non-parametric A2FastModel version) widened from a
  // 3x3 GEMV (9 weights/tap) to a 4x4 GEMV (16 weights/tap, 4 accumulators). Same
  // column-major weight convention: w[j*Channels + i] for output row i, input col j.
  // Verified numerically against the generic ParametricWaveNet reference by
  // test_parametric_matches_generic_medium.
  else if constexpr (Channels == 4)
  {
    // Preload per-layer FiLM coefficients (constant across all frames in this buffer).
    const float g0 = _film_gamma[li][0], g1 = _film_gamma[li][1], g2 = _film_gamma[li][2], g3 = _film_gamma[li][3];
    const float bt0 = _film_beta[li][0], bt1 = _film_beta[li][1], bt2 = _film_beta[li][2], bt3 = _film_beta[li][3];

    float* z = _z.data();

    // Tap 0: seed z with conv_b.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3];
      const float w4 = wk[4], w5 = wk[5], w6 = wk[6], w7 = wk[7];
      const float w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11];
      const float w12 = wk[12], w13 = wk[13], w14 = wk[14], w15 = wk[15];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2], cb3 = L.conv_b[3];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 4];
        float* zf = z + static_cast<size_t>(f) * 4;
        zf[0] = cb0 + w0 * src[0] + w4 * src[1] + w8 * src[2] + w12 * src[3];
        zf[1] = cb1 + w1 * src[0] + w5 * src[1] + w9 * src[2] + w13 * src[3];
        zf[2] = cb2 + w2 * src[0] + w6 * src[1] + w10 * src[2] + w14 * src[3];
        zf[3] = cb3 + w3 * src[0] + w7 * src[1] + w11 * src[2] + w15 * src[3];
      }
    }

    // Taps 1..K-2: accumulate.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 16];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3];
      const float w4 = wk[4], w5 = wk[5], w6 = wk[6], w7 = wk[7];
      const float w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11];
      const float w12 = wk[12], w13 = wk[13], w14 = wk[14], w15 = wk[15];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 4];
        float* zf = z + static_cast<size_t>(f) * 4;
        zf[0] += w0 * src[0] + w4 * src[1] + w8 * src[2] + w12 * src[3];
        zf[1] += w1 * src[0] + w5 * src[1] + w9 * src[2] + w13 * src[3];
        zf[2] += w2 * src[0] + w6 * src[1] + w10 * src[2] + w14 * src[3];
        zf[3] += w3 * src[0] + w7 * src[1] + w11 * src[2] + w15 * src[3];
      }
    }

    // Final tap + mixin + FiLM + LeakyReLU + [head_sum accumulate] + l1x1 residual.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 16];
    const int tap_base_last = tap_base_phys(0);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2], cw3 = wk_last[3];
    const float cw4 = wk_last[4], cw5 = wk_last[5], cw6 = wk_last[6], cw7 = wk_last[7];
    const float cw8 = wk_last[8], cw9 = wk_last[9], cw10 = wk_last[10], cw11 = wk_last[11];
    const float cw12 = wk_last[12], cw13 = wk_last[13], cw14 = wk_last[14], cw15 = wk_last[15];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2], mw3 = L.mixin_w[3];
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2], lw03 = L.l1x1_w[3];
    const float lw10 = L.l1x1_w[4], lw11 = L.l1x1_w[5], lw12 = L.l1x1_w[6], lw13 = L.l1x1_w[7];
    const float lw20 = L.l1x1_w[8], lw21 = L.l1x1_w[9], lw22 = L.l1x1_w[10], lw23 = L.l1x1_w[11];
    const float lw30 = L.l1x1_w[12], lw31 = L.l1x1_w[13], lw32 = L.l1x1_w[14], lw33 = L.l1x1_w[15];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2], lb3 = L.l1x1_b[3];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 4];
      const float* zf = z + static_cast<size_t>(f) * 4;
      // Final tap GEMV.
      float a0 = zf[0] + cw0 * src[0] + cw4 * src[1] + cw8 * src[2] + cw12 * src[3];
      float a1 = zf[1] + cw1 * src[0] + cw5 * src[1] + cw9 * src[2] + cw13 * src[3];
      float a2 = zf[2] + cw2 * src[0] + cw6 * src[1] + cw10 * src[2] + cw14 * src[3];
      float a3 = zf[3] + cw3 * src[0] + cw7 * src[1] + cw11 * src[2] + cw15 * src[3];
      // Mixin.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a3 += mw3 * cf;
      // FiLM: gamma * (conv+mixin) + beta.
      a0 = g0 * a0 + bt0;
      a1 = g1 * a1 + bt1;
      a2 = g2 * a2 + bt2;
      a3 = g3 * a3 + bt3;
      // LeakyReLU.
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      a3 = (a3 >= 0.0f) ? a3 : a3 * kLeakySlope;
      // Skip connections: accumulate this layer's post-activation into the head sum
      // (same site the static A2FastModel uses -- after FiLM+LeakyReLU, BEFORE the
      // l1x1 residual).
      float* __restrict__ hsum = &_head_sum[static_cast<size_t>(f) * 4];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      hsum[3] += a3;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 4];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2 + lw30 * a3;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2 + lw31 * a3;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2 + lw32 * a3;
      lin[3] += lb3 + lw03 * a0 + lw13 * a1 + lw23 * a2 + lw33 * a3;
    }
  }
  // [Fork] END: hand-unrolled Channels==4 kernel
  else if constexpr (Channels == 5)
  {
    // Preload per-layer FiLM coefficients (constant across all frames in this buffer).
    const float g0 = _film_gamma[li][0], g1 = _film_gamma[li][1], g2 = _film_gamma[li][2], g3 = _film_gamma[li][3], g4 = _film_gamma[li][4];
    const float bt0 = _film_beta[li][0], bt1 = _film_beta[li][1], bt2 = _film_beta[li][2], bt3 = _film_beta[li][3], bt4 = _film_beta[li][4];

    float* z = _z.data();

    // Tap 0: seed z with conv_b.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4];
      const float w5 = wk[5], w6 = wk[6], w7 = wk[7], w8 = wk[8], w9 = wk[9];
      const float w10 = wk[10], w11 = wk[11], w12 = wk[12], w13 = wk[13], w14 = wk[14];
      const float w15 = wk[15], w16 = wk[16], w17 = wk[17], w18 = wk[18], w19 = wk[19];
      const float w20 = wk[20], w21 = wk[21], w22 = wk[22], w23 = wk[23], w24 = wk[24];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2], cb3 = L.conv_b[3], cb4 = L.conv_b[4];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 5];
        float* zf = z + static_cast<size_t>(f) * 5;
        zf[0] = cb0 + w0 * src[0] + w5 * src[1] + w10 * src[2] + w15 * src[3] + w20 * src[4];
        zf[1] = cb1 + w1 * src[0] + w6 * src[1] + w11 * src[2] + w16 * src[3] + w21 * src[4];
        zf[2] = cb2 + w2 * src[0] + w7 * src[1] + w12 * src[2] + w17 * src[3] + w22 * src[4];
        zf[3] = cb3 + w3 * src[0] + w8 * src[1] + w13 * src[2] + w18 * src[3] + w23 * src[4];
        zf[4] = cb4 + w4 * src[0] + w9 * src[1] + w14 * src[2] + w19 * src[3] + w24 * src[4];
      }
    }

    // Taps 1..K-2: accumulate.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 25];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4];
      const float w5 = wk[5], w6 = wk[6], w7 = wk[7], w8 = wk[8], w9 = wk[9];
      const float w10 = wk[10], w11 = wk[11], w12 = wk[12], w13 = wk[13], w14 = wk[14];
      const float w15 = wk[15], w16 = wk[16], w17 = wk[17], w18 = wk[18], w19 = wk[19];
      const float w20 = wk[20], w21 = wk[21], w22 = wk[22], w23 = wk[23], w24 = wk[24];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 5];
        float* zf = z + static_cast<size_t>(f) * 5;
        zf[0] += w0 * src[0] + w5 * src[1] + w10 * src[2] + w15 * src[3] + w20 * src[4];
        zf[1] += w1 * src[0] + w6 * src[1] + w11 * src[2] + w16 * src[3] + w21 * src[4];
        zf[2] += w2 * src[0] + w7 * src[1] + w12 * src[2] + w17 * src[3] + w22 * src[4];
        zf[3] += w3 * src[0] + w8 * src[1] + w13 * src[2] + w18 * src[3] + w23 * src[4];
        zf[4] += w4 * src[0] + w9 * src[1] + w14 * src[2] + w19 * src[3] + w24 * src[4];
      }
    }

    // Final tap + mixin + FiLM + LeakyReLU + [head_sum accumulate] + l1x1 residual.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 25];
    const int tap_base_last = tap_base_phys(0);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2], cw3 = wk_last[3], cw4 = wk_last[4];
    const float cw5 = wk_last[5], cw6 = wk_last[6], cw7 = wk_last[7], cw8 = wk_last[8], cw9 = wk_last[9];
    const float cw10 = wk_last[10], cw11 = wk_last[11], cw12 = wk_last[12], cw13 = wk_last[13], cw14 = wk_last[14];
    const float cw15 = wk_last[15], cw16 = wk_last[16], cw17 = wk_last[17], cw18 = wk_last[18], cw19 = wk_last[19];
    const float cw20 = wk_last[20], cw21 = wk_last[21], cw22 = wk_last[22], cw23 = wk_last[23], cw24 = wk_last[24];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2], mw3 = L.mixin_w[3], mw4 = L.mixin_w[4];
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2], lw03 = L.l1x1_w[3], lw04 = L.l1x1_w[4];
    const float lw10 = L.l1x1_w[5], lw11 = L.l1x1_w[6], lw12 = L.l1x1_w[7], lw13 = L.l1x1_w[8], lw14 = L.l1x1_w[9];
    const float lw20 = L.l1x1_w[10], lw21 = L.l1x1_w[11], lw22 = L.l1x1_w[12], lw23 = L.l1x1_w[13], lw24 = L.l1x1_w[14];
    const float lw30 = L.l1x1_w[15], lw31 = L.l1x1_w[16], lw32 = L.l1x1_w[17], lw33 = L.l1x1_w[18], lw34 = L.l1x1_w[19];
    const float lw40 = L.l1x1_w[20], lw41 = L.l1x1_w[21], lw42 = L.l1x1_w[22], lw43 = L.l1x1_w[23], lw44 = L.l1x1_w[24];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2], lb3 = L.l1x1_b[3], lb4 = L.l1x1_b[4];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 5];
      const float* zf = z + static_cast<size_t>(f) * 5;
      // Final tap GEMV.
      float a0 = zf[0] + cw0 * src[0] + cw5 * src[1] + cw10 * src[2] + cw15 * src[3] + cw20 * src[4];
      float a1 = zf[1] + cw1 * src[0] + cw6 * src[1] + cw11 * src[2] + cw16 * src[3] + cw21 * src[4];
      float a2 = zf[2] + cw2 * src[0] + cw7 * src[1] + cw12 * src[2] + cw17 * src[3] + cw22 * src[4];
      float a3 = zf[3] + cw3 * src[0] + cw8 * src[1] + cw13 * src[2] + cw18 * src[3] + cw23 * src[4];
      float a4 = zf[4] + cw4 * src[0] + cw9 * src[1] + cw14 * src[2] + cw19 * src[3] + cw24 * src[4];
      // Mixin.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a3 += mw3 * cf;
      a4 += mw4 * cf;
      // FiLM: gamma * (conv+mixin) + beta.
      a0 = g0 * a0 + bt0;
      a1 = g1 * a1 + bt1;
      a2 = g2 * a2 + bt2;
      a3 = g3 * a3 + bt3;
      a4 = g4 * a4 + bt4;
      // LeakyReLU.
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      a3 = (a3 >= 0.0f) ? a3 : a3 * kLeakySlope;
      a4 = (a4 >= 0.0f) ? a4 : a4 * kLeakySlope;
      // Skip connections: accumulate this layer's post-activation into the head sum
      // (same site the static A2FastModel uses -- after FiLM+LeakyReLU, BEFORE the
      // l1x1 residual).
      float* __restrict__ hsum = &_head_sum[static_cast<size_t>(f) * 5];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      hsum[3] += a3;
      hsum[4] += a4;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 5];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2 + lw30 * a3 + lw40 * a4;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2 + lw31 * a3 + lw41 * a4;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2 + lw32 * a3 + lw42 * a4;
      lin[3] += lb3 + lw03 * a0 + lw13 * a1 + lw23 * a2 + lw33 * a3 + lw43 * a4;
      lin[4] += lb4 + lw04 * a0 + lw14 * a1 + lw24 * a2 + lw34 * a3 + lw44 * a4;
    }
  }
  // [Fork] END: hand-unrolled Channels==5 kernel
  else if constexpr (Channels == 6)
  {
    // Preload per-layer FiLM coefficients (constant across all frames in this buffer).
    const float g0 = _film_gamma[li][0], g1 = _film_gamma[li][1], g2 = _film_gamma[li][2], g3 = _film_gamma[li][3], g4 = _film_gamma[li][4], g5 = _film_gamma[li][5];
    const float bt0 = _film_beta[li][0], bt1 = _film_beta[li][1], bt2 = _film_beta[li][2], bt3 = _film_beta[li][3], bt4 = _film_beta[li][4], bt5 = _film_beta[li][5];

    float* z = _z.data();

    // Tap 0: seed z with conv_b.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11];
      const float w12 = wk[12], w13 = wk[13], w14 = wk[14], w15 = wk[15], w16 = wk[16], w17 = wk[17];
      const float w18 = wk[18], w19 = wk[19], w20 = wk[20], w21 = wk[21], w22 = wk[22], w23 = wk[23];
      const float w24 = wk[24], w25 = wk[25], w26 = wk[26], w27 = wk[27], w28 = wk[28], w29 = wk[29];
      const float w30 = wk[30], w31 = wk[31], w32 = wk[32], w33 = wk[33], w34 = wk[34], w35 = wk[35];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2], cb3 = L.conv_b[3], cb4 = L.conv_b[4], cb5 = L.conv_b[5];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 6];
        float* zf = z + static_cast<size_t>(f) * 6;
        zf[0] = cb0 + w0 * src[0] + w6 * src[1] + w12 * src[2] + w18 * src[3] + w24 * src[4] + w30 * src[5];
        zf[1] = cb1 + w1 * src[0] + w7 * src[1] + w13 * src[2] + w19 * src[3] + w25 * src[4] + w31 * src[5];
        zf[2] = cb2 + w2 * src[0] + w8 * src[1] + w14 * src[2] + w20 * src[3] + w26 * src[4] + w32 * src[5];
        zf[3] = cb3 + w3 * src[0] + w9 * src[1] + w15 * src[2] + w21 * src[3] + w27 * src[4] + w33 * src[5];
        zf[4] = cb4 + w4 * src[0] + w10 * src[1] + w16 * src[2] + w22 * src[3] + w28 * src[4] + w34 * src[5];
        zf[5] = cb5 + w5 * src[0] + w11 * src[1] + w17 * src[2] + w23 * src[3] + w29 * src[4] + w35 * src[5];
      }
    }

    // Taps 1..K-2: accumulate.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 36];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11];
      const float w12 = wk[12], w13 = wk[13], w14 = wk[14], w15 = wk[15], w16 = wk[16], w17 = wk[17];
      const float w18 = wk[18], w19 = wk[19], w20 = wk[20], w21 = wk[21], w22 = wk[22], w23 = wk[23];
      const float w24 = wk[24], w25 = wk[25], w26 = wk[26], w27 = wk[27], w28 = wk[28], w29 = wk[29];
      const float w30 = wk[30], w31 = wk[31], w32 = wk[32], w33 = wk[33], w34 = wk[34], w35 = wk[35];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 6];
        float* zf = z + static_cast<size_t>(f) * 6;
        zf[0] += w0 * src[0] + w6 * src[1] + w12 * src[2] + w18 * src[3] + w24 * src[4] + w30 * src[5];
        zf[1] += w1 * src[0] + w7 * src[1] + w13 * src[2] + w19 * src[3] + w25 * src[4] + w31 * src[5];
        zf[2] += w2 * src[0] + w8 * src[1] + w14 * src[2] + w20 * src[3] + w26 * src[4] + w32 * src[5];
        zf[3] += w3 * src[0] + w9 * src[1] + w15 * src[2] + w21 * src[3] + w27 * src[4] + w33 * src[5];
        zf[4] += w4 * src[0] + w10 * src[1] + w16 * src[2] + w22 * src[3] + w28 * src[4] + w34 * src[5];
        zf[5] += w5 * src[0] + w11 * src[1] + w17 * src[2] + w23 * src[3] + w29 * src[4] + w35 * src[5];
      }
    }

    // Final tap + mixin + FiLM + LeakyReLU + [head_sum accumulate] + l1x1 residual.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 36];
    const int tap_base_last = tap_base_phys(0);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2], cw3 = wk_last[3], cw4 = wk_last[4], cw5 = wk_last[5];
    const float cw6 = wk_last[6], cw7 = wk_last[7], cw8 = wk_last[8], cw9 = wk_last[9], cw10 = wk_last[10], cw11 = wk_last[11];
    const float cw12 = wk_last[12], cw13 = wk_last[13], cw14 = wk_last[14], cw15 = wk_last[15], cw16 = wk_last[16], cw17 = wk_last[17];
    const float cw18 = wk_last[18], cw19 = wk_last[19], cw20 = wk_last[20], cw21 = wk_last[21], cw22 = wk_last[22], cw23 = wk_last[23];
    const float cw24 = wk_last[24], cw25 = wk_last[25], cw26 = wk_last[26], cw27 = wk_last[27], cw28 = wk_last[28], cw29 = wk_last[29];
    const float cw30 = wk_last[30], cw31 = wk_last[31], cw32 = wk_last[32], cw33 = wk_last[33], cw34 = wk_last[34], cw35 = wk_last[35];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2], mw3 = L.mixin_w[3], mw4 = L.mixin_w[4], mw5 = L.mixin_w[5];
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2], lw03 = L.l1x1_w[3], lw04 = L.l1x1_w[4], lw05 = L.l1x1_w[5];
    const float lw10 = L.l1x1_w[6], lw11 = L.l1x1_w[7], lw12 = L.l1x1_w[8], lw13 = L.l1x1_w[9], lw14 = L.l1x1_w[10], lw15 = L.l1x1_w[11];
    const float lw20 = L.l1x1_w[12], lw21 = L.l1x1_w[13], lw22 = L.l1x1_w[14], lw23 = L.l1x1_w[15], lw24 = L.l1x1_w[16], lw25 = L.l1x1_w[17];
    const float lw30 = L.l1x1_w[18], lw31 = L.l1x1_w[19], lw32 = L.l1x1_w[20], lw33 = L.l1x1_w[21], lw34 = L.l1x1_w[22], lw35 = L.l1x1_w[23];
    const float lw40 = L.l1x1_w[24], lw41 = L.l1x1_w[25], lw42 = L.l1x1_w[26], lw43 = L.l1x1_w[27], lw44 = L.l1x1_w[28], lw45 = L.l1x1_w[29];
    const float lw50 = L.l1x1_w[30], lw51 = L.l1x1_w[31], lw52 = L.l1x1_w[32], lw53 = L.l1x1_w[33], lw54 = L.l1x1_w[34], lw55 = L.l1x1_w[35];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2], lb3 = L.l1x1_b[3], lb4 = L.l1x1_b[4], lb5 = L.l1x1_b[5];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 6];
      const float* zf = z + static_cast<size_t>(f) * 6;
      // Final tap GEMV.
      float a0 = zf[0] + cw0 * src[0] + cw6 * src[1] + cw12 * src[2] + cw18 * src[3] + cw24 * src[4] + cw30 * src[5];
      float a1 = zf[1] + cw1 * src[0] + cw7 * src[1] + cw13 * src[2] + cw19 * src[3] + cw25 * src[4] + cw31 * src[5];
      float a2 = zf[2] + cw2 * src[0] + cw8 * src[1] + cw14 * src[2] + cw20 * src[3] + cw26 * src[4] + cw32 * src[5];
      float a3 = zf[3] + cw3 * src[0] + cw9 * src[1] + cw15 * src[2] + cw21 * src[3] + cw27 * src[4] + cw33 * src[5];
      float a4 = zf[4] + cw4 * src[0] + cw10 * src[1] + cw16 * src[2] + cw22 * src[3] + cw28 * src[4] + cw34 * src[5];
      float a5 = zf[5] + cw5 * src[0] + cw11 * src[1] + cw17 * src[2] + cw23 * src[3] + cw29 * src[4] + cw35 * src[5];
      // Mixin.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a3 += mw3 * cf;
      a4 += mw4 * cf;
      a5 += mw5 * cf;
      // FiLM: gamma * (conv+mixin) + beta.
      a0 = g0 * a0 + bt0;
      a1 = g1 * a1 + bt1;
      a2 = g2 * a2 + bt2;
      a3 = g3 * a3 + bt3;
      a4 = g4 * a4 + bt4;
      a5 = g5 * a5 + bt5;
      // LeakyReLU.
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      a3 = (a3 >= 0.0f) ? a3 : a3 * kLeakySlope;
      a4 = (a4 >= 0.0f) ? a4 : a4 * kLeakySlope;
      a5 = (a5 >= 0.0f) ? a5 : a5 * kLeakySlope;
      // Skip connections: accumulate this layer's post-activation into the head sum
      // (same site the static A2FastModel uses -- after FiLM+LeakyReLU, BEFORE the
      // l1x1 residual).
      float* __restrict__ hsum = &_head_sum[static_cast<size_t>(f) * 6];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      hsum[3] += a3;
      hsum[4] += a4;
      hsum[5] += a5;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 6];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2 + lw30 * a3 + lw40 * a4 + lw50 * a5;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2 + lw31 * a3 + lw41 * a4 + lw51 * a5;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2 + lw32 * a3 + lw42 * a4 + lw52 * a5;
      lin[3] += lb3 + lw03 * a0 + lw13 * a1 + lw23 * a2 + lw33 * a3 + lw43 * a4 + lw53 * a5;
      lin[4] += lb4 + lw04 * a0 + lw14 * a1 + lw24 * a2 + lw34 * a3 + lw44 * a4 + lw54 * a5;
      lin[5] += lb5 + lw05 * a0 + lw15 * a1 + lw25 * a2 + lw35 * a3 + lw45 * a4 + lw55 * a5;
    }
  }
  // [Fork] END: hand-unrolled Channels==6 kernel
  else if constexpr (Channels == 7)
  {
    // Preload per-layer FiLM coefficients (constant across all frames in this buffer).
    const float g0 = _film_gamma[li][0], g1 = _film_gamma[li][1], g2 = _film_gamma[li][2], g3 = _film_gamma[li][3], g4 = _film_gamma[li][4], g5 = _film_gamma[li][5], g6 = _film_gamma[li][6];
    const float bt0 = _film_beta[li][0], bt1 = _film_beta[li][1], bt2 = _film_beta[li][2], bt3 = _film_beta[li][3], bt4 = _film_beta[li][4], bt5 = _film_beta[li][5], bt6 = _film_beta[li][6];

    float* z = _z.data();

    // Tap 0: seed z with conv_b.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5], w6 = wk[6];
      const float w7 = wk[7], w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11], w12 = wk[12], w13 = wk[13];
      const float w14 = wk[14], w15 = wk[15], w16 = wk[16], w17 = wk[17], w18 = wk[18], w19 = wk[19], w20 = wk[20];
      const float w21 = wk[21], w22 = wk[22], w23 = wk[23], w24 = wk[24], w25 = wk[25], w26 = wk[26], w27 = wk[27];
      const float w28 = wk[28], w29 = wk[29], w30 = wk[30], w31 = wk[31], w32 = wk[32], w33 = wk[33], w34 = wk[34];
      const float w35 = wk[35], w36 = wk[36], w37 = wk[37], w38 = wk[38], w39 = wk[39], w40 = wk[40], w41 = wk[41];
      const float w42 = wk[42], w43 = wk[43], w44 = wk[44], w45 = wk[45], w46 = wk[46], w47 = wk[47], w48 = wk[48];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2], cb3 = L.conv_b[3], cb4 = L.conv_b[4], cb5 = L.conv_b[5], cb6 = L.conv_b[6];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 7];
        float* zf = z + static_cast<size_t>(f) * 7;
        zf[0] = cb0 + w0 * src[0] + w7 * src[1] + w14 * src[2] + w21 * src[3] + w28 * src[4] + w35 * src[5] + w42 * src[6];
        zf[1] = cb1 + w1 * src[0] + w8 * src[1] + w15 * src[2] + w22 * src[3] + w29 * src[4] + w36 * src[5] + w43 * src[6];
        zf[2] = cb2 + w2 * src[0] + w9 * src[1] + w16 * src[2] + w23 * src[3] + w30 * src[4] + w37 * src[5] + w44 * src[6];
        zf[3] = cb3 + w3 * src[0] + w10 * src[1] + w17 * src[2] + w24 * src[3] + w31 * src[4] + w38 * src[5] + w45 * src[6];
        zf[4] = cb4 + w4 * src[0] + w11 * src[1] + w18 * src[2] + w25 * src[3] + w32 * src[4] + w39 * src[5] + w46 * src[6];
        zf[5] = cb5 + w5 * src[0] + w12 * src[1] + w19 * src[2] + w26 * src[3] + w33 * src[4] + w40 * src[5] + w47 * src[6];
        zf[6] = cb6 + w6 * src[0] + w13 * src[1] + w20 * src[2] + w27 * src[3] + w34 * src[4] + w41 * src[5] + w48 * src[6];
      }
    }

    // Taps 1..K-2: accumulate.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 49];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5], w6 = wk[6];
      const float w7 = wk[7], w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11], w12 = wk[12], w13 = wk[13];
      const float w14 = wk[14], w15 = wk[15], w16 = wk[16], w17 = wk[17], w18 = wk[18], w19 = wk[19], w20 = wk[20];
      const float w21 = wk[21], w22 = wk[22], w23 = wk[23], w24 = wk[24], w25 = wk[25], w26 = wk[26], w27 = wk[27];
      const float w28 = wk[28], w29 = wk[29], w30 = wk[30], w31 = wk[31], w32 = wk[32], w33 = wk[33], w34 = wk[34];
      const float w35 = wk[35], w36 = wk[36], w37 = wk[37], w38 = wk[38], w39 = wk[39], w40 = wk[40], w41 = wk[41];
      const float w42 = wk[42], w43 = wk[43], w44 = wk[44], w45 = wk[45], w46 = wk[46], w47 = wk[47], w48 = wk[48];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 7];
        float* zf = z + static_cast<size_t>(f) * 7;
        zf[0] += w0 * src[0] + w7 * src[1] + w14 * src[2] + w21 * src[3] + w28 * src[4] + w35 * src[5] + w42 * src[6];
        zf[1] += w1 * src[0] + w8 * src[1] + w15 * src[2] + w22 * src[3] + w29 * src[4] + w36 * src[5] + w43 * src[6];
        zf[2] += w2 * src[0] + w9 * src[1] + w16 * src[2] + w23 * src[3] + w30 * src[4] + w37 * src[5] + w44 * src[6];
        zf[3] += w3 * src[0] + w10 * src[1] + w17 * src[2] + w24 * src[3] + w31 * src[4] + w38 * src[5] + w45 * src[6];
        zf[4] += w4 * src[0] + w11 * src[1] + w18 * src[2] + w25 * src[3] + w32 * src[4] + w39 * src[5] + w46 * src[6];
        zf[5] += w5 * src[0] + w12 * src[1] + w19 * src[2] + w26 * src[3] + w33 * src[4] + w40 * src[5] + w47 * src[6];
        zf[6] += w6 * src[0] + w13 * src[1] + w20 * src[2] + w27 * src[3] + w34 * src[4] + w41 * src[5] + w48 * src[6];
      }
    }

    // Final tap + mixin + FiLM + LeakyReLU + [head_sum accumulate] + l1x1 residual.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 49];
    const int tap_base_last = tap_base_phys(0);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2], cw3 = wk_last[3], cw4 = wk_last[4], cw5 = wk_last[5], cw6 = wk_last[6];
    const float cw7 = wk_last[7], cw8 = wk_last[8], cw9 = wk_last[9], cw10 = wk_last[10], cw11 = wk_last[11], cw12 = wk_last[12], cw13 = wk_last[13];
    const float cw14 = wk_last[14], cw15 = wk_last[15], cw16 = wk_last[16], cw17 = wk_last[17], cw18 = wk_last[18], cw19 = wk_last[19], cw20 = wk_last[20];
    const float cw21 = wk_last[21], cw22 = wk_last[22], cw23 = wk_last[23], cw24 = wk_last[24], cw25 = wk_last[25], cw26 = wk_last[26], cw27 = wk_last[27];
    const float cw28 = wk_last[28], cw29 = wk_last[29], cw30 = wk_last[30], cw31 = wk_last[31], cw32 = wk_last[32], cw33 = wk_last[33], cw34 = wk_last[34];
    const float cw35 = wk_last[35], cw36 = wk_last[36], cw37 = wk_last[37], cw38 = wk_last[38], cw39 = wk_last[39], cw40 = wk_last[40], cw41 = wk_last[41];
    const float cw42 = wk_last[42], cw43 = wk_last[43], cw44 = wk_last[44], cw45 = wk_last[45], cw46 = wk_last[46], cw47 = wk_last[47], cw48 = wk_last[48];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2], mw3 = L.mixin_w[3], mw4 = L.mixin_w[4], mw5 = L.mixin_w[5], mw6 = L.mixin_w[6];
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2], lw03 = L.l1x1_w[3], lw04 = L.l1x1_w[4], lw05 = L.l1x1_w[5], lw06 = L.l1x1_w[6];
    const float lw10 = L.l1x1_w[7], lw11 = L.l1x1_w[8], lw12 = L.l1x1_w[9], lw13 = L.l1x1_w[10], lw14 = L.l1x1_w[11], lw15 = L.l1x1_w[12], lw16 = L.l1x1_w[13];
    const float lw20 = L.l1x1_w[14], lw21 = L.l1x1_w[15], lw22 = L.l1x1_w[16], lw23 = L.l1x1_w[17], lw24 = L.l1x1_w[18], lw25 = L.l1x1_w[19], lw26 = L.l1x1_w[20];
    const float lw30 = L.l1x1_w[21], lw31 = L.l1x1_w[22], lw32 = L.l1x1_w[23], lw33 = L.l1x1_w[24], lw34 = L.l1x1_w[25], lw35 = L.l1x1_w[26], lw36 = L.l1x1_w[27];
    const float lw40 = L.l1x1_w[28], lw41 = L.l1x1_w[29], lw42 = L.l1x1_w[30], lw43 = L.l1x1_w[31], lw44 = L.l1x1_w[32], lw45 = L.l1x1_w[33], lw46 = L.l1x1_w[34];
    const float lw50 = L.l1x1_w[35], lw51 = L.l1x1_w[36], lw52 = L.l1x1_w[37], lw53 = L.l1x1_w[38], lw54 = L.l1x1_w[39], lw55 = L.l1x1_w[40], lw56 = L.l1x1_w[41];
    const float lw60 = L.l1x1_w[42], lw61 = L.l1x1_w[43], lw62 = L.l1x1_w[44], lw63 = L.l1x1_w[45], lw64 = L.l1x1_w[46], lw65 = L.l1x1_w[47], lw66 = L.l1x1_w[48];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2], lb3 = L.l1x1_b[3], lb4 = L.l1x1_b[4], lb5 = L.l1x1_b[5], lb6 = L.l1x1_b[6];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 7];
      const float* zf = z + static_cast<size_t>(f) * 7;
      // Final tap GEMV.
      float a0 = zf[0] + cw0 * src[0] + cw7 * src[1] + cw14 * src[2] + cw21 * src[3] + cw28 * src[4] + cw35 * src[5] + cw42 * src[6];
      float a1 = zf[1] + cw1 * src[0] + cw8 * src[1] + cw15 * src[2] + cw22 * src[3] + cw29 * src[4] + cw36 * src[5] + cw43 * src[6];
      float a2 = zf[2] + cw2 * src[0] + cw9 * src[1] + cw16 * src[2] + cw23 * src[3] + cw30 * src[4] + cw37 * src[5] + cw44 * src[6];
      float a3 = zf[3] + cw3 * src[0] + cw10 * src[1] + cw17 * src[2] + cw24 * src[3] + cw31 * src[4] + cw38 * src[5] + cw45 * src[6];
      float a4 = zf[4] + cw4 * src[0] + cw11 * src[1] + cw18 * src[2] + cw25 * src[3] + cw32 * src[4] + cw39 * src[5] + cw46 * src[6];
      float a5 = zf[5] + cw5 * src[0] + cw12 * src[1] + cw19 * src[2] + cw26 * src[3] + cw33 * src[4] + cw40 * src[5] + cw47 * src[6];
      float a6 = zf[6] + cw6 * src[0] + cw13 * src[1] + cw20 * src[2] + cw27 * src[3] + cw34 * src[4] + cw41 * src[5] + cw48 * src[6];
      // Mixin.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a3 += mw3 * cf;
      a4 += mw4 * cf;
      a5 += mw5 * cf;
      a6 += mw6 * cf;
      // FiLM: gamma * (conv+mixin) + beta.
      a0 = g0 * a0 + bt0;
      a1 = g1 * a1 + bt1;
      a2 = g2 * a2 + bt2;
      a3 = g3 * a3 + bt3;
      a4 = g4 * a4 + bt4;
      a5 = g5 * a5 + bt5;
      a6 = g6 * a6 + bt6;
      // LeakyReLU.
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      a3 = (a3 >= 0.0f) ? a3 : a3 * kLeakySlope;
      a4 = (a4 >= 0.0f) ? a4 : a4 * kLeakySlope;
      a5 = (a5 >= 0.0f) ? a5 : a5 * kLeakySlope;
      a6 = (a6 >= 0.0f) ? a6 : a6 * kLeakySlope;
      // Skip connections: accumulate this layer's post-activation into the head sum
      // (same site the static A2FastModel uses -- after FiLM+LeakyReLU, BEFORE the
      // l1x1 residual).
      float* __restrict__ hsum = &_head_sum[static_cast<size_t>(f) * 7];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      hsum[3] += a3;
      hsum[4] += a4;
      hsum[5] += a5;
      hsum[6] += a6;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 7];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2 + lw30 * a3 + lw40 * a4 + lw50 * a5 + lw60 * a6;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2 + lw31 * a3 + lw41 * a4 + lw51 * a5 + lw61 * a6;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2 + lw32 * a3 + lw42 * a4 + lw52 * a5 + lw62 * a6;
      lin[3] += lb3 + lw03 * a0 + lw13 * a1 + lw23 * a2 + lw33 * a3 + lw43 * a4 + lw53 * a5 + lw63 * a6;
      lin[4] += lb4 + lw04 * a0 + lw14 * a1 + lw24 * a2 + lw34 * a3 + lw44 * a4 + lw54 * a5 + lw64 * a6;
      lin[5] += lb5 + lw05 * a0 + lw15 * a1 + lw25 * a2 + lw35 * a3 + lw45 * a4 + lw55 * a5 + lw65 * a6;
      lin[6] += lb6 + lw06 * a0 + lw16 * a1 + lw26 * a2 + lw36 * a3 + lw46 * a4 + lw56 * a5 + lw66 * a6;
    }
  }
  // [Fork] END: hand-unrolled Channels==7 kernel

  else if constexpr (Channels == 8)
  {
    // Preload per-layer FiLM coefficients (constant across all frames in this buffer).
    const float g0 = _film_gamma[li][0], g1 = _film_gamma[li][1], g2 = _film_gamma[li][2], g3 = _film_gamma[li][3], g4 = _film_gamma[li][4], g5 = _film_gamma[li][5], g6 = _film_gamma[li][6], g7 = _film_gamma[li][7];
    const float bt0 = _film_beta[li][0], bt1 = _film_beta[li][1], bt2 = _film_beta[li][2], bt3 = _film_beta[li][3], bt4 = _film_beta[li][4], bt5 = _film_beta[li][5], bt6 = _film_beta[li][6], bt7 = _film_beta[li][7];

    float* z = _z.data();

    // Tap 0: seed z with conv_b.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5], w6 = wk[6], w7 = wk[7];
      const float w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11], w12 = wk[12], w13 = wk[13], w14 = wk[14], w15 = wk[15];
      const float w16 = wk[16], w17 = wk[17], w18 = wk[18], w19 = wk[19], w20 = wk[20], w21 = wk[21], w22 = wk[22], w23 = wk[23];
      const float w24 = wk[24], w25 = wk[25], w26 = wk[26], w27 = wk[27], w28 = wk[28], w29 = wk[29], w30 = wk[30], w31 = wk[31];
      const float w32 = wk[32], w33 = wk[33], w34 = wk[34], w35 = wk[35], w36 = wk[36], w37 = wk[37], w38 = wk[38], w39 = wk[39];
      const float w40 = wk[40], w41 = wk[41], w42 = wk[42], w43 = wk[43], w44 = wk[44], w45 = wk[45], w46 = wk[46], w47 = wk[47];
      const float w48 = wk[48], w49 = wk[49], w50 = wk[50], w51 = wk[51], w52 = wk[52], w53 = wk[53], w54 = wk[54], w55 = wk[55];
      const float w56 = wk[56], w57 = wk[57], w58 = wk[58], w59 = wk[59], w60 = wk[60], w61 = wk[61], w62 = wk[62], w63 = wk[63];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2], cb3 = L.conv_b[3], cb4 = L.conv_b[4], cb5 = L.conv_b[5], cb6 = L.conv_b[6], cb7 = L.conv_b[7];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 8];
        float* zf = z + static_cast<size_t>(f) * 8;
        zf[0] = cb0 + w0 * src[0] + w8 * src[1] + w16 * src[2] + w24 * src[3] + w32 * src[4] + w40 * src[5] + w48 * src[6] + w56 * src[7];
        zf[1] = cb1 + w1 * src[0] + w9 * src[1] + w17 * src[2] + w25 * src[3] + w33 * src[4] + w41 * src[5] + w49 * src[6] + w57 * src[7];
        zf[2] = cb2 + w2 * src[0] + w10 * src[1] + w18 * src[2] + w26 * src[3] + w34 * src[4] + w42 * src[5] + w50 * src[6] + w58 * src[7];
        zf[3] = cb3 + w3 * src[0] + w11 * src[1] + w19 * src[2] + w27 * src[3] + w35 * src[4] + w43 * src[5] + w51 * src[6] + w59 * src[7];
        zf[4] = cb4 + w4 * src[0] + w12 * src[1] + w20 * src[2] + w28 * src[3] + w36 * src[4] + w44 * src[5] + w52 * src[6] + w60 * src[7];
        zf[5] = cb5 + w5 * src[0] + w13 * src[1] + w21 * src[2] + w29 * src[3] + w37 * src[4] + w45 * src[5] + w53 * src[6] + w61 * src[7];
        zf[6] = cb6 + w6 * src[0] + w14 * src[1] + w22 * src[2] + w30 * src[3] + w38 * src[4] + w46 * src[5] + w54 * src[6] + w62 * src[7];
        zf[7] = cb7 + w7 * src[0] + w15 * src[1] + w23 * src[2] + w31 * src[3] + w39 * src[4] + w47 * src[5] + w55 * src[6] + w63 * src[7];
      }
    }

    // Taps 1..K-2: accumulate.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 64];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2], w3 = wk[3], w4 = wk[4], w5 = wk[5], w6 = wk[6], w7 = wk[7];
      const float w8 = wk[8], w9 = wk[9], w10 = wk[10], w11 = wk[11], w12 = wk[12], w13 = wk[13], w14 = wk[14], w15 = wk[15];
      const float w16 = wk[16], w17 = wk[17], w18 = wk[18], w19 = wk[19], w20 = wk[20], w21 = wk[21], w22 = wk[22], w23 = wk[23];
      const float w24 = wk[24], w25 = wk[25], w26 = wk[26], w27 = wk[27], w28 = wk[28], w29 = wk[29], w30 = wk[30], w31 = wk[31];
      const float w32 = wk[32], w33 = wk[33], w34 = wk[34], w35 = wk[35], w36 = wk[36], w37 = wk[37], w38 = wk[38], w39 = wk[39];
      const float w40 = wk[40], w41 = wk[41], w42 = wk[42], w43 = wk[43], w44 = wk[44], w45 = wk[45], w46 = wk[46], w47 = wk[47];
      const float w48 = wk[48], w49 = wk[49], w50 = wk[50], w51 = wk[51], w52 = wk[52], w53 = wk[53], w54 = wk[54], w55 = wk[55];
      const float w56 = wk[56], w57 = wk[57], w58 = wk[58], w59 = wk[59], w60 = wk[60], w61 = wk[61], w62 = wk[62], w63 = wk[63];
      for (int f = 0; f < num_frames; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 8];
        float* zf = z + static_cast<size_t>(f) * 8;
        zf[0] += w0 * src[0] + w8 * src[1] + w16 * src[2] + w24 * src[3] + w32 * src[4] + w40 * src[5] + w48 * src[6] + w56 * src[7];
        zf[1] += w1 * src[0] + w9 * src[1] + w17 * src[2] + w25 * src[3] + w33 * src[4] + w41 * src[5] + w49 * src[6] + w57 * src[7];
        zf[2] += w2 * src[0] + w10 * src[1] + w18 * src[2] + w26 * src[3] + w34 * src[4] + w42 * src[5] + w50 * src[6] + w58 * src[7];
        zf[3] += w3 * src[0] + w11 * src[1] + w19 * src[2] + w27 * src[3] + w35 * src[4] + w43 * src[5] + w51 * src[6] + w59 * src[7];
        zf[4] += w4 * src[0] + w12 * src[1] + w20 * src[2] + w28 * src[3] + w36 * src[4] + w44 * src[5] + w52 * src[6] + w60 * src[7];
        zf[5] += w5 * src[0] + w13 * src[1] + w21 * src[2] + w29 * src[3] + w37 * src[4] + w45 * src[5] + w53 * src[6] + w61 * src[7];
        zf[6] += w6 * src[0] + w14 * src[1] + w22 * src[2] + w30 * src[3] + w38 * src[4] + w46 * src[5] + w54 * src[6] + w62 * src[7];
        zf[7] += w7 * src[0] + w15 * src[1] + w23 * src[2] + w31 * src[3] + w39 * src[4] + w47 * src[5] + w55 * src[6] + w63 * src[7];
      }
    }

    // Final tap + mixin + FiLM + LeakyReLU + [head_sum accumulate] + l1x1 residual.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 64];
    const int tap_base_last = tap_base_phys(0);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2], cw3 = wk_last[3], cw4 = wk_last[4], cw5 = wk_last[5], cw6 = wk_last[6], cw7 = wk_last[7];
    const float cw8 = wk_last[8], cw9 = wk_last[9], cw10 = wk_last[10], cw11 = wk_last[11], cw12 = wk_last[12], cw13 = wk_last[13], cw14 = wk_last[14], cw15 = wk_last[15];
    const float cw16 = wk_last[16], cw17 = wk_last[17], cw18 = wk_last[18], cw19 = wk_last[19], cw20 = wk_last[20], cw21 = wk_last[21], cw22 = wk_last[22], cw23 = wk_last[23];
    const float cw24 = wk_last[24], cw25 = wk_last[25], cw26 = wk_last[26], cw27 = wk_last[27], cw28 = wk_last[28], cw29 = wk_last[29], cw30 = wk_last[30], cw31 = wk_last[31];
    const float cw32 = wk_last[32], cw33 = wk_last[33], cw34 = wk_last[34], cw35 = wk_last[35], cw36 = wk_last[36], cw37 = wk_last[37], cw38 = wk_last[38], cw39 = wk_last[39];
    const float cw40 = wk_last[40], cw41 = wk_last[41], cw42 = wk_last[42], cw43 = wk_last[43], cw44 = wk_last[44], cw45 = wk_last[45], cw46 = wk_last[46], cw47 = wk_last[47];
    const float cw48 = wk_last[48], cw49 = wk_last[49], cw50 = wk_last[50], cw51 = wk_last[51], cw52 = wk_last[52], cw53 = wk_last[53], cw54 = wk_last[54], cw55 = wk_last[55];
    const float cw56 = wk_last[56], cw57 = wk_last[57], cw58 = wk_last[58], cw59 = wk_last[59], cw60 = wk_last[60], cw61 = wk_last[61], cw62 = wk_last[62], cw63 = wk_last[63];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2], mw3 = L.mixin_w[3], mw4 = L.mixin_w[4], mw5 = L.mixin_w[5], mw6 = L.mixin_w[6], mw7 = L.mixin_w[7];
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2], lw03 = L.l1x1_w[3], lw04 = L.l1x1_w[4], lw05 = L.l1x1_w[5], lw06 = L.l1x1_w[6], lw07 = L.l1x1_w[7];
    const float lw10 = L.l1x1_w[8], lw11 = L.l1x1_w[9], lw12 = L.l1x1_w[10], lw13 = L.l1x1_w[11], lw14 = L.l1x1_w[12], lw15 = L.l1x1_w[13], lw16 = L.l1x1_w[14], lw17 = L.l1x1_w[15];
    const float lw20 = L.l1x1_w[16], lw21 = L.l1x1_w[17], lw22 = L.l1x1_w[18], lw23 = L.l1x1_w[19], lw24 = L.l1x1_w[20], lw25 = L.l1x1_w[21], lw26 = L.l1x1_w[22], lw27 = L.l1x1_w[23];
    const float lw30 = L.l1x1_w[24], lw31 = L.l1x1_w[25], lw32 = L.l1x1_w[26], lw33 = L.l1x1_w[27], lw34 = L.l1x1_w[28], lw35 = L.l1x1_w[29], lw36 = L.l1x1_w[30], lw37 = L.l1x1_w[31];
    const float lw40 = L.l1x1_w[32], lw41 = L.l1x1_w[33], lw42 = L.l1x1_w[34], lw43 = L.l1x1_w[35], lw44 = L.l1x1_w[36], lw45 = L.l1x1_w[37], lw46 = L.l1x1_w[38], lw47 = L.l1x1_w[39];
    const float lw50 = L.l1x1_w[40], lw51 = L.l1x1_w[41], lw52 = L.l1x1_w[42], lw53 = L.l1x1_w[43], lw54 = L.l1x1_w[44], lw55 = L.l1x1_w[45], lw56 = L.l1x1_w[46], lw57 = L.l1x1_w[47];
    const float lw60 = L.l1x1_w[48], lw61 = L.l1x1_w[49], lw62 = L.l1x1_w[50], lw63 = L.l1x1_w[51], lw64 = L.l1x1_w[52], lw65 = L.l1x1_w[53], lw66 = L.l1x1_w[54], lw67 = L.l1x1_w[55];
    const float lw70 = L.l1x1_w[56], lw71 = L.l1x1_w[57], lw72 = L.l1x1_w[58], lw73 = L.l1x1_w[59], lw74 = L.l1x1_w[60], lw75 = L.l1x1_w[61], lw76 = L.l1x1_w[62], lw77 = L.l1x1_w[63];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2], lb3 = L.l1x1_b[3], lb4 = L.l1x1_b[4], lb5 = L.l1x1_b[5], lb6 = L.l1x1_b[6], lb7 = L.l1x1_b[7];
    for (int f = 0; f < num_frames; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 8];
      const float* zf = z + static_cast<size_t>(f) * 8;
      // Final tap GEMV.
      float a0 = zf[0] + cw0 * src[0] + cw8 * src[1] + cw16 * src[2] + cw24 * src[3] + cw32 * src[4] + cw40 * src[5] + cw48 * src[6] + cw56 * src[7];
      float a1 = zf[1] + cw1 * src[0] + cw9 * src[1] + cw17 * src[2] + cw25 * src[3] + cw33 * src[4] + cw41 * src[5] + cw49 * src[6] + cw57 * src[7];
      float a2 = zf[2] + cw2 * src[0] + cw10 * src[1] + cw18 * src[2] + cw26 * src[3] + cw34 * src[4] + cw42 * src[5] + cw50 * src[6] + cw58 * src[7];
      float a3 = zf[3] + cw3 * src[0] + cw11 * src[1] + cw19 * src[2] + cw27 * src[3] + cw35 * src[4] + cw43 * src[5] + cw51 * src[6] + cw59 * src[7];
      float a4 = zf[4] + cw4 * src[0] + cw12 * src[1] + cw20 * src[2] + cw28 * src[3] + cw36 * src[4] + cw44 * src[5] + cw52 * src[6] + cw60 * src[7];
      float a5 = zf[5] + cw5 * src[0] + cw13 * src[1] + cw21 * src[2] + cw29 * src[3] + cw37 * src[4] + cw45 * src[5] + cw53 * src[6] + cw61 * src[7];
      float a6 = zf[6] + cw6 * src[0] + cw14 * src[1] + cw22 * src[2] + cw30 * src[3] + cw38 * src[4] + cw46 * src[5] + cw54 * src[6] + cw62 * src[7];
      float a7 = zf[7] + cw7 * src[0] + cw15 * src[1] + cw23 * src[2] + cw31 * src[3] + cw39 * src[4] + cw47 * src[5] + cw55 * src[6] + cw63 * src[7];
      // Mixin.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      a3 += mw3 * cf;
      a4 += mw4 * cf;
      a5 += mw5 * cf;
      a6 += mw6 * cf;
      a7 += mw7 * cf;
      // FiLM: gamma * (conv+mixin) + beta.
      a0 = g0 * a0 + bt0;
      a1 = g1 * a1 + bt1;
      a2 = g2 * a2 + bt2;
      a3 = g3 * a3 + bt3;
      a4 = g4 * a4 + bt4;
      a5 = g5 * a5 + bt5;
      a6 = g6 * a6 + bt6;
      a7 = g7 * a7 + bt7;
      // LeakyReLU.
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      a3 = (a3 >= 0.0f) ? a3 : a3 * kLeakySlope;
      a4 = (a4 >= 0.0f) ? a4 : a4 * kLeakySlope;
      a5 = (a5 >= 0.0f) ? a5 : a5 * kLeakySlope;
      a6 = (a6 >= 0.0f) ? a6 : a6 * kLeakySlope;
      a7 = (a7 >= 0.0f) ? a7 : a7 * kLeakySlope;
      // Skip connections: accumulate this layer's post-activation into the head sum
      // (same site the static A2FastModel uses -- after FiLM+LeakyReLU, BEFORE the
      // l1x1 residual).
      float* __restrict__ hsum = &_head_sum[static_cast<size_t>(f) * 8];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      hsum[3] += a3;
      hsum[4] += a4;
      hsum[5] += a5;
      hsum[6] += a6;
      hsum[7] += a7;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 8];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2 + lw30 * a3 + lw40 * a4 + lw50 * a5 + lw60 * a6 + lw70 * a7;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2 + lw31 * a3 + lw41 * a4 + lw51 * a5 + lw61 * a6 + lw71 * a7;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2 + lw32 * a3 + lw42 * a4 + lw52 * a5 + lw62 * a6 + lw72 * a7;
      lin[3] += lb3 + lw03 * a0 + lw13 * a1 + lw23 * a2 + lw33 * a3 + lw43 * a4 + lw53 * a5 + lw63 * a6 + lw73 * a7;
      lin[4] += lb4 + lw04 * a0 + lw14 * a1 + lw24 * a2 + lw34 * a3 + lw44 * a4 + lw54 * a5 + lw64 * a6 + lw74 * a7;
      lin[5] += lb5 + lw05 * a0 + lw15 * a1 + lw25 * a2 + lw35 * a3 + lw45 * a4 + lw55 * a5 + lw65 * a6 + lw75 * a7;
      lin[6] += lb6 + lw06 * a0 + lw16 * a1 + lw26 * a2 + lw36 * a3 + lw46 * a4 + lw56 * a5 + lw66 * a6 + lw76 * a7;
      lin[7] += lb7 + lw07 * a0 + lw17 * a1 + lw27 * a2 + lw37 * a3 + lw47 * a4 + lw57 * a5 + lw67 * a6 + lw77 * a7;
    }
  }
  // [Fork] END: hand-unrolled Channels==8 kernel

  else
  {
    using MatCC  = Eigen::Matrix<float, Channels, Channels>;
    using MatCDyn= Eigen::Matrix<float, Channels, Eigen::Dynamic>;
    using VecC   = Eigen::Matrix<float, Channels, 1>;
    using RowDyn = Eigen::Matrix<float, 1, Eigen::Dynamic>;

    Eigen::Map<const VecC>   conv_b_vec(L.conv_b.data());
    Eigen::Map<const VecC>   mixin_vec(L.mixin_w.data());
    Eigen::Map<const MatCC>  l1x1_mat(L.l1x1_w.data());
    Eigen::Map<const VecC>   l1x1_b_vec(L.l1x1_b.data());
    Eigen::Map<const RowDyn> cond_row(cond, 1, num_frames);
    Eigen::Map<const VecC>   gamma_vec(_film_gamma[li].data());
    Eigen::Map<const VecC>   beta_vec(_film_beta[li].data());

    // Aligned16: _z/_layer_in use Eigen::aligned_allocator (guaranteed 16-byte aligned).
    Eigen::Map<MatCDyn, Eigen::Aligned16> ztile(_z.data(), Channels, num_frames);
    Eigen::Map<MatCDyn, Eigen::Aligned16> lin_block(_layer_in.data(), Channels, num_frames);

    ztile.setZero();

    // Conv: one Channels×Channels × Channels×N GEMM per tap.
    for (int k = 0; k < K; k++)
    {
      const int tap_base = tap_base_phys(K - 1 - k);
      Eigen::Map<const MatCC>  W(&L.conv_w[static_cast<size_t>(k) * Channels * Channels]);
      Eigen::Map<const MatCDyn> input_block(&L.history[static_cast<size_t>(tap_base) * Channels], Channels, num_frames);
      ztile.noalias() += W * input_block;
    }

    // Post-conv: bias, mixin, FiLM, LeakyReLU, 1x1 residual.
    ztile.colwise() += conv_b_vec;
    ztile.noalias() += mixin_vec * cond_row;
    // FiLM: scale each row c by gamma[c], then add beta[c].
    ztile = gamma_vec.asDiagonal() * ztile; // O(C×N) — diagonal×dense
    ztile.colwise() += beta_vec;
    // LeakyReLU.
    ztile = (ztile.array() < 0.0f).select(ztile.array() * kLeakySlope, ztile.array());
    // Skip connections: accumulate post-activation into the head sum.
    Eigen::Map<MatCDyn, Eigen::Aligned16> hsum_block(_head_sum.data(), Channels, num_frames);
    hsum_block.noalias() += ztile;
    // layer1x1 residual.
    lin_block.noalias() += l1x1_mat * ztile;
    lin_block.colwise() += l1x1_b_vec;
  }
}

template <int Channels>
void ParametricA2FastModel<Channels>::_layer_forward(int li, const float* cond, int num_frames)
{
  Layer& L = _layers[li];
  _ring_write(L, num_frames);
  switch (L.kernel_size)
  {
    case 6:  _layer_forward_k<6>(L, li, cond, num_frames); break;
    case 15: _layer_forward_k<15>(L, li, cond, num_frames); break;
    default: throw std::runtime_error("ParametricA2FastModel: unexpected kernel_size "
                                      + std::to_string(L.kernel_size));
  }
}

// --- Head (identical to A2FastModel) ---
template <int Channels>
void ParametricA2FastModel<Channels>::_head_forward(float* output, int num_frames)
{
  _head_ring_write(num_frames);
  #if NAM_A2_RING_MODE == 1
  const int mask = _head_pow2_mask;
  auto col_of = [&](int f, int k) { return (_head_write_pos - num_frames + f - (kHeadKernelSize - 1 - k)) & mask; };
  #else
  const int base = _head_write_pos - num_frames;
  auto col_of = [&](int f, int k) { return base + f - (kHeadKernelSize - 1 - k); };
  #endif

  for (int f = 0; f < num_frames; f++)
  {
    float y = _head_b;
    for (int k = 0; k < kHeadKernelSize; k++)
    {
      const int col = col_of(f, k);
      const float* src = &_head_history[static_cast<size_t>(col) * Channels];
      const float* wk = _head_w[k].data();
      for (int b = 0; b < Channels; b++)
        y += wk[b] * src[b];
    }
    output[f] = y * _head_scale;
  }
}

// --- process ---
template <int Channels>
void ParametricA2FastModel<Channels>::process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames)
{
  if (num_frames > GetMaxBufferSize())
    SetMaxBufferSize(num_frames);

  // Rebuild per-layer gamma/beta from knob atomics (cheap: 23 × num_params × 2C FMAs).
  _recompute_film();

  const NAM_SAMPLE* in0 = input[0];
  NAM_SAMPLE* out0 = output[0];

  // Rechannel + float cond copy.
  float* cond = _cond.data();
  for (int f = 0; f < num_frames; f++)
  {
    const float x = static_cast<float>(in0[f]);
    cond[f] = x;
    float* lin = &_layer_in[static_cast<size_t>(f) * Channels];
    for (int c = 0; c < Channels; c++)
      lin[c] = _rechannel_w[c] * x;
  }

  // Zero the skip accumulator for this buffer.
  std::memset(_head_sum.data(), 0, static_cast<size_t>(num_frames) * Channels * sizeof(float));

  // Run all 23 layers. Afterwards _layer_in holds the final layer's l1x1+residual output
  // (feeds the next layer, not the head), and _head_sum holds the sum of every layer's
  // post-activation -- parametric models are SKIP-ONLY, so this is the only path.
  for (int li = 0; li < kNumLayers; li++)
    _layer_forward(li, cond, num_frames);

  // Head: reads _head_sum (skip accumulator) via _head_ring_write.
  float* head_out = _head_out.data();
  _head_forward(head_out, num_frames);

  // DC blocker: first-order IIR HP, fc ≈ 20 Hz. Matches ParametricWaveNet::process().
  float xp = _dc_x_prev, yp = _dc_y_prev;
  const float alpha = _dc_alpha;
  for (int f = 0; f < num_frames; f++)
  {
    const float xi = head_out[f];
    const float yi = xi - xp + alpha * yp;
    xp = xi;
    yp = yi;
    out0[f] = static_cast<NAM_SAMPLE>(yi);
  }
  _dc_x_prev = xp;
  _dc_y_prev = yp;
}

// [Fork] END: ParametricA2FastModel<Channels>

// --- ParametricA2FastConfig ---
struct ParametricA2FastConfig : public ModelConfig
{
  int channels = 0;
  int num_params = 0;
  std::vector<ParametricParam> param_defs;

  template <int C>
  std::unique_ptr<DSP> _make(std::vector<float>&& weights, double sampleRate)
  {
    return std::make_unique<ParametricA2FastModel<C>>(std::move(weights), sampleRate,
                                                      num_params, param_defs);
  }

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override
  {
    if (channels == 3)
      return _make<3>(std::move(weights), sampleRate);
    if (channels == 4) // [Fork] BEGIN: intermediate widths
      return _make<4>(std::move(weights), sampleRate);
    if (channels == 5)
      return _make<5>(std::move(weights), sampleRate);
    if (channels == 6)
      return _make<6>(std::move(weights), sampleRate);
    if (channels == 7) // [Fork] END
      return _make<7>(std::move(weights), sampleRate);
    if (channels == 8)
      return _make<8>(std::move(weights), sampleRate);
    throw std::runtime_error("ParametricA2FastConfig: unsupported channel count "
                             + std::to_string(channels));
  }
};

} // namespace

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------
bool is_a2_shape(const nlohmann::json& config, int* channels)
{
  // Exactly one layer array
  auto layers_it = config.find("layers");
  if (layers_it == config.end() || !layers_it->is_array() || layers_it->size() != 1)
    return false;

  // No post-stack head
  auto head_it = config.find("head");
  if (head_it != config.end() && !head_it->is_null())
    return false;

  // No conditioning DSP. When given a non-null condition_dsp the generic WaveNet
  // builds a nested model and routes the conditioning signal through it before the
  // layer stack; the fast path has no such stage and feeds the raw input as the
  // condition. The condition DSP carries its own weights, so the parent weight
  // stream is identical with or without it and the loader cannot detect the
  // difference -- the detector must reject it here, or the fast path would silently
  // produce different audio than the model it replaces.
  auto cond_it = config.find("condition_dsp");
  if (cond_it != config.end() && !cond_it->is_null())
    return false;

  // head_scale is loaded from the trailing weight, but require the field to
  // stay schema-compatible with the generic WaveNet parser.
  auto hs_it = config.find("head_scale");
  if (hs_it == config.end() || !hs_it->is_number())
    return false;

  // in_channels defaults to 1, must be 1
  if (config.value("in_channels", 1) != 1)
    return false;

  const auto& la = (*layers_it)[0];

  if (la.value("input_size", 0) != 1)
    return false;
  if (la.value("condition_size", 0) != 1)
    return false;

  const int ch = la.value("channels", 0);
  const int bn = la.value("bottleneck", 0);
  if (ch != bn)
    return false;
  // [Fork] widened from "3 or 8" to also allow 4, 5, 6, 7
  if (ch != 3 && ch != 4 && ch != 5 && ch != 6 && ch != 7 && ch != 8)
    return false;

  // kernel_sizes must match kKernelSizes exactly
  auto ks_it = la.find("kernel_sizes");
  if (ks_it == la.end() || !ks_it->is_array() || ks_it->size() != kNumLayers)
    return false;
  for (int i = 0; i < kNumLayers; i++)
  {
    if (!(*ks_it)[i].is_number_integer() || (*ks_it)[i].get<int>() != kKernelSizes[i])
      return false;
  }

  // dilations must match kDilations exactly
  auto dl_it = la.find("dilations");
  if (dl_it == la.end() || !dl_it->is_array() || dl_it->size() != kNumLayers)
    return false;
  for (int i = 0; i < kNumLayers; i++)
  {
    if (!(*dl_it)[i].is_number_integer() || (*dl_it)[i].get<int>() != kDilations[i])
      return false;
  }

  // activation: all LeakyReLU(0.01)
  auto act_it = la.find("activation");
  if (act_it == la.end() || !act_it->is_array() || act_it->size() != kNumLayers)
    return false;
  for (const auto& a : *act_it)
  {
    if (!a.is_object() || a.value("type", std::string()) != "LeakyReLU")
      return false;
    if (!close_to(a.value("negative_slope", 0.0f), kLeakySlope))
      return false;
  }

  // gating_mode: all "none" (or field absent)
  auto gm_it = la.find("gating_mode");
  if (gm_it != la.end() && !gm_it->is_null())
  {
    if (!all_none_strings(*gm_it) || gm_it->size() != kNumLayers)
      return false;
  }

  // Legacy boolean `gated` (the pre-gating_mode schema): the generic parser maps
  // gated==true to GATED layers, which the fast path does not implement. A genuinely
  // gated model has a larger weight stream and the loader would throw, but reject it
  // here so the boundary is enforced by the detector rather than a downstream error.
  auto gated_it = la.find("gated");
  if (gated_it != la.end() && gated_it->is_boolean() && gated_it->get<bool>())
    return false;

  // secondary_activation: all null (or field absent)
  auto sa_it = la.find("secondary_activation");
  if (sa_it != la.end() && !sa_it->is_null())
  {
    if (!all_null(*sa_it) || sa_it->size() != kNumLayers)
      return false;
  }

  // head1x1 inactive
  auto h1x1_it = la.find("head1x1");
  if (h1x1_it != la.end() && h1x1_it->is_object() && h1x1_it->value("active", false))
    return false;

  // layer1x1 active with groups=1
  auto l1x1_it = la.find("layer1x1");
  if (l1x1_it == la.end() || !l1x1_it->is_object())
    return false;
  if (!l1x1_it->value("active", false))
    return false;
  if (l1x1_it->value("groups", 1) != 1)
    return false;

  // Layer-array head rechannel: k=16, out_channels=1, bias=true
  auto lah_it = la.find("head");
  if (lah_it == la.end() || !lah_it->is_object())
    return false;
  if (lah_it->value("out_channels", 0) != 1)
    return false;
  if (lah_it->value("kernel_size", 0) != kHeadKernelSize)
    return false;
  if (lah_it->value("head_dilation", 1) != 1)
    return false;
  if (!lah_it->value("bias", false))
    return false;

  // No FiLM anywhere
  for (const char* key : {"conv_pre_film", "conv_post_film", "input_mixin_pre_film", "input_mixin_post_film",
                          "activation_pre_film", "activation_post_film", "layer1x1_post_film", "head1x1_post_film"})
  {
    if (!film_inactive(la, key))
      return false;
  }

  // No grouped convolutions
  if (la.value("groups_input", 1) != 1)
    return false;
  if (la.value("groups_input_mixin", 1) != 1)
    return false;

  // Not slimmable
  auto slim_it = la.find("slimmable");
  if (slim_it != la.end() && !slim_it->is_null())
    return false;

  if (channels)
    *channels = ch;
  return true;
}

std::unique_ptr<ModelConfig> create_a2_fast_config(const nlohmann::json& config, double sampleRate)
{
  (void)sampleRate;
  int ch = 0;
  if (!is_a2_shape(config, &ch))
    throw std::runtime_error("create_a2_fast_config: config does not match A2 shape");
  auto out = std::make_unique<A2FastConfig>();
  out->channels = ch;
  return out;
}

// [Fork] Public API for parametric fast path — called only from parametric_wavenet.cpp.
// ParametricWaveNet detector: config["layers"] is an integer (channel count), not an array.
// All ParametricWaveNet models currently use the A2 topology, so the only check beyond
// "channels in the supported fast-path set" is that a valid parametric block is present.
bool is_parametric_a2_shape(const nlohmann::json& config, int* channels, int* num_params)
{
  auto layers_it = config.find("layers");
  if (layers_it == config.end() || !layers_it->is_number_integer())
    return false;
  const int ch = layers_it->get<int>();
  // [Fork] widened from "3 or 8" to also allow 4, 5, 6, 7
  if (ch != 3 && ch != 4 && ch != 5 && ch != 6 && ch != 7 && ch != 8)
    return false;

  auto param_it = config.find("parametric");
  if (param_it == config.end() || !param_it->is_object())
    return false;
  const int np = param_it->value("condition_size", 0);
  if (np <= 0 || np > ParametricWaveNet::kMaxParams)
    return false;

  if (channels)
    *channels = ch;
  if (num_params)
    *num_params = np;
  return true;
}

std::unique_ptr<ModelConfig> create_parametric_a2_fast_config(const nlohmann::json& config,
                                                               double sampleRate)
{
  (void)sampleRate;
  int ch = 0, np = 0;
  if (!is_parametric_a2_shape(config, &ch, &np))
    throw std::runtime_error("create_parametric_a2_fast_config: config does not match parametric A2 shape");

  // Validate required fields (match the slow path's error contract).
  if (!config.contains("head_scale"))
    throw std::runtime_error("ParametricWaveNet: missing 'head_scale'");
  if (!config.contains("parametric"))
    throw std::runtime_error("ParametricWaveNet: missing 'parametric' block");

  const auto& param_block = config["parametric"];
  std::vector<ParametricParam> param_defs;
  if (param_block.contains("parameters"))
  {
    for (const auto& p : param_block["parameters"])
    {
      ParametricParam pd;
      pd.name        = p.value("name", "");
      pd.min_val     = p.value("min", p.value("minimum", 0.0f));
      pd.max_val     = p.value("max", p.value("maximum", 1.0f));
      pd.default_val = p.value("default", 0.5f);
      pd.steps       = p.value("steps", 0);
      param_defs.push_back(std::move(pd));
    }
  }

  // [Fork] Reject legacy/residual/untagged models and unknown schemas (shared guard,
  // also used by the generic path). Skip-only: there is no head to select.
  require_supported_parametric_model(config);

  auto out = std::make_unique<ParametricA2FastConfig>();
  out->channels   = ch;
  out->num_params = np;
  out->param_defs = std::move(param_defs);
  return out;
}

} // namespace a2_fast
} // namespace wavenet
} // namespace nam

#endif // NAM_ENABLE_A2_FAST
