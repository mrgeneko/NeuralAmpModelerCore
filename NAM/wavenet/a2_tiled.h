#pragma once
// [Fork] Frame-major, register-tiled forward pass for the A2 WaveNet shape with per-layer FiLM
// (the ParametricA2FastModel network). Header-only so no build-system file lists change.
//
// Why: the per-width hand-unrolled kernels in a2_fast.cpp walk one frame at a time (or, for
// Channels 8/9, go through Eigen). This engine lays every activation out as one time-contiguous
// row per channel, so the SIMD lanes run across 8 frames and any channel count fills them. One
// register tile is 4 output channels x up to 3 vectors of 8 frames (24 accumulators + inputs fit
// NEON's 32 registers). Measured on an M5 against the previous path at 64-sample blocks:
// width 4 1.5x, 5 2.4x, 6 2.9x, 8 4.5x, 9 3.5x, with output within float rounding (ESR below
// -94 dB). Width 3 is slower here (too little work per load), so it keeps the unrolled kernel.
//
// Layout and scheduling:
//   - Histories are power-of-two rings with a mirror of the first kChunk columns, so every tap's
//     window is contiguous. A block is split into chunks of at most kChunk frames (rows stay in
//     L1), each padded to a multiple of 8 frames. Padded lanes compute on zero input and are
//     never output; real frames only read earlier columns, so they never see the padding, and
//     the output does not depend on how the host splits blocks.
//   - The residual stream is written straight into the next layer's ring and the skip sum
//     straight into the head ring, so there are no copy passes. The last layer's residual is not
//     computed: parametric models are skip-only, so nothing reads it.
//   - FiLM (gamma * (conv + mixin + bias) + beta, then LeakyReLU) is applied inside the tile
//     epilogue, before the activation is stored.
//
//   - Silence skip: once the input has been exactly zero (and FiLM unchanged) for longer than the
//     receptive field, the network has no feedback, so every history holds one constant per row
//     and the output is one constant. Later all-zero chunks are answered with that constant
//     without running the network; the histories are first filled with their constants so a
//     restart reads exactly what a full run would have left. Bit-identical to not skipping.
//     Define NAM_A2_TILED_NO_SILENCE_SKIP to turn it off.
//
// Clean-room: written from the A2 layer shape and this fork's weight order. It borrows nothing
// from other NAM engines.

#if defined(NAM_ENABLE_A2_FAST) && !defined(NAM_A2_TILED_DISABLE) && defined(__aarch64__) && defined(__ARM_NEON)
  #define NAM_A2_TILED 1

  #include <arm_neon.h>

  #include <algorithm>
  #include <type_traits>
  #include <array>
  #include <cstring>
  #include <vector>

  #include "a2_fast.h"

namespace nam
{
namespace wavenet
{
namespace a2_fast
{
namespace tiled
{
/// Smallest width this engine is used for; narrower models keep the unrolled kernels.
constexpr int kMinChannels = 4;

template <int C>
class Engine
{
public:
  static constexpr int NL = kNumLayers;
  static constexpr int HK = kHeadKernelSize;
  static constexpr int NOT = (C + 3) / 4; // 4-output tiles
  static constexpr int kChunk = 128; // frames per chunk; a multiple of 8

  /// conv: [k * C * C + j * C + i] (tap k, input j, output i); l1x1: [j * C + i].
  void setLayer(int li, const float* conv, const float* convBias, const float* mixin, const float* l1x1,
                const float* l1x1Bias)
  {
    const int K = kKernelSizes[li];
    convW_[li].assign(static_cast<size_t>(NOT) * K * C * 4, 0.0f);
    w1_[li].assign(static_cast<size_t>(NOT) * C * 4, 0.0f);
    for (int o = 0; o < C; ++o)
    {
      for (int j = 0; j < C; ++j)
      {
        for (int k = 0; k < K; ++k)
          convW_[li][((static_cast<size_t>(o / 4) * K + k) * C + j) * 4 + (o & 3)] =
            conv[static_cast<size_t>(k) * C * C + static_cast<size_t>(j) * C + o];
        w1_[li][(static_cast<size_t>(o / 4) * C + j) * 4 + (o & 3)] = l1x1[static_cast<size_t>(j) * C + o];
      }
      convB_[li][o] = convBias[o];
      mix_[li][o] = mixin[o];
      b1_[li][o] = l1x1Bias[o];
    }
  }

  void setRechannel(const float* w)
  {
    for (int c = 0; c < C; ++c)
      rech_[c] = w[c];
  }

  void setHead(const std::array<std::array<float, C>, HK>& w, float bias, float scale)
  {
    for (int k = 0; k < HK; ++k)
      for (int i = 0; i < C; ++i)
        hw_[k * C + i] = w[k][i];
    hb_ = bias;
    hscale_ = scale;
  }

  /// Allocates and zeroes all state.
  void reset()
  {
    for (int li = 0; li < NL; ++li)
      ring_[li].alloc((kKernelSizes[li] - 1) * kDilations[li]);
    head_.alloc(HK - 1);
    act_.assign(static_cast<size_t>(C) * kChunk, 0.0f);
    cond_.assign(kChunk, 0.0f);
    out_.assign(kChunk, 0.0f);
    pos_ = 0;
    havePrev_ = haveSteady_ = filled_ = false;
    stableRun_ = 0;
  }

  /// Runs n frames (any n >= 0) and writes n float outputs, before any DC blocker. With FiLM:
  /// gamma * (conv + mixin + bias) + beta ahead of the activation.
  void process(const NAM_SAMPLE* in, float* out, int n, const std::array<std::array<float, C>, NL>& gamma,
               const std::array<std::array<float, C>, NL>& beta)
  {
    for (int li = 0; li < NL; ++li)
      for (int o = 0; o < C; ++o)
      {
        gam_[li][o] = gamma[li][o];
        bet_[li][o] = beta[li][o];
      }
    run(in, out, n);
  }

  /// The same without FiLM (plain A2 models): gamma 1, beta 0, which leaves every value unchanged.
  void process(const NAM_SAMPLE* in, float* out, int n)
  {
    for (int li = 0; li < NL; ++li)
      for (int o = 0; o < C; ++o)
      {
        gam_[li][o] = 1.0f;
        bet_[li][o] = 0.0f;
      }
    run(in, out, n);
  }

private:
  // Frames of exact-zero input after which the whole network state is constant.
  static constexpr unsigned receptiveField()
  {
    unsigned r = HK;
    for (int li = 0; li < NL; ++li)
      r += static_cast<unsigned>((kKernelSizes[li] - 1) * kDilations[li]);
    return r;
  }

  void run(const NAM_SAMPLE* in, float* out, int n)
  {
#if defined(NAM_A2_TILED_NO_SILENCE_SKIP)
    for (int done = 0; done < n; done += kChunk)
      chunk(in + done, out + done, std::min(kChunk, n - done));
#else
    const bool filmSame = havePrev_ && std::memcmp(gam_, prevGam_, sizeof(gam_)) == 0
                          && std::memcmp(bet_, prevBet_, sizeof(bet_)) == 0;
    std::memcpy(prevGam_, gam_, sizeof(gam_));
    std::memcpy(prevBet_, bet_, sizeof(bet_));
    havePrev_ = true;
    if (!filmSame)
    {
      stableRun_ = 0;
      haveSteady_ = false;
    }
    for (int done = 0; done < n; done += kChunk)
    {
      const int f = std::min(kChunk, n - done);
      bool zero = true;
      for (int i = 0; i < f && zero; ++i)
        zero = in[done + i] == NAM_SAMPLE(0);
      if (zero && haveSteady_)
      {
        if (!filled_)
        {
          fillSteady();
          filled_ = true;
        }
        std::fill(out + done, out + done + f, steadyOut_);
        pos_ += static_cast<unsigned>(f);
        continue;
      }
      chunk(in + done, out + done, f);
      filled_ = false;
      if (zero)
      {
        stableRun_ = std::min(stableRun_ + static_cast<unsigned>(f), 2u * receptiveField());
        if (stableRun_ >= receptiveField())
        {
          steadyOut_ = out[done + f - 1];
          haveSteady_ = true;
        }
      }
      else
      {
        stableRun_ = 0;
        haveSteady_ = false;
      }
    }
#endif
  }

  // Sets every history to its row constant (the value in the last written column).
  void fillSteady()
  {
    auto fillRing = [&](Ring& r) {
      const size_t col = static_cast<size_t>((pos_ - 1u) & static_cast<unsigned>(r.mask));
      for (int c = 0; c < C; ++c)
      {
        float* row = r.row(c);
        std::fill(row, row + r.stride, row[col]);
      }
    };
    for (int li = 0; li < NL; ++li)
      fillRing(ring_[li]);
    fillRing(head_);
  }

  struct Ring
  {
    std::vector<float> buf;
    int size = 0, mask = 0, stride = 0;

    void alloc(int lookback)
    {
      int p = 1;
      while (p < lookback + kChunk)
        p <<= 1;
      size = p;
      mask = p - 1;
      stride = p + kChunk;
      buf.assign(static_cast<size_t>(C) * static_cast<size_t>(stride), 0.0f);
    }
    float* row(int c) { return buf.data() + static_cast<size_t>(c) * static_cast<size_t>(stride); }

    // After writing n columns at pos: fold a wrapped tail back to the start of the ring, then
    // refresh the mirror of the first kChunk columns.
    void sync(unsigned pos, int n)
    {
      const int wp = static_cast<int>(pos & static_cast<unsigned>(mask));
      const int end = wp + n;
      if (end > size)
        for (int c = 0; c < C; ++c)
          std::memcpy(row(c), row(c) + size, static_cast<size_t>(end - size) * sizeof(float));
      const int hi = std::min(std::min(end, size), kChunk);
      if (wp < hi)
        for (int c = 0; c < C; ++c)
          std::memcpy(row(c) + size + wp, row(c) + wp, static_cast<size_t>(hi - wp) * sizeof(float));
    }
  };

  using f4 = float32x4_t;

  // One accumulator tile: 4 outputs x NV vectors of 8 frames, reduced over nTap source windows.
  template <int NV, class Init, class Fin>
  static inline void tile(const float* W, int nTap, const float* const* src, int sstride, int v0, Init&& init,
                          Fin&& fin)
  {
    f4 a[4][NV][2];
    init(a);
    for (int t = 0; t < nTap; ++t)
    {
      const float* s = src[t] + 8 * v0;
      const float* wt = W + static_cast<size_t>(t) * C * 4;
      for (int i = 0; i < C; ++i)
      {
        const float* p = s + static_cast<size_t>(i) * static_cast<size_t>(sstride);
        f4 x[NV][2];
        for (int v = 0; v < NV; ++v)
        {
          x[v][0] = vld1q_f32(p + 8 * v);
          x[v][1] = vld1q_f32(p + 8 * v + 4);
        }
        const f4 w = vld1q_f32(wt + 4 * i);
        for (int v = 0; v < NV; ++v)
          for (int h = 0; h < 2; ++h)
          {
            a[0][v][h] = vfmaq_laneq_f32(a[0][v][h], x[v][h], w, 0);
            a[1][v][h] = vfmaq_laneq_f32(a[1][v][h], x[v][h], w, 1);
            a[2][v][h] = vfmaq_laneq_f32(a[2][v][h], x[v][h], w, 2);
            a[3][v][h] = vfmaq_laneq_f32(a[3][v][h], x[v][h], w, 3);
          }
      }
    }
    fin(a);
  }

  // Dilated conv + mixin + bias, FiLM, LeakyReLU; stores the activation and adds it to the skip sum.
  template <int NV>
  void convTile(int li, int ot, int v0)
  {
    Ring& R = ring_[li];
    const int K = kKernelSizes[li], D = kDilations[li];
    const float* src[16];
    for (int k = 0; k < K; ++k)
      src[k] = R.buf.data()
               + static_cast<int>((pos_ - static_cast<unsigned>((K - 1 - k) * D)) & static_cast<unsigned>(R.mask));
    const unsigned hwp = pos_ & static_cast<unsigned>(head_.mask);
    tile<NV>(
      convW_[li].data() + static_cast<size_t>(ot) * K * C * 4, K, src, R.stride, v0,
      [&](f4 (&a)[4][NV][2]) {
        const f4 bq = vld1q_f32(convB_[li] + ot * 4), mq = vld1q_f32(mix_[li] + ot * 4);
        for (int v = 0; v < NV; ++v)
        {
          const f4 c0 = vld1q_f32(cond_.data() + 8 * (v0 + v)), c1 = vld1q_f32(cond_.data() + 8 * (v0 + v) + 4);
          a[0][v][0] = vfmaq_laneq_f32(vdupq_laneq_f32(bq, 0), c0, mq, 0);
          a[0][v][1] = vfmaq_laneq_f32(vdupq_laneq_f32(bq, 0), c1, mq, 0);
          a[1][v][0] = vfmaq_laneq_f32(vdupq_laneq_f32(bq, 1), c0, mq, 1);
          a[1][v][1] = vfmaq_laneq_f32(vdupq_laneq_f32(bq, 1), c1, mq, 1);
          a[2][v][0] = vfmaq_laneq_f32(vdupq_laneq_f32(bq, 2), c0, mq, 2);
          a[2][v][1] = vfmaq_laneq_f32(vdupq_laneq_f32(bq, 2), c1, mq, 2);
          a[3][v][0] = vfmaq_laneq_f32(vdupq_laneq_f32(bq, 3), c0, mq, 3);
          a[3][v][1] = vfmaq_laneq_f32(vdupq_laneq_f32(bq, 3), c1, mq, 3);
        }
      },
      [&](f4 (&a)[4][NV][2]) {
        for (int l = 0; l < 4; ++l)
        {
          const int o = ot * 4 + l;
          if (o >= C)
            break;
          float* ap = act_.data() + static_cast<size_t>(o) * F_ + 8 * v0;
          float* hp = head_.row(o) + hwp;
          const float gl = gam_[li][o], bl = bet_[li][o];
          for (int v = 0; v < NV; ++v)
            for (int h = 0; h < 2; ++h)
            {
              const f4 zf = vfmaq_n_f32(vdupq_n_f32(bl), a[l][v][h], gl);
              const f4 z = vmaxq_f32(zf, vmulq_n_f32(zf, kLeakySlope));
              vst1q_f32(ap + 8 * v + 4 * h, z);
              float* q = hp + 8 * (v0 + v) + 4 * h;
              vst1q_f32(q, li == 0 ? z : vaddq_f32(vld1q_f32(q), z));
            }
        }
      });
  }

  // layer1x1 + residual, written into the next layer's ring.
  template <int NV>
  void resTile(int li, int ot, int v0)
  {
    Ring& R = ring_[li];
    Ring& N = ring_[li + 1];
    const unsigned wpI = pos_ & static_cast<unsigned>(R.mask), wpO = pos_ & static_cast<unsigned>(N.mask);
    const float* src[1] = {act_.data()};
    tile<NV>(
      w1_[li].data() + static_cast<size_t>(ot) * C * 4, 1, src, F_, v0,
      [&](f4 (&a)[4][NV][2]) {
        for (int l = 0; l < 4; ++l)
        {
          const int o = ot * 4 + l;
          for (int v = 0; v < NV; ++v)
            for (int h = 0; h < 2; ++h)
              a[l][v][h] = o < C ? vaddq_f32(vld1q_f32(R.row(o) + wpI + 8 * (v0 + v) + 4 * h), vdupq_n_f32(b1_[li][o]))
                                 : vdupq_n_f32(0.f);
        }
      },
      [&](f4 (&a)[4][NV][2]) {
        for (int l = 0; l < 4; ++l)
        {
          const int o = ot * 4 + l;
          if (o >= C)
            break;
          for (int v = 0; v < NV; ++v)
            for (int h = 0; h < 2; ++h)
              vst1q_f32(N.row(o) + wpO + 8 * (v0 + v) + 4 * h, a[l][v][h]);
        }
      });
  }

  template <int NV>
  void headTile(int v0)
  {
    const unsigned mk = static_cast<unsigned>(head_.mask);
    f4 p[4][NV][2];
    for (int q = 0; q < 4; ++q)
      for (int v = 0; v < NV; ++v)
        p[q][v][0] = p[q][v][1] = vdupq_n_f32(0.f);
    for (int k0 = 0; k0 < HK; k0 += 4)
      for (int q = 0; q < 4; ++q)
      {
        const int k = k0 + q;
        const float* s =
          head_.buf.data() + static_cast<int>((pos_ - static_cast<unsigned>(HK - 1 - k)) & mk) + 8 * v0;
        for (int i = 0; i < C; ++i)
        {
          const float* r = s + static_cast<size_t>(i) * static_cast<size_t>(head_.stride);
          const float w = hw_[k * C + i];
          for (int v = 0; v < NV; ++v)
          {
            p[q][v][0] = vfmaq_n_f32(p[q][v][0], vld1q_f32(r + 8 * v), w);
            p[q][v][1] = vfmaq_n_f32(p[q][v][1], vld1q_f32(r + 8 * v + 4), w);
          }
        }
      }
    for (int v = 0; v < NV; ++v)
      for (int h = 0; h < 2; ++h)
      {
        const f4 s = vaddq_f32(vaddq_f32(p[0][v][h], p[1][v][h]), vaddq_f32(p[2][v][h], p[3][v][h]));
        vst1q_f32(out_.data() + 8 * (v0 + v) + 4 * h, vmulq_n_f32(vaddq_f32(s, vdupq_n_f32(hb_)), hscale_));
      }
  }

  // Calls fn(integral_constant<NV>, v0) over the F_/8 frame vectors in tiles of up to 3.
  template <class Fn>
  void vecs(Fn&& fn)
  {
    const int nVec = F_ / 8;
    int v0 = 0;
    while (v0 < nVec)
    {
      const int r = nVec - v0;
      if (r >= 3)
      {
        fn(std::integral_constant<int, 3>{}, v0);
        v0 += 3;
      }
      else if (r == 2)
      {
        fn(std::integral_constant<int, 2>{}, v0);
        v0 += 2;
      }
      else
      {
        fn(std::integral_constant<int, 1>{}, v0);
        v0 += 1;
      }
    }
  }

  void chunk(const NAM_SAMPLE* in, float* out, int f)
  {
    F_ = (f + 7) & ~7;
    for (int i = 0; i < f; ++i)
      cond_[i] = static_cast<float>(in[i]);
    for (int i = f; i < F_; ++i)
      cond_[i] = 0.0f;
    {
      const unsigned wp = pos_ & static_cast<unsigned>(ring_[0].mask);
      for (int c = 0; c < C; ++c)
      {
        float* r = ring_[0].row(c) + wp;
        for (int i = 0; i < F_; ++i)
          r[i] = rech_[c] * cond_[i];
      }
      ring_[0].sync(pos_, F_);
    }
    for (int li = 0; li < NL; ++li)
    {
      for (int ot = 0; ot < NOT; ++ot)
        vecs([&](auto nv, int v0) { convTile<decltype(nv)::value>(li, ot, v0); });
      if (li + 1 < NL)
      {
        for (int ot = 0; ot < NOT; ++ot)
          vecs([&](auto nv, int v0) { resTile<decltype(nv)::value>(li, ot, v0); });
        ring_[li + 1].sync(pos_, F_);
      }
    }
    head_.sync(pos_, F_);
    vecs([&](auto nv, int v0) { headTile<decltype(nv)::value>(v0); });
    std::memcpy(out, out_.data(), static_cast<size_t>(f) * sizeof(float));
    pos_ += static_cast<unsigned>(f);
  }

  Ring ring_[NL];
  Ring head_;
  std::vector<float> convW_[NL], w1_[NL];
  alignas(16) float convB_[NL][NOT * 4] = {};
  alignas(16) float mix_[NL][NOT * 4] = {};
  alignas(16) float b1_[NL][NOT * 4] = {};
  float gam_[NL][NOT * 4] = {};
  float bet_[NL][NOT * 4] = {};
  float rech_[C] = {};
  float hw_[HK * C] = {};
  float hb_ = 0.0f, hscale_ = 1.0f;
  std::vector<float> act_, cond_, out_;
  unsigned pos_ = 0;
  int F_ = 0;
  // Silence skip state.
  float prevGam_[NL][NOT * 4] = {};
  float prevBet_[NL][NOT * 4] = {};
  bool havePrev_ = false, haveSteady_ = false, filled_ = false;
  unsigned stableRun_ = 0;
  float steadyOut_ = 0.0f;
};
} // namespace tiled
} // namespace a2_fast
} // namespace wavenet
} // namespace nam
#endif
