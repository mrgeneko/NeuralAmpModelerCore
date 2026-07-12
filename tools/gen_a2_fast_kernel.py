#!/usr/bin/env python3
"""
[Fork] Generates hand-unrolled A2FastModel<N>/ParametricA2FastModel<N>
_layer_forward_k kernels for NAM/wavenet/a2_fast.cpp, for channel widths that
still fall back to the generic Eigen fixed-size-matrix path (see the `else`
branch after the highest `else if constexpr (Channels == ...)` in each
_layer_forward_k). That Eigen path measures far more CPU than its arithmetic
complexity implies at these small inner dimensions -- see
docs/Architecture/NAM_PERFORMANCE_INVESTIGATION.md in the outer Fork
repo for the full investigation.

Usage:
    python3 gen_a2_fast_kernel.py <N> plain   # A2FastModel<N> (has head_sum)
    python3 gen_a2_fast_kernel.py <N> param   # ParametricA2FastModel<N> (FiLM + head_sum; skip-only)

Before trusting output for a NEW width, validate the generator itself: run it
against the HIGHEST already-verified width (e.g. `python3
gen_a2_fast_kernel.py 7 plain`) and diff the output against the existing code
in a2_fast.cpp for that width -- it should match byte-for-byte. Only then use
it to generate the new width. This is how each new width (5/6/7, then 8) was
validated when originally added.
"""
import sys

def gen_nonparametric(N):
    NN = N * N
    lines = []
    lines.append(f"  else if constexpr (Channels == {N})")
    lines.append("  {")
    lines.append("    float* z = _z.data();")
    lines.append("")
    lines.append("    // Tap 0: seed z with conv_b (saves the memset-to-zero pass) and fold in")
    lines.append("    // the first tap's FMAs.")
    lines.append("    {")
    lines.append("      const float* wk = &L.conv_w[0];")
    lines.append("      const int tap_base = tap_base_phys(K - 1);")
    # weight decls, N per line
    for row_start in range(0, NN, N):
        decls = ", ".join(f"w{k} = wk[{k}]" for k in range(row_start, min(row_start + N, NN)))
        lines.append(f"      const float {decls};")
    cb_decls = ", ".join(f"cb{i} = L.conv_b[{i}]" for i in range(N))
    lines.append(f"      const float {cb_decls};")
    lines.append("      for (int f = 0; f < num_frames; f++)")
    lines.append("      {")
    lines.append(f"        const float* src = &L.history[static_cast<size_t>(tap_base + f) * {N}];")
    for i in range(N):
        lines.append(f"        float a{i} = cb{i} + w{i} * src[0];")
    for j in range(1, N):
        for i in range(N):
            lines.append(f"        a{i} += w{j*N+i} * src[{j}];")
    lines.append(f"        float* zf = z + static_cast<size_t>(f) * {N};")
    for i in range(N):
        lines.append(f"        zf[{i}] = a{i};")
    lines.append("      }")
    lines.append("    }")
    lines.append("")
    lines.append("    // Taps 1..K-2: accumulate into z with the same unrolled inner kernel.")
    lines.append("    for (int k = 1; k < K - 1; k++)")
    lines.append("    {")
    lines.append(f"      const float* wk = &L.conv_w[static_cast<size_t>(k) * {NN}];")
    lines.append("      const int tap_base = tap_base_phys(K - 1 - k);")
    for row_start in range(0, NN, N):
        decls = ", ".join(f"w{k} = wk[{k}]" for k in range(row_start, min(row_start + N, NN)))
        lines.append(f"      const float {decls};")
    lines.append("      for (int f = 0; f < num_frames; f++)")
    lines.append("      {")
    lines.append(f"        const float* src = &L.history[static_cast<size_t>(tap_base + f) * {N}];")
    lines.append(f"        float* zf = z + static_cast<size_t>(f) * {N};")
    for i in range(N):
        lines.append(f"        float a{i} = zf[{i}] + w{i} * src[0];")
    for j in range(1, N):
        for i in range(N):
            lines.append(f"        a{i} += w{j*N+i} * src[{j}];")
    for i in range(N):
        lines.append(f"        zf[{i}] = a{i};")
    lines.append("      }")
    lines.append("    }")
    lines.append("")
    lines.append("    // Final tap (K-1, offset 0) fully inlined with the post-conv tail.")
    lines.append(f"    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * {NN}];")
    lines.append("    const int tap_base_last = tap_base_phys(0);")
    for row_start in range(0, NN, N):
        decls = ", ".join(f"cw{k} = wk_last[{k}]" for k in range(row_start, min(row_start + N, NN)))
        lines.append(f"    const float {decls};")
    mw_decls = ", ".join(f"mw{i} = L.mixin_w[{i}]" for i in range(N))
    lines.append(f"    const float {mw_decls};")
    lines.append(f"    // layer1x1 col-major: lw[b*{N} + c] is weight from bottleneck b to output c.")
    for b in range(N):
        decls = ", ".join(f"lw{b}{c} = L.l1x1_w[{b*N+c}]" for c in range(N))
        lines.append(f"    const float {decls};")
    lb_decls = ", ".join(f"lb{i} = L.l1x1_b[{i}]" for i in range(N))
    lines.append(f"    const float {lb_decls};")
    lines.append("    for (int f = 0; f < num_frames; f++)")
    lines.append("    {")
    lines.append(f"      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * {N}];")
    lines.append(f"      const float* zf_mem = z + static_cast<size_t>(f) * {N};")
    lines.append("      // Final tap GEMV.")
    for i in range(N):
        lines.append(f"      float a{i} = zf_mem[{i}] + cw{i} * src[0];")
    for j in range(1, N):
        for i in range(N):
            lines.append(f"      a{i} += cw{j*N+i} * src[{j}];")
    lines.append("      // Mixin + LeakyReLU.")
    lines.append("      const float cf = cond[f];")
    for i in range(N):
        lines.append(f"      a{i} += mw{i} * cf;")
    for i in range(N):
        lines.append(f"      a{i} = (a{i} >= 0.0f) ? a{i} : a{i} * kLeakySlope;")
    lines.append("      // Head sum accumulate.")
    lines.append(f"      float* hsum = &_head_sum[static_cast<size_t>(f) * {N}];")
    for i in range(N):
        lines.append(f"      hsum[{i}] += a{i};")
    lines.append("      // layer1x1 residual.")
    lines.append(f"      float* lin = &_layer_in[static_cast<size_t>(f) * {N}];")
    for c in range(N):
        terms = " + ".join(f"lw{b}{c} * a{b}" for b in range(N))
        lines.append(f"      lin[{c}] += lb{c} + {terms};")
    lines.append("    }")
    lines.append("  }")
    lines.append(f"  // [Fork] END: hand-unrolled Channels=={N} kernel")
    return "\n".join(lines)


def gen_parametric(N):
    NN = N * N
    lines = []
    lines.append(f"  else if constexpr (Channels == {N})")
    lines.append("  {")
    lines.append("    // Preload per-layer FiLM coefficients (constant across all frames in this buffer).")
    g_decls = ", ".join(f"g{i} = _film_gamma[li][{i}]" for i in range(N))
    lines.append(f"    const float {g_decls};")
    bt_decls = ", ".join(f"bt{i} = _film_beta[li][{i}]" for i in range(N))
    lines.append(f"    const float {bt_decls};")
    lines.append("")
    lines.append("    float* z = _z.data();")
    lines.append("")
    lines.append("    // Tap 0: seed z with conv_b.")
    lines.append("    {")
    lines.append("      const float* wk = &L.conv_w[0];")
    lines.append("      const int tap_base = tap_base_phys(K - 1);")
    for row_start in range(0, NN, N):
        decls = ", ".join(f"w{k} = wk[{k}]" for k in range(row_start, min(row_start + N, NN)))
        lines.append(f"      const float {decls};")
    cb_decls = ", ".join(f"cb{i} = L.conv_b[{i}]" for i in range(N))
    lines.append(f"      const float {cb_decls};")
    lines.append("      for (int f = 0; f < num_frames; f++)")
    lines.append("      {")
    lines.append(f"        const float* src = &L.history[static_cast<size_t>(tap_base + f) * {N}];")
    lines.append(f"        float* zf = z + static_cast<size_t>(f) * {N};")
    for i in range(N):
        terms = " + ".join(f"w{j*N+i} * src[{j}]" for j in range(N))
        lines.append(f"        zf[{i}] = cb{i} + {terms};")
    lines.append("      }")
    lines.append("    }")
    lines.append("")
    lines.append("    // Taps 1..K-2: accumulate.")
    lines.append("    for (int k = 1; k < K - 1; k++)")
    lines.append("    {")
    lines.append(f"      const float* wk = &L.conv_w[static_cast<size_t>(k) * {NN}];")
    lines.append("      const int tap_base = tap_base_phys(K - 1 - k);")
    for row_start in range(0, NN, N):
        decls = ", ".join(f"w{k} = wk[{k}]" for k in range(row_start, min(row_start + N, NN)))
        lines.append(f"      const float {decls};")
    lines.append("      for (int f = 0; f < num_frames; f++)")
    lines.append("      {")
    lines.append(f"        const float* src = &L.history[static_cast<size_t>(tap_base + f) * {N}];")
    lines.append(f"        float* zf = z + static_cast<size_t>(f) * {N};")
    for i in range(N):
        terms = " + ".join(f"w{j*N+i} * src[{j}]" for j in range(N))
        lines.append(f"        zf[{i}] += {terms};")
    lines.append("      }")
    lines.append("    }")
    lines.append("")
    lines.append("    // Final tap + mixin + FiLM + LeakyReLU + [head_sum accumulate] + l1x1 residual.")
    lines.append(f"    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * {NN}];")
    lines.append("    const int tap_base_last = tap_base_phys(0);")
    for row_start in range(0, NN, N):
        decls = ", ".join(f"cw{k} = wk_last[{k}]" for k in range(row_start, min(row_start + N, NN)))
        lines.append(f"    const float {decls};")
    mw_decls = ", ".join(f"mw{i} = L.mixin_w[{i}]" for i in range(N))
    lines.append(f"    const float {mw_decls};")
    for b in range(N):
        decls = ", ".join(f"lw{b}{c} = L.l1x1_w[{b*N+c}]" for c in range(N))
        lines.append(f"    const float {decls};")
    lb_decls = ", ".join(f"lb{i} = L.l1x1_b[{i}]" for i in range(N))
    lines.append(f"    const float {lb_decls};")
    lines.append("    for (int f = 0; f < num_frames; f++)")
    lines.append("    {")
    lines.append(f"      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * {N}];")
    lines.append(f"      const float* zf = z + static_cast<size_t>(f) * {N};")
    lines.append("      // Final tap GEMV.")
    for i in range(N):
        terms = " + ".join(f"cw{j*N+i} * src[{j}]" for j in range(N))
        lines.append(f"      float a{i} = zf[{i}] + {terms};")
    lines.append("      // Mixin.")
    lines.append("      const float cf = cond[f];")
    for i in range(N):
        lines.append(f"      a{i} += mw{i} * cf;")
    lines.append("      // FiLM: gamma * (conv+mixin) + beta.")
    for i in range(N):
        lines.append(f"      a{i} = g{i} * a{i} + bt{i};")
    lines.append("      // LeakyReLU.")
    for i in range(N):
        lines.append(f"      a{i} = (a{i} >= 0.0f) ? a{i} : a{i} * kLeakySlope;")
    lines.append("      // Skip connections: accumulate this layer's post-activation into the head sum")
    lines.append("      // (same site the static A2FastModel uses -- after FiLM+LeakyReLU, BEFORE the")
    lines.append("      // l1x1 residual).")
    lines.append(f"      float* __restrict__ hsum = &_head_sum[static_cast<size_t>(f) * {N}];")
    for i in range(N):
        lines.append(f"      hsum[{i}] += a{i};")
    lines.append("      // layer1x1 residual.")
    lines.append(f"      float* lin = &_layer_in[static_cast<size_t>(f) * {N}];")
    for c in range(N):
        terms = " + ".join(f"lw{b}{c} * a{b}" for b in range(N))
        lines.append(f"      lin[{c}] += lb{c} + {terms};")
    lines.append("    }")
    lines.append("  }")
    lines.append(f"  // [Fork] END: hand-unrolled Channels=={N} kernel")
    return "\n".join(lines)


if __name__ == "__main__":
    N = int(sys.argv[1])
    variant = sys.argv[2]
    if variant == "plain":
        print(gen_nonparametric(N))
    else:
        print(gen_parametric(N))
