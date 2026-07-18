# NeuralAmpModelerCore (Fork fork)

[![Build](https://github.com/mrgeneko/NeuralAmpModelerCore/actions/workflows/build.yml/badge.svg)](https://github.com/mrgeneko/NeuralAmpModelerCore/actions/workflows/build.yml)

Core C++ DSP library for NAM plugins.

This is a fork of [sdatkinson/NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore),
the reference implementation used by [NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin).
It stays source-compatible with upstream for standard (non-parametric, non-slimmable) model loading and
playback — see [`Fork additions`](#fork-additions) below for what's new, and periodically merges upstream's
own commits to stay current.

## Fork additions

Everything below is additive: existing `.nam` files, existing upstream API calls, and stock NAM plugins
all continue to work unchanged.

* **Parametric (knob-controllable) models.** A `"ParametricWaveNet"` architecture whose FiLM conditioning
  responds to live knob values instead of being baked in at training time — `SetKnobValues()`/
  `GetNumParams()`/`GetParameterDefs()` on `nam::DSP` (default no-op bodies, so existing subclasses are
  unaffected). Knob changes are smoothed per-buffer (a one-pole filter) to avoid zipper noise. A hand-optimized
  fast path (`ParametricA2FastModel<Channels>`) exists alongside the generic path for channel widths 3–8,
  mirroring the non-parametric `A2FastModel` fast path this repo already had. See `NAM/wavenet/parametric_wavenet.h`
  and `NAM/wavenet/a2_fast.h`.
* **Slimmable / multi-tier models.** `ContainerModel` holds several submodels at different capacities
  (e.g. a small/fast tier and a large/accurate tier) and switches between them at runtime via
  `SetSlimmableSize()`. Adds `GetAllowedSizes()` (the discrete sizes/channel-counts a host can pick between)
  and `PrewarmForRatio()` (lets a host stage — `Reset()` + prewarm — the next tier on a background thread
  before activating it, so the actual switch is a cheap atomic flip rather than blocking the audio thread).
  See `NAM/container.h`, `NAM/slimmable.h`.
* **Dual-payload `.nam` files.** A single `.nam` file can carry a standard baked snapshot at the top level
  (so any stock NAM plugin loads and plays it, ignoring unknown keys) plus the full parametric model nested
  under an `"embedded_parametric"` key (which a parametric-aware host reads instead, for live knobs). See
  `NAM/get_dsp.cpp`. Built by the training pipeline, not in this repo.
* **`render_parametric`**, a CLI tool for rendering a parametric `.nam` at a fixed set of knob values —
  useful for auditioning a specific tone or for regression-testing parametric models offline.

## Included Tools

This repo includes a few helpful tools.
For guidance on building them, have a look at the workflow provided in `.github/workflows/build.yml`.

* [`run_tests`](tools/CMakeLists.txt), which runs a suite of unit tests.
* [`loadmodel`](tools/CMakeLists.txt), which allows you to test loading a `.nam` file.
* [`benchmodel`](tools/CMakeLists.txt), which allows you to test how quickly a model runs in real time. _Note: For more granular profiling tools, check out the [`main-profiling`](https://github.com/sdatkinson/NeuralAmpModelerCore/tree/main-profiling) branch on upstream.
* [`render_parametric`](tools/render_parametric.cpp), which renders a parametric `.nam` file at a fixed knob setting (fork addition, see above).

## Sharp edges
This library uses [Eigen](http://eigen.tuxfamily.org) to do the linear algebra routines that its neural networks require. Since these models hold their parameters as eigen object members, there is a risk with certain compilers and compiler optimizations that their memory is not aligned properly. This can be worked around by providing two preprocessor macros: `EIGEN_MAX_ALIGN_BYTES 0` and `EIGEN_DONT_VECTORIZE`, though this will probably harm performance. See [Structs Having Eigen Members](http://eigen.tuxfamily.org/dox-3.2/group__TopicStructHavingEigenMembers.html) for more information. This is being tracked as [Issue 67](https://github.com/sdatkinson/NeuralAmpModelerCore/issues/67) on upstream.

## Sponsors

<div align="center">
  <img src="media/tone3000-logo.svg" alt="Tone3000 logo">
</div>

Development of version 0.4.0 of the original (upstream) library was generously supported by [TONE3000](https://tone3000.com).
**Thank you!**
