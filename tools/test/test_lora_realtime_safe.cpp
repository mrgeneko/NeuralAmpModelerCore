// Test to verify LoRA::Process and LoRA::Process_ are real-time safe (no allocations/frees)

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

#include "NAM/lora.h"
#include "allocation_tracking.h"

namespace test_lora_realtime_safe
{
using namespace allocation_tracking;

void set_weights(nam::LoRA& lora, const int condition_dim, const int channels, const int rank, float scale)
{
  const int a_count = (channels * rank) * condition_dim + channels * rank;
  const int b_count = (rank * channels) * condition_dim + rank * channels;
  std::vector<float> weights;
  weights.reserve(a_count + b_count);
  for (int i = 0; i < a_count; i++)
    weights.push_back(scale * 0.01f * (float)(i + 1));
  for (int i = 0; i < b_count; i++)
    weights.push_back(scale * -0.02f * (float)(i + 1));
  auto it = weights.begin();
  lora.set_weights_(it);
}

// LoRA::Process with a CONSTANT condition (the ParametricWaveNet case) must not allocate --
// this exercises the condition-constant cache path (see lora.h's Process()).
void test_lora_process_constant_condition_realtime_safe()
{
  const int condition_dim = 3;
  const int channels = 8;
  const int rank = 4;
  nam::LoRA lora(condition_dim, channels, rank);

  const int maxBufferSize = 256;
  lora.SetMaxBufferSize(maxBufferSize);
  set_weights(lora, condition_dim, channels, rank, 1.0f);

  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};
  for (int buffer_size : buffer_sizes)
  {
    Eigen::MatrixXf input(channels, buffer_size);
    Eigen::MatrixXf condition(condition_dim, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.3f); // constant across the block -> cache path

    std::string test_name = "LoRA Process (constant condition) - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr,
      [&]() { lora.Process(input, condition, buffer_size); },
      nullptr, test_name.c_str());

    auto output = lora.GetOutput().leftCols(buffer_size);
    assert(output.rows() == channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(channels - 1, buffer_size - 1)));
  }
}

// LoRA::Process with a TIME-VARYING condition must also not allocate -- exercises the
// per-frame path (non-cond_constant branch).
void test_lora_process_varying_condition_realtime_safe()
{
  const int condition_dim = 3;
  const int channels = 8;
  const int rank = 4;
  nam::LoRA lora(condition_dim, channels, rank);

  const int maxBufferSize = 256;
  lora.SetMaxBufferSize(maxBufferSize);
  set_weights(lora, condition_dim, channels, rank, 1.0f);

  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};
  for (int buffer_size : buffer_sizes)
  {
    Eigen::MatrixXf input(channels, buffer_size);
    Eigen::MatrixXf condition(condition_dim, buffer_size);
    input.setConstant(0.5f);
    condition.setRandom(); // varies per column -> per-frame path

    std::string test_name = "LoRA Process (varying condition) - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr,
      [&]() { lora.Process(input, condition, buffer_size); },
      nullptr, test_name.c_str());

    auto output = lora.GetOutput().leftCols(buffer_size);
    assert(output.rows() == channels && output.cols() == buffer_size);
    assert(std::isfinite(output(0, 0)));
    assert(std::isfinite(output(channels - 1, buffer_size - 1)));
  }
}

// LoRA::Process_() in-place add must not allocate.
void test_lora_process_inplace_realtime_safe()
{
  const int condition_dim = 2;
  const int channels = 6;
  const int rank = 2;
  nam::LoRA lora(condition_dim, channels, rank);

  const int maxBufferSize = 256;
  lora.SetMaxBufferSize(maxBufferSize);
  set_weights(lora, condition_dim, channels, rank, 1.0f);

  std::vector<int> buffer_sizes{1, 8, 16, 32, 64, 128, 256};
  for (int buffer_size : buffer_sizes)
  {
    Eigen::MatrixXf input(channels, buffer_size);
    Eigen::MatrixXf condition(condition_dim, buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.4f);

    std::string test_name = "LoRA Process_ (in-place) - Buffer size " + std::to_string(buffer_size);
    run_allocation_test_no_allocations(
      nullptr,
      [&]() { lora.Process_(input, condition, buffer_size); },
      nullptr, test_name.c_str());

    assert(input.rows() == channels && input.cols() >= buffer_size);
    assert(std::isfinite(input(0, 0)));
    assert(std::isfinite(input(channels - 1, buffer_size - 1)));
  }
}

// Partial-frame processing (buffer smaller than maxBufferSize) must not allocate.
void test_lora_process_partial_frames_realtime_safe()
{
  const int condition_dim = 2;
  const int channels = 4;
  const int rank = 2;
  nam::LoRA lora(condition_dim, channels, rank);

  const int maxBufferSize = 128;
  lora.SetMaxBufferSize(maxBufferSize);
  set_weights(lora, condition_dim, channels, rank, 1.0f);

  const int full_buffer_size = 64;
  std::vector<int> partial_buffer_sizes{1, 8, 16, 32};

  for (int buffer_size : partial_buffer_sizes)
  {
    Eigen::MatrixXf input(channels, full_buffer_size);
    Eigen::MatrixXf condition(condition_dim, full_buffer_size);
    input.setConstant(0.5f);
    condition.setConstant(0.5f);

    std::string test_name =
      "LoRA Process (partial frames: " + std::to_string(buffer_size) + " of " + std::to_string(full_buffer_size) + ")";
    run_allocation_test_no_allocations(
      nullptr,
      [&]() { lora.Process(input, condition, buffer_size); },
      nullptr, test_name.c_str());

    auto output = lora.GetOutput().leftCols(buffer_size);
    assert(output.rows() == channels && output.cols() >= buffer_size);
    assert(std::isfinite(output(0, 0)));
  }
}

// Multiple consecutive calls (simulating repeated audio buffers) must not allocate --
// including a weight change mid-stream, which invalidates the constant-condition cache.
void test_lora_process_consecutive_calls_realtime_safe()
{
  const int condition_dim = 2;
  const int channels = 6;
  const int rank = 3;
  nam::LoRA lora(condition_dim, channels, rank);

  const int maxBufferSize = 256;
  lora.SetMaxBufferSize(maxBufferSize);
  set_weights(lora, condition_dim, channels, rank, 1.0f);

  const int buffer_size = 64;
  const int num_consecutive_calls = 10;

  Eigen::MatrixXf input(channels, buffer_size);
  Eigen::MatrixXf condition(condition_dim, buffer_size);
  input.setConstant(0.5f);
  condition.setConstant(0.5f);

  std::string test_name = "LoRA Process (consecutive calls: " + std::to_string(num_consecutive_calls) + " calls)";
  run_allocation_test_no_allocations(
    nullptr,
    [&]() {
      for (int i = 0; i < num_consecutive_calls; i++)
        lora.Process(input, condition, buffer_size);
    },
    nullptr, test_name.c_str());

  auto output = lora.GetOutput().leftCols(buffer_size);
  assert(output.rows() == channels && output.cols() >= buffer_size);
  assert(std::isfinite(output(0, 0)));
}
} // namespace test_lora_realtime_safe
