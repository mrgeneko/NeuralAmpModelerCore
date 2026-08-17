// Tests for LoRA (low-rank knob-conditioned additive weight update)

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <vector>

#include "NAM/lora.h"

namespace test_lora
{
// Fills net_A/net_B's weight matrices with zero and their biases with the given flat
// vectors, so LoRA's A/B outputs are FIXED constants regardless of condition -- the same
// bias-only trick test_film.cpp uses, letting us hand-compute the expected A@(B@x).
void set_bias_only_weights(nam::LoRA& lora, const int condition_dim, const int channels, const int rank,
                           const std::vector<float>& bias_A, const std::vector<float>& bias_B)
{
  std::vector<float> weights;
  const int a_weight_count = (channels * rank) * condition_dim;
  const int b_weight_count = (rank * channels) * condition_dim;
  weights.resize(a_weight_count, 0.0f);
  weights.insert(weights.end(), bias_A.begin(), bias_A.end());
  weights.resize(weights.size() + b_weight_count, 0.0f);
  weights.insert(weights.end(), bias_B.begin(), bias_B.end());

  auto it = weights.begin();
  lora.set_weights_(it);
  assert(it == weights.end());
}

void test_set_max_buffer_size()
{
  const int condition_dim = 2;
  const int channels = 3;
  const int rank = 2;
  nam::LoRA lora(condition_dim, channels, rank);

  const int maxBufferSize = 128;
  lora.SetMaxBufferSize(maxBufferSize);

  const auto out = lora.GetOutput();
  assert(out.rows() == channels);
  assert(out.cols() == maxBufferSize);
  assert(lora.get_condition_dim() == condition_dim);
  assert(lora.get_channels() == channels);
  assert(lora.get_rank() == rank);
}

void test_process_matches_hand_computed_matmul()
{
  // A = [[1,0],[0,1],[1,1]] (channels=3, rank=2), row-major flat: 1,0, 0,1, 1,1
  // B = [[1,0,1],[0,1,1]]   (rank=2, channels=3), row-major flat: 1,0,1, 0,1,1
  const int condition_dim = 2;
  const int channels = 3;
  const int rank = 2;
  nam::LoRA lora(condition_dim, channels, rank);

  const int maxBufferSize = 64;
  lora.SetMaxBufferSize(maxBufferSize);

  set_bias_only_weights(lora, condition_dim, channels, rank, //
                        {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, // bias_A
                        {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f}); // bias_B

  const int num_frames = 2;
  Eigen::MatrixXf input(channels, num_frames);
  input << 1.0f, 4.0f, //
    2.0f, 5.0f, //
    3.0f, 6.0f;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom(); // doesn't matter -- net_A/net_B weights are zero, only bias matters

  lora.Process(input, condition, num_frames);
  const auto out = lora.GetOutput().leftCols(num_frames);

  // Hand-computed A @ (B @ input):
  //   Bx = [[4,10],[5,11]]
  //   A @ Bx = [[4,10],[5,11],[9,21]]
  const float expected[3][2] = {{4.0f, 10.0f}, {5.0f, 11.0f}, {9.0f, 21.0f}};
  for (int c = 0; c < channels; c++)
    for (int t = 0; t < num_frames; t++)
      assert(std::abs(out(c, t) - expected[c][t]) < 1e-5f);
}

void test_process_inplace_adds_to_input()
{
  const int condition_dim = 2;
  const int channels = 3;
  const int rank = 2;
  nam::LoRA lora(condition_dim, channels, rank);

  const int maxBufferSize = 64;
  lora.SetMaxBufferSize(maxBufferSize);

  set_bias_only_weights(lora, condition_dim, channels, rank, //
                        {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, //
                        {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f});

  const int num_frames = 2;
  Eigen::MatrixXf input(channels, num_frames);
  input << 1.0f, 4.0f, //
    2.0f, 5.0f, //
    3.0f, 6.0f;
  const Eigen::MatrixXf input_original = input;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom();

  lora.Process_(input, condition, num_frames);

  const float delta[3][2] = {{4.0f, 10.0f}, {5.0f, 11.0f}, {9.0f, 21.0f}};
  for (int c = 0; c < channels; c++)
    for (int t = 0; t < num_frames; t++)
      assert(std::abs(input(c, t) - (input_original(c, t) + delta[c][t])) < 1e-5f);
}

void test_process_partial_frames()
{
  // Only the first num_frames columns should be touched, matching FiLM's contract
  // (see test_film.cpp's test_process_inplace_partial_frames).
  const int condition_dim = 2;
  const int channels = 3;
  const int rank = 2;
  nam::LoRA lora(condition_dim, channels, rank);

  const int maxBufferSize = 64;
  lora.SetMaxBufferSize(maxBufferSize);

  set_bias_only_weights(lora, condition_dim, channels, rank, //
                        {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, //
                        {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f});

  const int total_cols = 4;
  const int num_frames = 2;
  Eigen::MatrixXf input(channels, total_cols);
  input << 1.0f, 4.0f, 100.0f, 200.0f, //
    2.0f, 5.0f, 300.0f, 400.0f, //
    3.0f, 6.0f, 500.0f, 600.0f;
  const Eigen::MatrixXf input_original = input;

  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom();

  lora.Process_(input, condition, num_frames);

  const float delta[3][2] = {{4.0f, 10.0f}, {5.0f, 11.0f}, {9.0f, 21.0f}};
  for (int c = 0; c < channels; c++)
  {
    for (int t = 0; t < num_frames; t++)
      assert(std::abs(input(c, t) - (input_original(c, t) + delta[c][t])) < 1e-5f);
    for (int t = num_frames; t < total_cols; t++)
      assert(std::abs(input(c, t) - input_original(c, t)) < 1e-6f);
  }
}

void test_zero_weights_produce_zero_output()
{
  // All-zero net_A/net_B (weights AND biases) => A=0 (or B=0) => delta == 0 for any input --
  // sanity check on the degenerate case, mirroring param_train.py's LoRA zero-init contract
  // (net_A zero-init alone is already sufficient there; here we go further and zero both).
  const int condition_dim = 2;
  const int channels = 3;
  const int rank = 2;
  nam::LoRA lora(condition_dim, channels, rank);

  const int maxBufferSize = 32;
  lora.SetMaxBufferSize(maxBufferSize);

  std::vector<float> weights((channels * rank) * condition_dim + (channels * rank) //
                             + (rank * channels) * condition_dim + (rank * channels),
                             0.0f);
  auto it = weights.begin();
  lora.set_weights_(it);
  assert(it == weights.end());

  const int num_frames = 3;
  Eigen::MatrixXf input(channels, num_frames);
  input.setRandom();
  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition.setRandom();

  lora.Process(input, condition, num_frames);
  const auto out = lora.GetOutput().leftCols(num_frames);
  for (int c = 0; c < channels; c++)
    for (int t = 0; t < num_frames; t++)
      assert(std::abs(out(c, t)) < 1e-6f);
}

void test_process_varying_condition_matches_per_frame_recompute()
{
  // When the condition is NOT constant across the block, LoRA must take the per-frame path
  // (see the non-cond_constant branch in lora.h) and still match processing each frame alone.
  const int condition_dim = 2;
  const int channels = 3;
  const int rank = 2;
  nam::LoRA lora(condition_dim, channels, rank);

  const int maxBufferSize = 64;
  lora.SetMaxBufferSize(maxBufferSize);

  // Non-zero weight matrices this time (not just bias) so A/B genuinely vary with condition.
  std::vector<float> weights;
  const int a_weight_count = (channels * rank) * condition_dim;
  const int b_weight_count = (rank * channels) * condition_dim;
  for (int i = 0; i < a_weight_count; i++)
    weights.push_back(0.1f * (float)(i + 1));
  for (int i = 0; i < channels * rank; i++)
    weights.push_back(0.01f * (float)i);
  for (int i = 0; i < b_weight_count; i++)
    weights.push_back(-0.05f * (float)(i + 1));
  for (int i = 0; i < rank * channels; i++)
    weights.push_back(0.02f * (float)i);
  auto it = weights.begin();
  lora.set_weights_(it);
  assert(it == weights.end());

  const int num_frames = 3;
  Eigen::MatrixXf input(channels, num_frames);
  input << 1.0f, -2.0f, 0.5f, //
    2.0f, 1.0f, -1.5f, //
    -1.0f, 3.0f, 2.0f;
  Eigen::MatrixXf condition(condition_dim, num_frames);
  condition << 0.1f, 0.9f, 0.3f, //
    0.4f, -0.2f, 0.7f;

  lora.Process(input, condition, num_frames);
  const Eigen::MatrixXf block_result = lora.GetOutput().leftCols(num_frames);

  for (int f = 0; f < num_frames; f++)
  {
    nam::LoRA single(condition_dim, channels, rank);
    single.SetMaxBufferSize(maxBufferSize);
    auto it2 = weights.begin();
    single.set_weights_(it2);
    single.Process(input.col(f), condition.col(f), 1);
    for (int c = 0; c < channels; c++)
      assert(std::abs(single.GetOutput()(c, 0) - block_result(c, f)) < 1e-5f);
  }
}
} // namespace test_lora
