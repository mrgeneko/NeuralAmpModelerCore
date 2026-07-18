// [Fork] Integration test for tools/render_parametric.cpp (Gap 3).
//
// render_parametric is a pure CLI tool (parses argv, loads a parametric .nam, renders
// audio at a fixed knob setting) with no library entry point to unit-test directly. This
// test exercises the REAL compiled executable as a subprocess: it builds a tiny real
// parametric model fixture on disk, invokes the tool twice with two different --knobs
// settings against the repo's real example_audio/input.wav, and checks:
//   1. Both invocations exit 0 and produce a non-empty, finite-valued WAV.
//   2. The two knob settings produce MEASURABLY DIFFERENT output -- proving the tool
//      actually threads --knobs through to the model, not just that it runs.
//
// This must run with the current working directory at the repo root (same requirement
// as run_tests itself, for the relative path to example_audio/input.wav) and requires
// the `render_parametric` target to have been built alongside `run_tests` (wired via
// add_dependencies() in tools/CMakeLists.txt).

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if !defined(_WIN32)
  #include <sys/wait.h>
#endif

#include "json.hpp"

namespace test_render_parametric
{
namespace
{

// Runs `cmd` via the shell and returns the process's real exit code (unlike raw
// std::system(), which on POSIX returns the packed wait() status).
int run_and_get_exit_code(const std::string& cmd)
{
  const int status = std::system(cmd.c_str());
#if defined(_WIN32)
  return status;
#else
  if (status == -1)
    return -1;
  if (WIFEXITED(status))
    return WEXITSTATUS(status);
  return -1;
#endif
}

// Minimal reader for the exact 44-byte-header / float32 / mono WAV format
// render_parametric.cpp's SaveWavFloat32() writes -- avoids pulling in
// AudioDSPTools' wav.cpp just for this one test.
bool load_wav_float32(const std::string& path, std::vector<float>& samples, double& sample_rate)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open())
    return false;

  std::vector<char> header(44);
  in.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!in || in.gcount() != 44)
    return false;
  if (std::memcmp(header.data(), "RIFF", 4) != 0 || std::memcmp(header.data() + 8, "WAVE", 4) != 0
      || std::memcmp(header.data() + 36, "data", 4) != 0)
    return false;

  uint32_t sr_raw = 0, data_bytes = 0;
  std::memcpy(&sr_raw, header.data() + 24, 4);
  std::memcpy(&data_bytes, header.data() + 40, 4);
  sample_rate = static_cast<double>(sr_raw);

  samples.resize(data_bytes / sizeof(float));
  in.read(reinterpret_cast<char*>(samples.data()), static_cast<std::streamsize>(data_bytes));
  return static_cast<bool>(in) && in.gcount() == static_cast<std::streamsize>(data_bytes);
}

// Builds a tiny real parametric .nam fixture (A2 nano shape, channels=3, num_params=1)
// on disk. Weight count 2147 is the same known-correct constant test_parametric_wavenet.cpp
// derives and uses for this exact (channels=3, num_params=1) shape.
std::string write_parametric_fixture(const std::filesystem::path& dir)
{
  nlohmann::json cfg;
  cfg["layers"] = 3;
  cfg["head_scale"] = 1.0f;
  cfg["parametric"]["condition_size"] = 1;
  cfg["parametric"]["head_mode"] = "skip";
  cfg["parametric"]["schema_version"] = 1;
  cfg["parametric"]["parameters"] = {{{"name", "drive"}, {"minimum", 0.0f}, {"maximum", 1.0f}, {"default", 0.5f}}};

  constexpr int kWeightCount = 2147;

  nlohmann::json nam_file;
  nam_file["version"] = "0.7.0";
  nam_file["architecture"] = "ParametricWaveNet";
  nam_file["config"] = cfg;
  nam_file["weights"] = std::vector<float>(kWeightCount, 0.1f);
  nam_file["sample_rate"] = 48000;

  const auto path = dir / "render_parametric_fixture.param.nam";
  std::ofstream out(path);
  out << nam_file.dump();
  out.close();
  return path.string();
}

} // namespace

void test_render_parametric_respects_knobs()
{
  const std::string render_parametric_bin = "./build/tools/render_parametric";
  assert(std::filesystem::exists(render_parametric_bin)
         && "render_parametric binary not found -- build the run_tests target (which now "
            "depends on render_parametric) from the repo root before running this test");

  const std::string input_wav = "example_audio/input.wav";
  assert(std::filesystem::exists(input_wav));

  const auto tmp_dir = std::filesystem::temp_directory_path();
  const std::string model_path = write_parametric_fixture(tmp_dir);
  const std::string out_lo = (tmp_dir / "render_parametric_test_out_lo.wav").string();
  const std::string out_hi = (tmp_dir / "render_parametric_test_out_hi.wav").string();

  const std::string cmd_lo = render_parametric_bin + " " + model_path + " " + input_wav + " " + out_lo
                             + " --knobs 0.05 --block 64 > /dev/null 2>&1";
  const std::string cmd_hi = render_parametric_bin + " " + model_path + " " + input_wav + " " + out_hi
                             + " --knobs 0.95 --block 64 > /dev/null 2>&1";

  const int rc_lo = run_and_get_exit_code(cmd_lo);
  const int rc_hi = run_and_get_exit_code(cmd_hi);
  assert(rc_lo == 0);
  assert(rc_hi == 0);

  std::vector<float> samples_lo, samples_hi;
  double sr_lo = 0.0, sr_hi = 0.0;
  assert(load_wav_float32(out_lo, samples_lo, sr_lo));
  assert(load_wav_float32(out_hi, samples_hi, sr_hi));

  assert(!samples_lo.empty());
  assert(!samples_hi.empty());
  assert(samples_lo.size() == samples_hi.size());
  assert(sr_lo > 0.0);
  assert(sr_hi > 0.0);

  for (float v : samples_lo)
    assert(std::isfinite(v));
  for (float v : samples_hi)
    assert(std::isfinite(v));

  // The two knob settings (0.05 vs 0.95) must produce measurably different audio --
  // proving the tool actually respects --knobs, not just that it runs and writes a file.
  double max_diff = 0.0;
  for (size_t i = 0; i < samples_lo.size(); i++)
    max_diff = std::max(max_diff, static_cast<double>(std::fabs(samples_lo[i] - samples_hi[i])));
  if (!(max_diff > 1e-5))
  {
    std::cerr << "render_parametric: knobs 0.05 vs 0.95 produced near-identical output "
              << "(max |diff| = " << max_diff << ") -- --knobs may not be reaching the model" << std::endl;
    assert(false);
  }
  assert(max_diff > 1e-5);

  std::filesystem::remove(model_path);
  std::filesystem::remove(out_lo);
  std::filesystem::remove(out_hi);
}

} // namespace test_render_parametric
