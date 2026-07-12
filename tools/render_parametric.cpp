// [Fork] render_parametric — render a PARAMETRIC .nam at a fixed knob setting.
//
// Exists for the FiLM-on Python<->C++ A/B (skip-accumulation migration §4.2/4.3): it is the
// only way to get the real C++ parametric path (FiLM active, knobs set) to render audio we
// can compare against param_infer.py's Python forward.
//
//   render_parametric <model.param.nam> <input.wav> <output.wav> --knobs 0.3,0.7 [--block 64]
//
// Two behaviors matter when comparing against Python:
//   * KNOB SMOOTHING. The C++ models one-pole smooth knobs from their 0.5 default toward the
//     target (~40%/buffer at 512 frames), so the first ~0.1 s of output is a RAMP, not the
//     steady-state tone. Discard the head of the signal before comparing.
//   * DC BLOCKER. The parametric models apply a ~20 Hz IIR high-pass after the net (FiLM
//     shift terms inject DC). Python does not, so the comparison must high-pass Python too.
// Both are handled by the comparison script, not here — this tool renders the product's
// real signal path, warts and all.
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "wav.h"

namespace
{
bool SaveWavFloat32(const char* fileName, const float* samples, size_t numSamples, double sampleRate)
{
  std::ofstream out(fileName, std::ios::binary);
  if (!out.is_open())
    return false;
  const uint32_t dataBytes = static_cast<uint32_t>(numSamples * sizeof(float));
  const uint32_t byteRate = static_cast<uint32_t>(sampleRate) * 4;
  auto u32 = [&out](uint32_t v) { out.write(reinterpret_cast<const char*>(&v), 4); };
  auto u16 = [&out](uint16_t v) { out.write(reinterpret_cast<const char*>(&v), 2); };
  out.write("RIFF", 4);
  u32(36 + dataBytes);
  out.write("WAVEfmt ", 8);
  u32(16);
  u16(3); // IEEE float
  u16(1); // mono
  u32(static_cast<uint32_t>(sampleRate));
  u32(byteRate);
  u16(4);
  u16(32);
  out.write("data", 4);
  u32(dataBytes);
  out.write(reinterpret_cast<const char*>(samples), dataBytes);
  return out.good();
}
} // namespace

int main(int argc, char* argv[])
{
  std::vector<std::string> pos;
  std::vector<float> knobs;
  int block = 64;

  for (int i = 1; i < argc; i++)
  {
    const std::string a(argv[i]);
    if (a == "--knobs" && i + 1 < argc)
    {
      std::stringstream ss(argv[++i]);
      std::string tok;
      while (std::getline(ss, tok, ','))
        knobs.push_back(std::strtof(tok.c_str(), nullptr));
    }
    else if (a == "--block" && i + 1 < argc)
    {
      block = std::atoi(argv[++i]);
    }
    else
    {
      pos.push_back(a);
    }
  }

  if (pos.size() < 3)
  {
    std::cerr << "Usage: render_parametric <model.param.nam> <in.wav> <out.wav> "
                 "--knobs v1,v2,... [--block N]\n";
    return 1;
  }

  auto model = nam::get_dsp(std::filesystem::path(pos[0]));
  if (!model)
  {
    std::cerr << "Failed to load model\n";
    return 1;
  }

  const int num_params = model->GetNumParams();
  std::cerr << "model: " << pos[0] << "  knobs=" << num_params << "\n";
  if (num_params <= 0)
  {
    std::cerr << "Error: model is not parametric (GetNumParams()==0). If this is a dual .nam, "
                 "the embedded parametric payload should have been preferred.\n";
    return 1;
  }
  if (static_cast<int>(knobs.size()) != num_params)
  {
    std::cerr << "Error: --knobs supplied " << knobs.size() << " values, model expects "
              << num_params << "\n";
    return 1;
  }

  std::vector<float> inputAudio;
  double sr = 0.0;
  if (dsp::wav::Load(pos[1].c_str(), inputAudio, sr) != dsp::wav::LoadReturnCode::SUCCESS)
  {
    std::cerr << "Failed to load input WAV\n";
    return 1;
  }

  const double expected = model->GetExpectedSampleRate();
  const double rate = expected > 0 ? expected : sr;

  model->Reset(rate, block);
  model->SetKnobValues(knobs);

  std::vector<NAM_SAMPLE> in(block), out(block);
  NAM_SAMPLE* ip[1] = {in.data()};
  NAM_SAMPLE* op[1] = {out.data()};

  std::vector<float> rendered;
  rendered.reserve(inputAudio.size());
  size_t i = 0;
  while (i < inputAudio.size())
  {
    const int n = static_cast<int>(std::min<size_t>(block, inputAudio.size() - i));
    for (int f = 0; f < n; f++)
      in[f] = static_cast<NAM_SAMPLE>(inputAudio[i + f]);
    for (int f = n; f < block; f++)
      in[f] = 0.0;
    model->process(ip, op, n);
    for (int f = 0; f < n; f++)
      rendered.push_back(static_cast<float>(out[f]));
    i += n;
  }

  if (!SaveWavFloat32(pos[2].c_str(), rendered.data(), rendered.size(), rate))
  {
    std::cerr << "Failed to write output WAV\n";
    return 1;
  }
  std::cerr << "wrote " << pos[2] << " (" << rendered.size() << " samples @ " << rate << " Hz)\n";
  return 0;
}
