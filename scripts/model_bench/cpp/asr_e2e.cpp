// Encoder-decoder ASR model end-to-end in C++ (TensorRT runtime API): encoder -> first decoder step -> greedy
// decoder-with-past loop. CUDA-event timing per stage; token ids written for the (offline) WER step.
// Build: g++ -O2 -std=c++17 asr_e2e.cpp -I/usr/include/$(gcc -dumpmachine) -I/usr/local/cuda/include
//        -L/usr/local/cuda/lib64 -lnvinfer -lcudart -o asr_e2e
// Usage: ./asr_e2e <engines_dir> <prec: fp16|fp32_ref> <mel list.txt> <out.jsonl> [repeats=3] [realtime=0]
// Stdout: one "clip <id> <sec>s enc .. first .. dec med .. p99 .. tok .. wall .. ms RTF .." line per
// measured clip (the side-load harness waits for the first one), then "DONE".
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace nvinfer1;
using SteadyClock = std::chrono::steady_clock;

struct Logger : public ILogger {
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) std::cerr << message << "\n";
  }
} g_logger;

#define CK(x)                                                                             \
  do {                                                                                    \
    cudaError_t cuda_status = (x);                                                        \
    if (cuda_status != cudaSuccess) {                                                     \
      std::cerr << "CUDA " << cudaGetErrorString(cuda_status) << " @" << __LINE__ << "\n"; \
      exit(1);                                                                            \
    }                                                                                     \
  } while (0)

// Model geometry of the exported engines: 32 decoder layers, 20 heads of 64, 1500 encoder frames
// from a 128 x 3000 mel, 51866-token vocabulary, 448-token context.
static const int LAYERS = 32;
static const int HEADS = 20;
static const int HEAD_DIM = 64;
static const int ENCODER_FRAMES = 1500;
static const int HIDDEN = 1280;
static const int MEL_BINS = 128;
static const int MEL_FRAMES = 3000;
static const int VOCAB = 51866;
static const int MAX_LEN = 448;
static const int EOS = 50257;
static const int MAX_PROMPT_TOKENS = 8;
static const std::vector<int64_t> PROMPT = {50258, 50259, 50360, 50364};

struct Clip {
  int id;
  double seconds;
  std::string mel_path;
};

struct ClipTiming {
  SteadyClock::time_point t_start;
  float encoder_ms = 0;
  float first_step_ms = 0;
  std::vector<float> decode_ms;
  std::vector<int> tokens;
  double wall_ms = 0;
};

static size_t element_size(DataType type) {
  switch (type) {
    case DataType::kFLOAT:
      return 4;
    case DataType::kHALF:
      return 2;
    case DataType::kINT32:
      return 4;
    case DataType::kINT64:
      return 8;
    case DataType::kINT8:
      return 1;
    case DataType::kBOOL:
      return 1;
    default:
      return 4;
  }
}

static void* device_alloc(size_t bytes) {
  void* pointer;
  CK(cudaMalloc(&pointer, bytes));
  return pointer;
}

static std::string kv_name(const char* prefix, int layer, const char* side, const std::string& kind) {
  return std::string(prefix) + "." + std::to_string(layer) + "." + side + "." + kind;
}

struct Engine {
  ICudaEngine* engine = nullptr;
  IExecutionContext* context = nullptr;
  std::vector<std::string> inputs, outputs;
  std::map<std::string, DataType> dtype;

  void load(IRuntime* runtime, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> blob((std::istreambuf_iterator<char>(file)), {});
    engine = runtime->deserializeCudaEngine(blob.data(), blob.size());
    if (!engine) {
      std::cerr << "deserialize failed " << path << "\n";
      exit(1);
    }
    context = engine->createExecutionContext();
    for (int i = 0; i < engine->getNbIOTensors(); i++) {
      const char* name = engine->getIOTensorName(i);
      dtype[name] = engine->getTensorDataType(name);
      (engine->getTensorIOMode(name) == TensorIOMode::kINPUT ? inputs : outputs).push_back(name);
    }
  }
};

class AsrPipeline {
 public:
  AsrPipeline(const std::string& engines_dir, const std::string& precision, bool realtime)
      : realtime_(realtime) {
    IRuntime* runtime = createInferRuntime(g_logger);
    encoder_.load(runtime, engines_dir + "/encoder_" + precision + ".engine");
    decoder_first_.load(runtime, engines_dir + "/decoder_first_" + precision + ".engine");
    decoder_past_.load(runtime, engines_dir + "/decoder_past_" + precision + ".engine");
    CK(cudaStreamCreate(&stream_));
    allocate_buffers();
    for (cudaEvent_t* event : {&encoder_begin_, &encoder_end_, &first_step_end_, &step_begin_, &step_end_}) {
      CK(cudaEventCreate(event));
    }
  }

  void run_clip(const Clip& clip, std::ofstream& out, bool print) {
    ClipTiming timing;
    upload_mel(clip.mel_path);
    timing.t_start = SteadyClock::now();
    int next_token = run_encoder_and_first_step(timing);
    greedy_decode(next_token, timing);
    timing.wall_ms = std::chrono::duration<double, std::milli>(SteadyClock::now() - timing.t_start).count();
    if (print) report(clip, timing, out);
    if (realtime_) pace_to_realtime(clip.seconds, timing.wall_ms);
  }

 private:
  using KvBuffers = std::map<std::string, void*>;

  void allocate_buffers() {
    mel_ = device_alloc(MEL_BINS * MEL_FRAMES * element_size(encoder_.dtype["input_features"]));
    hidden_ =
        device_alloc((size_t)ENCODER_FRAMES * HIDDEN * element_size(encoder_.dtype["last_hidden_state"]));
    ids_ = device_alloc(MAX_PROMPT_TOKENS * 8);
    logits_first_ =
        device_alloc((size_t)MAX_PROMPT_TOKENS * VOCAB * element_size(decoder_first_.dtype["logits"]));
    logits_past_ = device_alloc((size_t)VOCAB * element_size(decoder_past_.dtype["logits"]));
    for (int layer = 0; layer < LAYERS; layer++) {
      for (std::string kind : {"key", "value"}) {
        encoder_kv_[kv_name("present", layer, "encoder", kind)] =
            device_alloc((size_t)HEADS * ENCODER_FRAMES * HEAD_DIM * 4);
        std::string self_name = kv_name("present", layer, "decoder", kind);
        self_kv_a_[self_name] = device_alloc((size_t)HEADS * MAX_LEN * HEAD_DIM * 4);
        self_kv_b_[self_name] = device_alloc((size_t)HEADS * MAX_LEN * HEAD_DIM * 4);
      }
    }
    CK(cudaMallocHost((void**)&logits_host_, (size_t)MAX_PROMPT_TOKENS * VOCAB * 4));
  }

  void upload_mel(const std::string& mel_path) {
    std::vector<float> mel(MEL_BINS * MEL_FRAMES);
    {
      std::ifstream file(mel_path, std::ios::binary);
      file.read((char*)mel.data(), mel.size() * 4);
    }
    if (encoder_.dtype["input_features"] == DataType::kHALF) {
      std::vector<__half> mel_half(mel.size());
      for (size_t i = 0; i < mel.size(); i++) mel_half[i] = __float2half(mel[i]);
      CK(cudaMemcpyAsync(mel_, mel_half.data(), mel_half.size() * 2, cudaMemcpyHostToDevice, stream_));
    } else {
      CK(cudaMemcpyAsync(mel_, mel.data(), mel.size() * 4, cudaMemcpyHostToDevice, stream_));
    }
  }

  int argmax_host() const {
    return (int)(std::max_element(logits_host_, logits_host_ + VOCAB) - logits_host_);
  }

  int run_encoder_and_first_step(ClipTiming& timing) {
    IExecutionContext* encoder = encoder_.context;
    encoder->setInputShape("input_features", Dims3(1, MEL_BINS, MEL_FRAMES));
    encoder->setTensorAddress("input_features", mel_);
    encoder->setTensorAddress("last_hidden_state", hidden_);
    CK(cudaEventRecord(encoder_begin_, stream_));
    encoder->enqueueV3(stream_);
    CK(cudaEventRecord(encoder_end_, stream_));

    IExecutionContext* first = decoder_first_.context;
    CK(cudaMemcpyAsync(ids_, PROMPT.data(), PROMPT.size() * 8, cudaMemcpyHostToDevice, stream_));
    first->setInputShape("input_ids", Dims2(1, (int)PROMPT.size()));
    first->setInputShape("encoder_hidden_states", Dims3(1, ENCODER_FRAMES, HIDDEN));
    first->setTensorAddress("input_ids", ids_);
    first->setTensorAddress("encoder_hidden_states", hidden_);
    first->setTensorAddress("logits", logits_first_);
    for (const auto& name : decoder_first_.outputs) {
      if (name == "logits") continue;
      bool is_cross = name.find(".encoder.") != std::string::npos;
      first->setTensorAddress(name.c_str(), is_cross ? encoder_kv_[name] : self_kv_a_[name]);
    }
    first->enqueueV3(stream_);
    CK(cudaEventRecord(first_step_end_, stream_));
    const char* last_prompt_logits = (char*)logits_first_ + (size_t)(PROMPT.size() - 1) * VOCAB * 4;
    CK(cudaMemcpyAsync(logits_host_, last_prompt_logits, VOCAB * 4, cudaMemcpyDeviceToHost, stream_));
    CK(cudaStreamSynchronize(stream_));
    CK(cudaEventElapsedTime(&timing.encoder_ms, encoder_begin_, encoder_end_));
    CK(cudaEventElapsedTime(&timing.first_step_ms, encoder_end_, first_step_end_));
    return argmax_host();
  }

  // Self-attention KV ping-pongs between two buffer sets: the step reads "current" and writes "next".
  void bind_decode_step(int past_tokens, KvBuffers& current, KvBuffers& next) {
    IExecutionContext* step = decoder_past_.context;
    step->setInputShape("input_ids", Dims2(1, 1));
    step->setTensorAddress("input_ids", ids_);
    step->setTensorAddress("logits", logits_past_);
    for (int layer = 0; layer < LAYERS; layer++) {
      for (std::string kind : {"key", "value"}) {
        std::string past_self = kv_name("past_key_values", layer, "decoder", kind);
        std::string past_cross = kv_name("past_key_values", layer, "encoder", kind);
        std::string present_self = kv_name("present", layer, "decoder", kind);
        step->setInputShape(past_self.c_str(), Dims4(1, HEADS, past_tokens, HEAD_DIM));
        step->setTensorAddress(past_self.c_str(), current[present_self]);
        step->setInputShape(past_cross.c_str(), Dims4(1, HEADS, ENCODER_FRAMES, HEAD_DIM));
        step->setTensorAddress(past_cross.c_str(), encoder_kv_[kv_name("present", layer, "encoder", kind)]);
        step->setTensorAddress(present_self.c_str(), next[present_self]);
      }
    }
  }

  void greedy_decode(int next_token, ClipTiming& timing) {
    timing.tokens.push_back(next_token);
    int past_tokens = (int)PROMPT.size();
    KvBuffers* current = &self_kv_a_;
    KvBuffers* next = &self_kv_b_;
    while (next_token != EOS && past_tokens < MAX_LEN - 1) {
      int64_t token = next_token;
      CK(cudaMemcpyAsync(ids_, &token, 8, cudaMemcpyHostToDevice, stream_));
      bind_decode_step(past_tokens, *current, *next);
      CK(cudaEventRecord(step_begin_, stream_));
      decoder_past_.context->enqueueV3(stream_);
      CK(cudaEventRecord(step_end_, stream_));
      CK(cudaMemcpyAsync(logits_host_, logits_past_, VOCAB * 4, cudaMemcpyDeviceToHost, stream_));
      CK(cudaStreamSynchronize(stream_));
      float step_ms;
      CK(cudaEventElapsedTime(&step_ms, step_begin_, step_end_));
      timing.decode_ms.push_back(step_ms);
      next_token = argmax_host();
      timing.tokens.push_back(next_token);
      past_tokens++;
      std::swap(current, next);
    }
  }

  static void report(const Clip& clip, const ClipTiming& timing, std::ofstream& out) {
    std::vector<float> sorted = timing.decode_ms;
    std::sort(sorted.begin(), sorted.end());
    float median = sorted.empty() ? 0 : sorted[sorted.size() / 2];
    float p99 = sorted.empty() ? 0 : sorted[std::min(sorted.size() - 1, (size_t)(0.99 * sorted.size()))];
    float ttft = timing.encoder_ms + timing.first_step_ms;
    float gpu_total = ttft + std::accumulate(timing.decode_ms.begin(), timing.decode_ms.end(), 0.0f);
    double rtf = timing.wall_ms / 1000.0 / clip.seconds;
    long long t_start_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(timing.t_start.time_since_epoch()).count();
    out << "{\"clip\":" << clip.id << ",\"t_start_ns\":" << t_start_ns << ",\"seconds\":" << clip.seconds
        << ",\"encoder_ms\":" << timing.encoder_ms << ",\"first_step_ms\":" << timing.first_step_ms
        << ",\"ttft_ms\":" << ttft << ",\"decode_ms_median\":" << median << ",\"decode_ms_p99\":" << p99
        << ",\"decode_ms_max\":" << (sorted.empty() ? 0 : sorted.back())
        << ",\"n_tokens\":" << timing.tokens.size() << ",\"gpu_total_ms\":" << gpu_total
        << ",\"wall_ms\":" << timing.wall_ms << ",\"rtf_wall\":" << rtf << ",\"tokens\":[";
    for (size_t i = 0; i < timing.tokens.size(); i++) out << (i ? "," : "") << timing.tokens[i];
    out << "]}\n";
    out.flush();
    std::cout << "clip " << clip.id << " " << clip.seconds << "s enc " << timing.encoder_ms << " first "
              << timing.first_step_ms << " dec med " << median << " p99 " << p99 << " tok "
              << timing.tokens.size() << " wall " << timing.wall_ms << " ms RTF " << rtf << "\n";
  }

  static void pace_to_realtime(double clip_seconds, double wall_ms) {
    double sleep_ms = clip_seconds * 1000.0 - wall_ms;
    if (sleep_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds((long)sleep_ms));
  }

  bool realtime_;
  Engine encoder_, decoder_first_, decoder_past_;
  cudaStream_t stream_;
  void* mel_ = nullptr;
  void* hidden_ = nullptr;
  void* ids_ = nullptr;
  void* logits_first_ = nullptr;
  void* logits_past_ = nullptr;
  float* logits_host_ = nullptr;
  KvBuffers encoder_kv_, self_kv_a_, self_kv_b_;
  cudaEvent_t encoder_begin_, encoder_end_, first_step_end_, step_begin_, step_end_;
};

static std::vector<Clip> read_clip_list(const std::string& list_path) {
  std::vector<Clip> clips;
  std::ifstream file(list_path);
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream fields(line);
    Clip clip;
    fields >> clip.id >> clip.seconds >> clip.mel_path;
    clips.push_back(clip);
  }
  return clips;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: engines_dir prec mel_list out.jsonl [repeats] [realtime]\n";
    return 1;
  }
  std::string engines_dir = argv[1], precision = argv[2], list_path = argv[3], out_path = argv[4];
  int repeats = argc > 5 ? atoi(argv[5]) : 3;
  bool realtime = argc > 6 ? atoi(argv[6]) != 0 : false;
  AsrPipeline pipeline(engines_dir, precision, realtime);
  std::ofstream out(out_path);
  std::vector<Clip> clips = read_clip_list(list_path);
  pipeline.run_clip(clips[0], out, false);  // warm-up, twice
  pipeline.run_clip(clips[0], out, false);
  for (int repeat = 0; repeat < repeats; repeat++) {
    for (const auto& clip : clips) pipeline.run_clip(clip, out, true);
  }
  std::cout << "DONE\n";
  return 0;
}
