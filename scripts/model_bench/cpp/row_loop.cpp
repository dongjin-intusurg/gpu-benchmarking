// row_loop - paced TensorRT engine loop for the co-location arms (no Python in the timed path).
//
// One row = one engine (or a "chain:" of engines timed as one row) enqueued at a target rate for N
// seconds on its own CUDA stream; every frame is timed with CUDA events and traced. Three ways to run:
//
//   row_loop <engine> <name> <hz> <deadline_ms> <seconds> <out.json> [trtexec-style flags...]
//       one process per row (plain / MPS / MIG arms). The harness arms a shared start:
//         ROW_READY_DIR   the row touches <dir>/ready_<name> once its engines are warm
//         ROW_START_FILE  the row waits for this file to hold the start time (steady-clock ns);
//                         no file within ROW_START_TIMEOUT_S (default 600) -> rc 3, row FAILED
//         ROW_TRACE_DIR   per-frame trace <dir>/trace_<name>.csv  (frame,t_launch_ns,t_done_ns,gpu_ms)
//         ROW_PHASE_MS    optional per-row offset inside the period (phase-shifted regime)
//       Without ROW_START_FILE the row starts on its own (paced solo reference).
//   row_loop --multi <seconds> <out.json> "name|engine|hz|deadline_ms|prio|flag flag ..."...
//       every row in ONE process / ONE CUDA context on its own stream (streams arm); prio is the
//       CUDA stream priority (numerically lower = higher). In-process barrier, shared start +1 s.
//   row_loop --prio-range
//       prints "<lowest> <highest>" (cudaDeviceGetStreamPriorityRange) and exits.
//
// Flags honoured: --staticPlugins=a.so,b.so  --shapes=t:1x3x224x224,...  --loadInputs=t:file.bin,...
// Chain: engine = "chain:e1.engine+e2.engine", flags grouped per stage with a ";;" token between
// groups; a later stage's input is wired to the same-named, same-sized output of an earlier stage.
// hz 0 = back-to-back (no pacing). Output: one JSON line per row (row_json below) plus the stdout
// summary line "<name> prio=.. n=.. p50=.. p99=.. miss=..% hz=.." that the harness parses.
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace nvinfer1;
using SteadyClock = std::chrono::steady_clock;

struct ErrorLogger : ILogger {
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) fprintf(stderr, "[TRT] %s\n", message);
  }
} g_logger;

#define CK(x)                                                                                  \
  do {                                                                                         \
    cudaError_t cuda_status = (x);                                                             \
    if (cuda_status != cudaSuccess) {                                                          \
      fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(cuda_status), __LINE__);             \
      exit(2);                                                                                 \
    }                                                                                          \
  } while (0)

static const int WARMUP_FRAMES = 20;
static const int START_FILE_TIMEOUT_DEFAULT_S = 600;
static const int BARRIER_POLL_MS = 10;
static const int BARRIER_MAX_POLLS = 60000;
static const int RC_ENGINE_OR_INPUT = 2;
static const int RC_NO_START = 3;
static const int RC_NO_FRAMES = 4;

// --multi: every row increments g_ready once warm; main arms one shared start for all of them.
static std::atomic<int> g_ready{0};
static std::atomic<long long> g_start_ns{0};
static int g_multi_rows = 0;

struct Stage {
  std::string engine_path;
  ICudaEngine* engine = nullptr;
  IExecutionContext* context = nullptr;
  std::map<std::string, std::pair<void*, size_t>> outputs;  // name -> (device pointer, bytes)
  std::vector<void*> owned_buffers;
  cudaEvent_t done_event;
};

struct StageOptions {
  std::map<std::string, Dims> shapes;
  std::map<std::string, std::string> input_files;
};

struct RandomInputs {
  std::mt19937 generator{0};
  std::uniform_real_distribution<float> uniform{0.f, 1.f};
  float next() { return uniform(generator); }
};

static std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> parts;
  std::stringstream stream(text);
  std::string part;
  while (std::getline(stream, part, delimiter)) {
    if (!part.empty()) parts.push_back(part);
  }
  return parts;
}

static size_t element_size(DataType type) {
  switch (type) {
    case DataType::kFLOAT:
    case DataType::kINT32:
      return 4;
    case DataType::kHALF:
    case DataType::kBF16:
      return 2;
    case DataType::kINT64:
      return 8;
    default:
      return 1;
  }
}

static long long ns_of(SteadyClock::time_point t) {
  return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

static float percentile(const std::vector<float>& sorted, double p) {
  if (sorted.empty()) return 0.f;
  return sorted[std::min(sorted.size() - 1, (size_t)(p * (sorted.size() - 1) + 0.5))];
}

static std::string basename_of(const std::string& path) {
  return path.substr(path.find_last_of('/') + 1);
}

static uint16_t float_to_half_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, 4);
  return (uint16_t)(((bits >> 16) & 0x8000) | ((((bits >> 23) & 0xff) - 112) << 10) | ((bits >> 13) & 0x3ff));
}

static std::vector<std::string> engine_paths_of(const std::string& engine_spec) {
  if (engine_spec.rfind("chain:", 0) == 0) return split(engine_spec.substr(6), '+');
  return {engine_spec};
}

// Flags belong to the current stage until a ";;" token moves to the next one.
static std::vector<std::vector<std::string>> group_flags_by_stage(const std::vector<std::string>& flags,
                                                                  size_t n_stages) {
  std::vector<std::vector<std::string>> grouped(n_stages);
  size_t stage = 0;
  for (const auto& flag : flags) {
    if (flag == ";;") {
      if (stage + 1 < n_stages) stage++;
      continue;
    }
    grouped[stage].push_back(flag);
  }
  return grouped;
}

static Dims parse_dims(const std::string& text) {
  Dims dims{};
  auto axes = split(text, 'x');
  dims.nbDims = (int)axes.size();
  for (int axis = 0; axis < dims.nbDims; axis++) dims.d[axis] = atoi(axes[axis].c_str());
  return dims;
}

static bool load_plugins(const std::string& library_list) {
  for (const auto& library : split(library_list, ',')) {
    if (dlopen(library.c_str(), RTLD_NOW | RTLD_GLOBAL)) continue;
    fprintf(stderr, "dlopen %s: %s\n", library.c_str(), dlerror());
    return false;
  }
  return true;
}

static bool parse_stage_flags(const std::string& row_name, const std::vector<std::string>& flags,
                              StageOptions& options) {
  for (const auto& flag : flags) {
    if (flag.rfind("--staticPlugins=", 0) == 0) {
      if (!load_plugins(flag.substr(16))) return false;
    } else if (flag.rfind("--shapes=", 0) == 0) {
      for (const auto& entry : split(flag.substr(9), ',')) {
        auto colon = entry.find(':');
        options.shapes[entry.substr(0, colon)] = parse_dims(entry.substr(colon + 1));
      }
    } else if (flag.rfind("--loadInputs=", 0) == 0) {
      for (const auto& entry : split(flag.substr(13), ',')) {
        auto colon = entry.find(':');
        options.input_files[entry.substr(0, colon)] = entry.substr(colon + 1);
      }
    } else {
      fprintf(stderr, "[row_loop] %s: flag ignored: %s\n", row_name.c_str(), flag.c_str());
    }
  }
  return true;
}

static bool load_engine(IRuntime* runtime, Stage& stage) {
  std::ifstream file(stage.engine_path, std::ios::binary);
  if (!file) {
    fprintf(stderr, "engine unreadable: %s\n", stage.engine_path.c_str());
    return false;
  }
  std::vector<char> blob((std::istreambuf_iterator<char>(file)), {});
  stage.engine = runtime->deserializeCudaEngine(blob.data(), blob.size());
  if (!stage.engine) {
    fprintf(stderr, "deserialize failed: %s\n", stage.engine_path.c_str());
    return false;
  }
  stage.context = stage.engine->createExecutionContext();
  return true;
}

// Chain wiring: an earlier stage's output with the same name and byte size feeds this input directly.
static void* wired_output(const std::vector<Stage>& stages, size_t stage_index, const char* name,
                          size_t bytes) {
  for (size_t earlier = 0; earlier < stage_index; earlier++) {
    auto found = stages[earlier].outputs.find(name);
    if (found != stages[earlier].outputs.end() && found->second.second == bytes) return found->second.first;
  }
  return nullptr;
}

static bool fill_host_input(const char* name, DataType type, size_t count, const StageOptions& options,
                            RandomInputs& random, std::vector<char>& host) {
  auto file_entry = options.input_files.find(name);
  if (file_entry != options.input_files.end()) {
    std::ifstream file(file_entry->second, std::ios::binary);
    if (!file) {
      fprintf(stderr, "loadInputs %s: unreadable %s\n", name, file_entry->second.c_str());
      return false;
    }
    file.read(host.data(), host.size());
    return true;
  }
  if (type == DataType::kFLOAT) {
    float* values = (float*)host.data();
    for (size_t i = 0; i < count; i++) values[i] = random.next();
  } else if (type == DataType::kHALF) {
    uint16_t* values = (uint16_t*)host.data();
    for (size_t i = 0; i < count; i++) values[i] = float_to_half_bits(random.next());
  }
  return true;
}

// Inputs must be shaped before output shapes can be queried, hence the separate output pass.
static bool bind_inputs(std::vector<Stage>& stages, size_t stage_index, const StageOptions& options,
                        RandomInputs& random) {
  Stage& stage = stages[stage_index];
  ICudaEngine* engine = stage.engine;
  for (int i = 0; i < engine->getNbIOTensors(); i++) {
    const char* name = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(name) != TensorIOMode::kINPUT) continue;
    auto shape_entry = options.shapes.find(name);
    Dims dims = shape_entry != options.shapes.end() ? shape_entry->second : engine->getTensorShape(name);
    for (int axis = 0; axis < dims.nbDims; axis++) {
      if (dims.d[axis] < 0) dims.d[axis] = 1;
    }
    if (!stage.context->setInputShape(name, dims)) {
      fprintf(stderr, "setInputShape %s failed\n", name);
      return false;
    }
    size_t count = 1;
    for (int axis = 0; axis < dims.nbDims; axis++) count *= dims.d[axis];
    DataType type = engine->getTensorDataType(name);
    size_t bytes = count * element_size(type);
    if (void* wired = wired_output(stages, stage_index, name, bytes)) {
      stage.context->setTensorAddress(name, wired);
      fprintf(stderr, "[chain] %s <- %s (wired, %zu bytes)\n", name,
              stages[stage_index - 1].engine_path.c_str(), bytes);
      continue;
    }
    std::vector<char> host(bytes, 0);
    if (!fill_host_input(name, type, count, options, random, host)) return false;
    void* device;
    CK(cudaMalloc(&device, bytes));
    CK(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice));
    stage.owned_buffers.push_back(device);
    stage.context->setTensorAddress(name, device);
  }
  return true;
}

static void bind_outputs(Stage& stage) {
  ICudaEngine* engine = stage.engine;
  for (int i = 0; i < engine->getNbIOTensors(); i++) {
    const char* name = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(name) != TensorIOMode::kOUTPUT) continue;
    Dims dims = stage.context->getTensorShape(name);
    size_t count = 1;
    for (int axis = 0; axis < dims.nbDims; axis++) count *= (dims.d[axis] > 0 ? dims.d[axis] : 1);
    size_t bytes = count * element_size(engine->getTensorDataType(name));
    void* device;
    CK(cudaMalloc(&device, bytes));
    stage.owned_buffers.push_back(device);
    stage.context->setTensorAddress(name, device);
    stage.outputs[name] = {device, bytes};
  }
}

// One frame = every stage back-to-back on the row's stream; returns the frame's GPU time (ms)
// and, for chains, appends each stage's share to stage_ms.
static float run_frame(std::vector<Stage>& stages, cudaStream_t stream, cudaEvent_t frame_begin,
                       cudaEvent_t frame_end, std::vector<std::vector<float>>& stage_ms) {
  CK(cudaEventRecord(frame_begin, stream));
  for (auto& stage : stages) {
    if (!stage.context->enqueueV3(stream)) {
      fprintf(stderr, "enqueue failed (%s)\n", stage.engine_path.c_str());
      exit(RC_ENGINE_OR_INPUT);
    }
    CK(cudaEventRecord(stage.done_event, stream));
  }
  CK(cudaEventRecord(frame_end, stream));
  CK(cudaEventSynchronize(frame_end));
  float frame_ms;
  CK(cudaEventElapsedTime(&frame_ms, frame_begin, frame_end));
  if (stages.size() > 1) {
    cudaEvent_t previous = frame_begin;
    for (size_t k = 0; k < stages.size(); k++) {
      float ms;
      CK(cudaEventElapsedTime(&ms, previous, stages[k].done_event));
      stage_ms[k].push_back(ms);
      previous = stages[k].done_event;
    }
  }
  return frame_ms;
}

static void announce_ready(const std::string& row_name) {
  const char* ready_dir = std::getenv("ROW_READY_DIR");
  if (!ready_dir) return;
  std::string ready_path = std::string(ready_dir) + "/ready_" + row_name;
  FILE* ready_file = fopen(ready_path.c_str(), "w");
  if (ready_file) fclose(ready_file);
}

static long long poll_start_file(const char* start_file, int timeout_s) {
  for (int poll = 0; poll < timeout_s * 100; poll++) {
    std::ifstream in(start_file);
    long long value = 0;
    if (in && (in >> value) && value > 0) return value;
    std::this_thread::sleep_for(std::chrono::milliseconds(BARRIER_POLL_MS));
  }
  return 0;
}

static long long wait_multi_barrier() {
  g_ready++;
  for (int poll = 0; poll < BARRIER_MAX_POLLS && g_start_ns.load() == 0; poll++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(BARRIER_POLL_MS));
  }
  return g_start_ns.load();
}

// Resolves the shared start (harness start file or --multi barrier); start_ns stays 0 for a solo row.
static int resolve_shared_start(const std::string& row_name, long long& start_ns, std::string& start_source) {
  start_ns = 0;
  start_source = "none";
  if (const char* start_file = std::getenv("ROW_START_FILE")) {
    const char* timeout_env = std::getenv("ROW_START_TIMEOUT_S");
    int timeout_s = timeout_env ? atoi(timeout_env) : START_FILE_TIMEOUT_DEFAULT_S;
    start_ns = poll_start_file(start_file, timeout_s);
    if (!start_ns) {
      fprintf(stderr, "[row_loop] %s: no start file after %d s - the harness never armed the grid; "
              "row FAILED\n", row_name.c_str(), timeout_s);
      return RC_NO_START;
    }
    start_source = "start_file";
    return 0;
  }
  if (g_multi_rows > 0) {
    start_ns = wait_multi_barrier();
    if (!start_ns) {
      fprintf(stderr, "[row_loop] %s: in-process barrier never released; row FAILED\n", row_name.c_str());
      return RC_NO_START;
    }
    start_source = "multi_barrier";
  }
  return 0;
}

static SteadyClock::time_point sleep_until_start(const std::string& row_name, long long start_ns,
                                                 double phase_ms) {
  auto start = SteadyClock::time_point(std::chrono::nanoseconds(start_ns)) +
               std::chrono::microseconds((long long)(phase_ms * 1000.0));
  if (start > SteadyClock::now()) {
    std::this_thread::sleep_until(start);
  } else {
    fprintf(stderr, "[row_loop] %s: start time already passed by %.1f ms\n", row_name.c_str(),
            std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count());
  }
  return start;
}

static FILE* open_trace(const std::string& row_name) {
  const char* trace_dir = std::getenv("ROW_TRACE_DIR");
  if (!trace_dir) return nullptr;
  std::string trace_path = std::string(trace_dir) + "/trace_" + row_name + ".csv";
  FILE* trace = fopen(trace_path.c_str(), "w");
  if (trace) fprintf(trace, "frame,t_launch_ns,t_done_ns,gpu_ms\n");
  return trace;
}

static std::string stages_json(const std::vector<Stage>& stages,
                               const std::vector<std::vector<float>>& stage_ms) {
  if (stages.size() <= 1) return "";
  std::string json = ",\"stages\":[";
  for (size_t k = 0; k < stages.size(); k++) {
    std::vector<float> sorted = stage_ms[k];
    std::sort(sorted.begin(), sorted.end());
    char entry[256];
    snprintf(entry, sizeof entry, "%s{\"engine\":\"%s\",\"p50_ms\":%.4f,\"p99_ms\":%.4f}",
             k ? "," : "", basename_of(stages[k].engine_path).c_str(), percentile(sorted, 0.5),
             percentile(sorted, 0.99));
    json += entry;
  }
  return json + "]";
}

static int runRow(const std::string& engine_spec, const std::string& row_name, double hz, double deadline_ms,
                  double seconds, const std::string& out_path, const std::vector<std::string>& flags,
                  int prio, std::string& line) {
  std::vector<std::string> engine_paths = engine_paths_of(engine_spec);
  std::vector<std::vector<std::string>> stage_flags = group_flags_by_stage(flags, engine_paths.size());
  initLibNvInferPlugins(&g_logger, "");
  cudaStream_t stream;
  CK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, prio));
  IRuntime* runtime = createInferRuntime(g_logger);
  std::vector<Stage> stages(engine_paths.size());
  RandomInputs random;
  for (size_t k = 0; k < stages.size(); k++) {
    stages[k].engine_path = engine_paths[k];
    StageOptions options;
    if (!parse_stage_flags(row_name, stage_flags[k], options)) return RC_ENGINE_OR_INPUT;
    if (!load_engine(runtime, stages[k])) return RC_ENGINE_OR_INPUT;
    if (!bind_inputs(stages, k, options, random)) return RC_ENGINE_OR_INPUT;
    bind_outputs(stages[k]);
    CK(cudaEventCreate(&stages[k].done_event));
  }

  cudaEvent_t frame_begin, frame_end;
  CK(cudaEventCreate(&frame_begin));
  CK(cudaEventCreate(&frame_end));
  std::vector<std::vector<float>> stage_ms(stages.size());
  for (int i = 0; i < WARMUP_FRAMES; i++) run_frame(stages, stream, frame_begin, frame_end, stage_ms);
  for (auto& samples : stage_ms) samples.clear();

  auto period = std::chrono::duration<double>(hz > 0 ? 1.0 / hz : 0.0);
  std::vector<float> latencies_ms;
  long overruns = 0;
  auto t0 = SteadyClock::now();
  auto next_launch = t0;
  announce_ready(row_name);
  long long start_ns;
  std::string start_source;
  if (int rc = resolve_shared_start(row_name, start_ns, start_source)) return rc;
  double phase_ms = std::getenv("ROW_PHASE_MS") ? atof(std::getenv("ROW_PHASE_MS")) : 0.0;
  if (start_ns > 0) {
    t0 = sleep_until_start(row_name, start_ns, phase_ms);
    next_launch = t0;
  }
  FILE* trace = open_trace(row_name);
  while (std::chrono::duration<double>(SteadyClock::now() - t0).count() < seconds) {
    if (hz > 0 && SteadyClock::now() < next_launch) std::this_thread::sleep_until(next_launch);
    auto t_launch = SteadyClock::now();
    latencies_ms.push_back(run_frame(stages, stream, frame_begin, frame_end, stage_ms));
    auto t_done = SteadyClock::now();
    if (trace) {
      fprintf(trace, "%zu,%lld,%lld,%.4f\n", latencies_ms.size() - 1, ns_of(t_launch), ns_of(t_done),
              latencies_ms.back());
    }
    if (hz > 0) {
      next_launch += std::chrono::duration_cast<SteadyClock::duration>(period);
      if (SteadyClock::now() > next_launch) {
        overruns++;
        next_launch = SteadyClock::now();
      }
    }
  }
  double elapsed_s = std::chrono::duration<double>(SteadyClock::now() - t0).count();
  std::vector<float> sorted = latencies_ms;
  std::sort(sorted.begin(), sorted.end());
  if (sorted.empty()) {
    fprintf(stderr, "[row_loop] %s: no frames measured\n", row_name.c_str());
    return RC_NO_FRAMES;
  }
  size_t n_frames = latencies_ms.size();
  long misses = std::count_if(latencies_ms.begin(), latencies_ms.end(),
                              [&](float ms) { return ms > deadline_ms; });
  std::string stages_part = stages_json(stages, stage_ms);
  char row_json[1400];
  snprintf(row_json, sizeof row_json,
           "{\"name\":\"%s\",\"driver\":\"row_loop_cpp\",\"status\":\"OK\",\"start_source\":\"%s\","
           "\"row_start_ns\":%lld,\"phase_ms\":%.3f,\"stream_priority\":%d,\"seconds\":%.1f,"
           "\"warmup_frames\":%d,\"n\":%zu,\"p50_ms\":%.4f,\"p99_ms\":%.4f,\"max_ms\":%.4f,"
           "\"deadline_ms\":%.3f,\"miss_frac\":%.5f,\"target_hz\":%.3f,\"achieved_hz\":%.3f,"
           "\"overrun_frac\":%.5f,\"n_engines\":%zu%s}",
           row_name.c_str(), start_source.c_str(), start_ns, phase_ms, prio, seconds, WARMUP_FRAMES, n_frames,
           percentile(sorted, 0.5), percentile(sorted, 0.99), sorted.back(), deadline_ms,
           (double)misses / n_frames, hz, n_frames / elapsed_s, (double)overruns / n_frames, stages.size(),
           stages_part.c_str());
  line = row_json;
  if (!out_path.empty()) {
    FILE* out = fopen(out_path.c_str(), "w");
    if (out) {
      fprintf(out, "%s\n", row_json);
      fclose(out);
    }
  }
  printf("%s prio=%d n=%zu p50=%.2f p99=%.2f miss=%.1f%% hz=%.1f%s\n", row_name.c_str(), prio, n_frames,
         percentile(sorted, 0.5), percentile(sorted, 0.99), 100.0 * misses / n_frames, n_frames / elapsed_s,
         stages.size() > 1 ? " (chain)" : "");
  if (trace) fclose(trace);
  for (auto& stage : stages) {
    for (void* buffer : stage.owned_buffers) cudaFree(buffer);
  }
  return 0;
}

static std::string failed_row_json(const std::vector<std::string>& fields, int rc, int prio) {
  return "{\"name\":\"" + fields[0] + "\",\"driver\":\"row_loop_cpp\",\"status\":\"FAILED\",\"rc\":" +
         std::to_string(rc) + ",\"n\":0,\"p50_ms\":null,\"p99_ms\":null,\"max_ms\":null,\"deadline_ms\":" +
         fields[3] + ",\"miss_frac\":1.0,\"target_hz\":" + fields[2] +
         ",\"achieved_hz\":0.0,\"overrun_frac\":1.0,\"stream_priority\":" + std::to_string(prio) + "}";
}

static void write_row_start_file(long long start_ns) {
  const char* trace_dir = std::getenv("ROW_TRACE_DIR");
  if (!trace_dir) return;
  std::string path = std::string(trace_dir) + "/row_start_ns.txt";
  FILE* file = fopen(path.c_str(), "w");
  if (!file) return;
  fprintf(file, "%lld\n", start_ns);
  fclose(file);
}

static int run_multi(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: row_loop --multi <seconds> <out.json> ROW...   "
                    "ROW = \"name|engine|hz|deadline_ms|prio|flag flag ...\"\n");
    return 1;
  }
  double seconds = atof(argv[2]);
  std::string out_path = argv[3];
  int lowest_prio, highest_prio;
  CK(cudaDeviceGetStreamPriorityRange(&lowest_prio, &highest_prio));
  fprintf(stderr, "stream priority range: lowest %d .. highest %d\n", lowest_prio, highest_prio);
  std::vector<std::thread> threads;
  std::vector<std::string> lines(argc - 4);
  g_multi_rows = argc - 4;
  // The barrier is in-process here; the file protocol must not interfere.
  unsetenv("ROW_START_FILE");
  unsetenv("ROW_READY_DIR");
  for (int i = 4; i < argc; i++) {
    auto fields = split(argv[i], '|');
    if (fields.size() < 5) {
      fprintf(stderr, "bad ROW: %s\n", argv[i]);
      return 1;
    }
    std::vector<std::string> flags;
    if (fields.size() > 5) flags = split(fields[5], ' ');
    int prio = atoi(fields[4].c_str());
    std::string& line = lines[i - 4];
    threads.emplace_back([=, &line]() {
      int rc = runRow(fields[1], fields[0], atof(fields[2].c_str()), atof(fields[3].c_str()), seconds, "",
                      flags, prio, line);
      if (rc == 0 && !line.empty()) return;
      g_ready++;  // a failed row must not hold the barrier
      line = failed_row_json(fields, rc, prio);
    });
  }
  for (int poll = 0; poll < BARRIER_MAX_POLLS && g_ready.load() < g_multi_rows; poll++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(BARRIER_POLL_MS));
  }
  long long start_ns = ns_of(SteadyClock::now() + std::chrono::seconds(1));
  g_start_ns.store(start_ns);
  fprintf(stderr, "[row_loop --multi] %d/%d rows warm; shared start armed (+1 s)\n", g_ready.load(),
          g_multi_rows);
  write_row_start_file(start_ns);
  for (auto& thread : threads) thread.join();
  FILE* out = fopen(out_path.c_str(), "w");
  if (!out) {
    fprintf(stderr, "cannot write %s\n", out_path.c_str());
    return 1;
  }
  fprintf(out,
          "{\"mode\":\"single_process_multistream\",\"row_start_ns\":%lld,\"stream_priority_range\":[%d,%d],"
          "\"rows\":[",
          g_start_ns.load(), lowest_prio, highest_prio);
  for (size_t k = 0; k < lines.size(); k++) fprintf(out, "%s%s", k ? "," : "", lines[k].c_str());
  fprintf(out, "]}\n");
  fclose(out);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "--prio-range") {
    int lowest_prio, highest_prio;
    CK(cudaDeviceGetStreamPriorityRange(&lowest_prio, &highest_prio));
    printf("%d %d\n", lowest_prio, highest_prio);
    return 0;
  }
  if (argc >= 2 && std::string(argv[1]) == "--multi") return run_multi(argc, argv);
  if (argc < 7) {
    fprintf(stderr,
            "usage: row_loop <engine> <name> <hz> <deadline_ms> <seconds> <out.json> [flags]   |   "
            "row_loop --multi <seconds> <out.json> ROW...   |   row_loop --prio-range\n");
    return 1;
  }
  std::vector<std::string> flags(argv + 7, argv + argc);
  std::string line;
  return runRow(argv[1], argv[2], atof(argv[3]), atof(argv[4]), atof(argv[5]), argv[6], flags, 0, line);
}
