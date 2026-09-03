// row_loop — paced TensorRT engine loop for co-location runs (no Python in the timed path).
// One process per mix row: deserializes the engine, sets shapes / loads inputs exactly as the trtexec
// flags in the mix 'extra' column, then enqueues at the target Hz for N seconds, timing each frame with
// CUDA events. Emits one JSON line (same keys the concurrent runner records).
//
//   row_loop <engine> <name> <hz> <deadline_ms> <seconds> <out.json> [trtexec-style flags...]
//   flags honoured: --staticPlugins=a.so,b.so  --shapes=t:1x3x224x224,...  --loadInputs=t:file.bin,...
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <algorithm>
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
struct L : ILogger { void log(Severity s, const char* m) noexcept override { if (s <= Severity::kERROR) fprintf(stderr, "[TRT] %s\n", m); } } gl;
#define CK(x) do { cudaError_t ce = (x); if (ce != cudaSuccess) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(ce), __LINE__); exit(2); } } while (0)
static std::vector<std::string> split(const std::string& s, char d) { std::vector<std::string> o; std::stringstream ss(s); std::string t; while (std::getline(ss, t, d)) if (!t.empty()) o.push_back(t); return o; }
static size_t esz(DataType t) { switch (t) { case DataType::kFLOAT: case DataType::kINT32: return 4; case DataType::kHALF: case DataType::kBF16: return 2; case DataType::kINT64: return 8; default: return 1; } }
// One row: engine + pacing + its own stream (optionally with a CUDA stream priority when several rows share a process).
static int runRow(std::string eng, std::string name, double hz, double dl, double secs, std::string outp, std::vector<std::string> flags, int prio, std::string& line) {
  std::map<std::string, Dims> shapes; std::map<std::string, std::string> loads;
  for (auto& a : flags) {
    if (a.rfind("--staticPlugins=", 0) == 0) { for (auto& p : split(a.substr(16), ',')) if (!dlopen(p.c_str(), RTLD_NOW | RTLD_GLOBAL)) { fprintf(stderr, "dlopen %s: %s\n", p.c_str(), dlerror()); return 2; } }
    else if (a.rfind("--shapes=", 0) == 0) { for (auto& kv : split(a.substr(9), ',')) { auto c = kv.find(':'); Dims d{}; auto ds = split(kv.substr(c + 1), 'x'); d.nbDims = (int)ds.size(); for (int k = 0; k < d.nbDims; k++) d.d[k] = atoi(ds[k].c_str()); shapes[kv.substr(0, c)] = d; } }
    else if (a.rfind("--loadInputs=", 0) == 0) { for (auto& kv : split(a.substr(13), ',')) { auto c = kv.find(':'); loads[kv.substr(0, c)] = kv.substr(c + 1); } }
  }
  initLibNvInferPlugins(&gl, "");
  std::ifstream f(eng, std::ios::binary); std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
  auto* rt = createInferRuntime(gl); auto* e = rt->deserializeCudaEngine(blob.data(), blob.size()); if (!e) { fprintf(stderr, "deserialize failed\n"); return 2; }
  auto* ctx = e->createExecutionContext(); cudaStream_t st; CK(cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, prio));
  std::vector<void*> bufs; std::mt19937 rng(0); std::uniform_real_distribution<float> U(0.f, 1.f);
  for (int i = 0; i < e->getNbIOTensors(); i++) {                       // inputs first (shapes must be set before outputs are queried)
    const char* n = e->getIOTensorName(i); if (e->getTensorIOMode(n) != TensorIOMode::kINPUT) continue;
    Dims d = shapes.count(n) ? shapes[n] : e->getTensorShape(n); for (int k = 0; k < d.nbDims; k++) if (d.d[k] < 0) d.d[k] = 1;
    if (!ctx->setInputShape(n, d)) { fprintf(stderr, "setInputShape %s failed\n", n); return 2; }
    size_t cnt = 1; for (int k = 0; k < d.nbDims; k++) cnt *= d.d[k]; DataType t = e->getTensorDataType(n); size_t bytes = cnt * esz(t);
    std::vector<char> h(bytes, 0);
    if (loads.count(n)) { std::ifstream lf(loads[n], std::ios::binary); lf.read(h.data(), bytes); }
    else if (t == DataType::kFLOAT) { float* p = (float*)h.data(); for (size_t k = 0; k < cnt; k++) p[k] = U(rng); }
    else if (t == DataType::kHALF) { uint16_t* p = (uint16_t*)h.data(); for (size_t k = 0; k < cnt; k++) { float v = U(rng); uint32_t b; memcpy(&b, &v, 4); p[k] = (uint16_t)(((b >> 16) & 0x8000) | ((((b >> 23) & 0xff) - 112) << 10) | ((b >> 13) & 0x3ff)); } }
    void* dp; CK(cudaMalloc(&dp, bytes)); CK(cudaMemcpy(dp, h.data(), bytes, cudaMemcpyHostToDevice)); bufs.push_back(dp); ctx->setTensorAddress(n, dp);
  }
  for (int i = 0; i < e->getNbIOTensors(); i++) {
    const char* n = e->getIOTensorName(i); if (e->getTensorIOMode(n) != TensorIOMode::kOUTPUT) continue;
    Dims d = ctx->getTensorShape(n); size_t cnt = 1; for (int k = 0; k < d.nbDims; k++) cnt *= (d.d[k] > 0 ? d.d[k] : 1);
    void* dp; CK(cudaMalloc(&dp, cnt * esz(e->getTensorDataType(n)))); bufs.push_back(dp); ctx->setTensorAddress(n, dp);
  }
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  auto once = [&]() { CK(cudaEventRecord(a, st)); if (!ctx->enqueueV3(st)) { fprintf(stderr, "enqueue failed\n"); exit(2); } CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms; };
  for (int i = 0; i < 20; i++) once();
  using clk = std::chrono::steady_clock; auto period = std::chrono::duration<double>(hz > 0 ? 1.0 / hz : 0.0);
  std::vector<float> lat; long overruns = 0; auto t0 = clk::now(); auto nxt = t0;
  if (const char* sn = std::getenv("ROW_START_NS")) {                                       // aligned launch grid shared by all rows
    auto ts = clk::time_point(std::chrono::nanoseconds(atoll(sn))); if (ts > clk::now()) std::this_thread::sleep_until(ts); t0 = ts; nxt = ts; }
  const char* tdir = std::getenv("ROW_TRACE_DIR"); FILE* tf = nullptr;                       // per-frame trace for period-makespan analysis
  if (tdir) { std::string tp = std::string(tdir) + "/trace_" + name + ".csv"; tf = fopen(tp.c_str(), "w"); if (tf) fprintf(tf, "frame,t_launch_ns,t_done_ns,gpu_ms\n"); }
  while (std::chrono::duration<double>(clk::now() - t0).count() < secs) {
    if (hz > 0 && clk::now() < nxt) std::this_thread::sleep_until(nxt);
    auto tl = clk::now(); lat.push_back(once()); auto td = clk::now();
    if (tf) fprintf(tf, "%zu,%lld,%lld,%.4f\n", lat.size() - 1, (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(tl.time_since_epoch()).count(), (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(td.time_since_epoch()).count(), lat.back());
    if (hz > 0) { nxt += std::chrono::duration_cast<clk::duration>(period); if (clk::now() > nxt) { overruns++; nxt = clk::now(); } }
  }
  double el = std::chrono::duration<double>(clk::now() - t0).count(); std::vector<float> s = lat; std::sort(s.begin(), s.end());
  auto pct = [&](double p) { return s[std::min(s.size() - 1, (size_t)(p * (s.size() - 1) + 0.5))]; };
  long miss = std::count_if(lat.begin(), lat.end(), [&](float v) { return v > dl; });
  char buf[512];
  snprintf(buf, sizeof buf, "{\"name\":\"%s\",\"driver\":\"row_loop_cpp\",\"stream_priority\":%d,\"n\":%zu,\"p50_ms\":%.4f,\"p99_ms\":%.4f,\"max_ms\":%.4f,\"deadline_ms\":%.3f,\"miss_frac\":%.5f,\"target_hz\":%.3f,\"achieved_hz\":%.3f,\"overrun_frac\":%.5f}",
          name.c_str(), prio, lat.size(), pct(0.5), pct(0.99), s.back(), dl, (double)miss / lat.size(), hz, lat.size() / el, (double)overruns / lat.size());
  line = buf;
  if (!outp.empty()) { FILE* o = fopen(outp.c_str(), "w"); fprintf(o, "%s\n", buf); fclose(o); }
  printf("%s prio=%d n=%zu p50=%.2f p99=%.2f miss=%.1f%% hz=%.1f\n", name.c_str(), prio, lat.size(), pct(0.5), pct(0.99), 100.0 * miss / lat.size(), lat.size() / el);
  if (tf) fclose(tf);
  for (void* p : bufs) cudaFree(p); return 0;
}
#include <mutex>
int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "--multi") {
    // row_loop --multi <seconds> <out.json> ROW ... ; ROW = "name|engine|hz|deadline_ms|prio|flag flag ..."
    // prio: 0 = default, -1..lowest = higher priority (CUDA convention: numerically lower = higher). All rows in ONE
    // CUDA context on separate streams -> kernels interleave/overlap instead of time-slicing between contexts.
    if (argc < 5) { fprintf(stderr, "usage: row_loop --multi <seconds> <out.json> ROW...\n"); return 1; }
    double secs = atof(argv[2]); std::string outp = argv[3]; int lo, hi; CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    fprintf(stderr, "stream priority range: lowest %d .. highest %d\n", lo, hi);
    std::vector<std::thread> th; std::vector<std::string> lines(argc - 4);
    if (!std::getenv("ROW_START_NS")) { auto st = std::chrono::steady_clock::now() + std::chrono::seconds(12); /* engines must load + warm up before the shared grid */ setenv("ROW_START_NS", std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(st.time_since_epoch()).count()).c_str(), 1); }
    for (int i = 4; i < argc; i++) {
      auto f = split(argv[i], '|'); if (f.size() < 5) { fprintf(stderr, "bad ROW: %s\n", argv[i]); return 1; }
      std::vector<std::string> flags; if (f.size() > 5) flags = split(f[5], ' ');
      int prio = atoi(f[4].c_str());
      th.emplace_back([=, &lines]() { runRow(f[1], f[0], atof(f[2].c_str()), atof(f[3].c_str()), secs, "", flags, prio, lines[i - 4]); });
    }
    for (auto& t : th) t.join();
    FILE* o = fopen(outp.c_str(), "w"); fprintf(o, "{\"mode\":\"single_process_multistream\",\"rows\":[");
    for (size_t k = 0; k < lines.size(); k++) fprintf(o, "%s%s", k ? "," : "", lines[k].c_str());
    fprintf(o, "]}\n"); fclose(o); return 0;
  }
  if (argc < 7) { fprintf(stderr, "usage: row_loop <engine> <name> <hz> <deadline_ms> <seconds> <out.json> [flags]   |   row_loop --multi <seconds> <out.json> ROW...\n"); return 1; }
  std::vector<std::string> flags(argv + 7, argv + argc); std::string line;
  return runRow(argv[1], argv[2], atof(argv[3]), atof(argv[4]), atof(argv[5]), argv[6], flags, 0, line);
}
