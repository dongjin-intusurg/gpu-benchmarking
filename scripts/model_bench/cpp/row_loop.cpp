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
// hz 0 = back-to-back (no pacing). Output: one JSON line per row (see the snprintf below).
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
struct L : ILogger { void log(Severity s, const char* m) noexcept override { if (s <= Severity::kERROR) fprintf(stderr, "[TRT] %s\n", m); } } gl;
#define CK(x) do { cudaError_t ce = (x); if (ce != cudaSuccess) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(ce), __LINE__); exit(2); } } while (0)
static std::vector<std::string> split(const std::string& s, char d) { std::vector<std::string> o; std::stringstream ss(s); std::string t; while (std::getline(ss, t, d)) if (!t.empty()) o.push_back(t); return o; }
static size_t esz(DataType t) { switch (t) { case DataType::kFLOAT: case DataType::kINT32: return 4; case DataType::kHALF: case DataType::kBF16: return 2; case DataType::kINT64: return 8; default: return 1; } }
static long long ns_of(std::chrono::steady_clock::time_point t) { return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count(); }
static std::atomic<int> g_ready{0}; static std::atomic<long long> g_start_ns{0}; static int g_multi_rows = 0;   // --multi: all rows ready -> one shared start
static const int WARMUP_FRAMES = 20;
struct Stage { std::string eng; ICudaEngine* e = nullptr; IExecutionContext* ctx = nullptr; std::map<std::string, std::pair<void*, size_t>> outs; std::vector<void*> own; cudaEvent_t ev; };

static int runRow(std::string eng, std::string name, double hz, double dl, double secs, std::string outp, std::vector<std::string> flags, int prio, std::string& line) {
  std::vector<std::string> engs; if (eng.rfind("chain:", 0) == 0) engs = split(eng.substr(6), '+'); else engs.push_back(eng);
  std::vector<std::vector<std::string>> fl(engs.size()); size_t g = 0;                                  // flags per stage (";;" separates groups)
  for (auto& a : flags) { if (a == ";;") { if (g + 1 < engs.size()) g++; continue; } fl[g].push_back(a); }
  initLibNvInferPlugins(&gl, "");
  cudaStream_t st; CK(cudaStreamCreateWithPriority(&st, cudaStreamNonBlocking, prio));
  auto* rt = createInferRuntime(gl); std::vector<Stage> S(engs.size()); std::mt19937 rng(0); std::uniform_real_distribution<float> U(0.f, 1.f);
  for (size_t k = 0; k < engs.size(); k++) {
    Stage& sg = S[k]; sg.eng = engs[k]; std::map<std::string, Dims> shapes; std::map<std::string, std::string> loads;
    for (auto& a : fl[k]) {
      if (a.rfind("--staticPlugins=", 0) == 0) { for (auto& p : split(a.substr(16), ',')) if (!dlopen(p.c_str(), RTLD_NOW | RTLD_GLOBAL)) { fprintf(stderr, "dlopen %s: %s\n", p.c_str(), dlerror()); return 2; } }
      else if (a.rfind("--shapes=", 0) == 0) { for (auto& kv : split(a.substr(9), ',')) { auto c = kv.find(':'); Dims d{}; auto ds = split(kv.substr(c + 1), 'x'); d.nbDims = (int)ds.size(); for (int q = 0; q < d.nbDims; q++) d.d[q] = atoi(ds[q].c_str()); shapes[kv.substr(0, c)] = d; } }
      else if (a.rfind("--loadInputs=", 0) == 0) { for (auto& kv : split(a.substr(13), ',')) { auto c = kv.find(':'); loads[kv.substr(0, c)] = kv.substr(c + 1); } }
      else fprintf(stderr, "[row_loop] %s: flag ignored: %s\n", name.c_str(), a.c_str());
    }
    std::ifstream f(sg.eng, std::ios::binary); if (!f) { fprintf(stderr, "engine unreadable: %s\n", sg.eng.c_str()); return 2; }
    std::vector<char> blob((std::istreambuf_iterator<char>(f)), {});
    sg.e = rt->deserializeCudaEngine(blob.data(), blob.size()); if (!sg.e) { fprintf(stderr, "deserialize failed: %s\n", sg.eng.c_str()); return 2; }
    sg.ctx = sg.e->createExecutionContext(); auto* e = sg.e; auto* ctx = sg.ctx;
    for (int i = 0; i < e->getNbIOTensors(); i++) {                       // inputs first (shapes must be set before outputs are queried)
      const char* n = e->getIOTensorName(i); if (e->getTensorIOMode(n) != TensorIOMode::kINPUT) continue;
      Dims d = shapes.count(n) ? shapes[n] : e->getTensorShape(n); for (int q = 0; q < d.nbDims; q++) if (d.d[q] < 0) d.d[q] = 1;
      if (!ctx->setInputShape(n, d)) { fprintf(stderr, "setInputShape %s failed\n", n); return 2; }
      size_t cnt = 1; for (int q = 0; q < d.nbDims; q++) cnt *= d.d[q]; DataType t = e->getTensorDataType(n); size_t bytes = cnt * esz(t);
      void* wired = nullptr;                                                // chain: reuse an earlier stage's same-named output of the same size
      for (size_t j = 0; j < k && !wired; j++) { auto it = S[j].outs.find(n); if (it != S[j].outs.end() && it->second.second == bytes) wired = it->second.first; }
      if (wired) { ctx->setTensorAddress(n, wired); fprintf(stderr, "[chain] %s <- %s (wired, %zu bytes)\n", n, S[k - 1].eng.c_str(), bytes); continue; }
      std::vector<char> h(bytes, 0);
      if (loads.count(n)) { std::ifstream lf(loads[n], std::ios::binary); if (!lf) { fprintf(stderr, "loadInputs %s: unreadable %s\n", n, loads[n].c_str()); return 2; } lf.read(h.data(), bytes); }
      else if (t == DataType::kFLOAT) { float* p = (float*)h.data(); for (size_t q = 0; q < cnt; q++) p[q] = U(rng); }
      else if (t == DataType::kHALF) { uint16_t* p = (uint16_t*)h.data(); for (size_t q = 0; q < cnt; q++) { float v = U(rng); uint32_t b; memcpy(&b, &v, 4); p[q] = (uint16_t)(((b >> 16) & 0x8000) | ((((b >> 23) & 0xff) - 112) << 10) | ((b >> 13) & 0x3ff)); } }
      void* dp; CK(cudaMalloc(&dp, bytes)); CK(cudaMemcpy(dp, h.data(), bytes, cudaMemcpyHostToDevice)); sg.own.push_back(dp); ctx->setTensorAddress(n, dp);
    }
    for (int i = 0; i < e->getNbIOTensors(); i++) {
      const char* n = e->getIOTensorName(i); if (e->getTensorIOMode(n) != TensorIOMode::kOUTPUT) continue;
      Dims d = ctx->getTensorShape(n); size_t cnt = 1; for (int q = 0; q < d.nbDims; q++) cnt *= (d.d[q] > 0 ? d.d[q] : 1); size_t bytes = cnt * esz(e->getTensorDataType(n));
      void* dp; CK(cudaMalloc(&dp, bytes)); sg.own.push_back(dp); ctx->setTensorAddress(n, dp); sg.outs[n] = {dp, bytes};
    }
    CK(cudaEventCreate(&sg.ev));
  }
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); std::vector<std::vector<float>> stage_ms(S.size());
  auto once = [&]() {                                                       // one frame = every stage back-to-back on the row's stream
    CK(cudaEventRecord(a, st));
    for (size_t k = 0; k < S.size(); k++) { if (!S[k].ctx->enqueueV3(st)) { fprintf(stderr, "enqueue failed (%s)\n", S[k].eng.c_str()); exit(2); } CK(cudaEventRecord(S[k].ev, st)); }
    CK(cudaEventRecord(b, st)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b));
    if (S.size() > 1) { cudaEvent_t prev = a; for (size_t k = 0; k < S.size(); k++) { float sm; CK(cudaEventElapsedTime(&sm, prev, S[k].ev)); stage_ms[k].push_back(sm); prev = S[k].ev; } }
    return ms; };
  for (int i = 0; i < WARMUP_FRAMES; i++) once();
  for (auto& v : stage_ms) v.clear();
  using clk = std::chrono::steady_clock; auto period = std::chrono::duration<double>(hz > 0 ? 1.0 / hz : 0.0);
  std::vector<float> lat; long overruns = 0; auto t0 = clk::now(); auto nxt = t0;
  // shared start: (a) announce warm, (b) wait for the harness's start time, or (c) the --multi in-process barrier
  long long start_ns = 0; std::string start_source = "none";
  if (const char* rd = std::getenv("ROW_READY_DIR")) { std::string rp = std::string(rd) + "/ready_" + name; FILE* rf = fopen(rp.c_str(), "w"); if (rf) fclose(rf); }
  if (const char* sf = std::getenv("ROW_START_FILE")) {
    int tmo = std::getenv("ROW_START_TIMEOUT_S") ? atoi(std::getenv("ROW_START_TIMEOUT_S")) : 600;
    for (int w = 0; w < tmo * 100; w++) { std::ifstream in(sf); long long v = 0; if (in && (in >> v) && v > 0) { start_ns = v; break; } std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
    if (!start_ns) { fprintf(stderr, "[row_loop] %s: no start file after %d s - the harness never armed the grid; row FAILED\n", name.c_str(), tmo); return 3; }
    start_source = "start_file";
  } else if (g_multi_rows > 0) {
    g_ready++; for (int w = 0; w < 60000 && g_start_ns.load() == 0; w++) std::this_thread::sleep_for(std::chrono::milliseconds(10)); start_ns = g_start_ns.load();
    if (!start_ns) { fprintf(stderr, "[row_loop] %s: in-process barrier never released; row FAILED\n", name.c_str()); return 3; }
    start_source = "multi_barrier";
  }
  double phase_ms = std::getenv("ROW_PHASE_MS") ? atof(std::getenv("ROW_PHASE_MS")) : 0.0;
  if (start_ns > 0) {
    auto ts = clk::time_point(std::chrono::nanoseconds(start_ns)) + std::chrono::microseconds((long long)(phase_ms * 1000.0));
    if (ts > clk::now()) std::this_thread::sleep_until(ts); else fprintf(stderr, "[row_loop] %s: start time already passed by %.1f ms\n", name.c_str(), std::chrono::duration<double, std::milli>(clk::now() - ts).count());
    t0 = ts; nxt = ts; }
  const char* tdir = std::getenv("ROW_TRACE_DIR"); FILE* tf = nullptr;                       // per-frame trace for the period-makespan analysis
  if (tdir) { std::string tp = std::string(tdir) + "/trace_" + name + ".csv"; tf = fopen(tp.c_str(), "w"); if (tf) fprintf(tf, "frame,t_launch_ns,t_done_ns,gpu_ms\n"); }
  while (std::chrono::duration<double>(clk::now() - t0).count() < secs) {
    if (hz > 0 && clk::now() < nxt) std::this_thread::sleep_until(nxt);
    auto tl = clk::now(); lat.push_back(once()); auto td = clk::now();
    if (tf) fprintf(tf, "%zu,%lld,%lld,%.4f\n", lat.size() - 1, ns_of(tl), ns_of(td), lat.back());
    if (hz > 0) { nxt += std::chrono::duration_cast<clk::duration>(period); if (clk::now() > nxt) { overruns++; nxt = clk::now(); } }
  }
  double el = std::chrono::duration<double>(clk::now() - t0).count(); std::vector<float> s = lat; std::sort(s.begin(), s.end());
  if (s.empty()) { fprintf(stderr, "[row_loop] %s: no frames measured\n", name.c_str()); return 4; }
  auto pct = [&](double p) { return s[std::min(s.size() - 1, (size_t)(p * (s.size() - 1) + 0.5))]; };
  long miss = std::count_if(lat.begin(), lat.end(), [&](float v) { return v > dl; });
  std::string stages_json;                                                   // chain breakdown: per-stage p50/p99 (ms), same frames
  if (S.size() > 1) { stages_json = ",\"stages\":["; for (size_t k = 0; k < S.size(); k++) { std::vector<float> v = stage_ms[k]; std::sort(v.begin(), v.end()); auto pk = [&](double p) { return v.empty() ? 0.f : v[std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5))]; };
      std::string bn = S[k].eng.substr(S[k].eng.find_last_of('/') + 1); char sb[256]; snprintf(sb, sizeof sb, "%s{\"engine\":\"%s\",\"p50_ms\":%.4f,\"p99_ms\":%.4f}", k ? "," : "", bn.c_str(), pk(0.5), pk(0.99)); stages_json += sb; } stages_json += "]"; }
  char buf[1400];
  snprintf(buf, sizeof buf, "{\"name\":\"%s\",\"driver\":\"row_loop_cpp\",\"status\":\"OK\",\"start_source\":\"%s\",\"row_start_ns\":%lld,\"phase_ms\":%.3f,\"stream_priority\":%d,\"seconds\":%.1f,\"warmup_frames\":%d,\"n\":%zu,\"p50_ms\":%.4f,\"p99_ms\":%.4f,\"max_ms\":%.4f,\"deadline_ms\":%.3f,\"miss_frac\":%.5f,\"target_hz\":%.3f,\"achieved_hz\":%.3f,\"overrun_frac\":%.5f,\"n_engines\":%zu%s}",
          name.c_str(), start_source.c_str(), start_ns, phase_ms, prio, secs, WARMUP_FRAMES, lat.size(), pct(0.5), pct(0.99), s.back(), dl, (double)miss / lat.size(), hz, lat.size() / el, (double)overruns / lat.size(), S.size(), stages_json.c_str());
  line = buf;
  if (!outp.empty()) { FILE* o = fopen(outp.c_str(), "w"); if (o) { fprintf(o, "%s\n", buf); fclose(o); } }
  printf("%s prio=%d n=%zu p50=%.2f p99=%.2f miss=%.1f%% hz=%.1f%s\n", name.c_str(), prio, lat.size(), pct(0.5), pct(0.99), 100.0 * miss / lat.size(), lat.size() / el, S.size() > 1 ? " (chain)" : "");
  if (tf) fclose(tf);
  for (auto& sg : S) for (void* p : sg.own) cudaFree(p);
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && std::string(argv[1]) == "--prio-range") { int lo, hi; CK(cudaDeviceGetStreamPriorityRange(&lo, &hi)); printf("%d %d\n", lo, hi); return 0; }
  if (argc >= 2 && std::string(argv[1]) == "--multi") {
    if (argc < 5) { fprintf(stderr, "usage: row_loop --multi <seconds> <out.json> ROW...   ROW = \"name|engine|hz|deadline_ms|prio|flag flag ...\"\n"); return 1; }
    double secs = atof(argv[2]); std::string outp = argv[3]; int lo, hi; CK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    fprintf(stderr, "stream priority range: lowest %d .. highest %d\n", lo, hi);
    std::vector<std::thread> th; std::vector<std::string> lines(argc - 4);
    g_multi_rows = argc - 4; unsetenv("ROW_START_FILE"); unsetenv("ROW_READY_DIR");        // the barrier is in-process here
    for (int i = 4; i < argc; i++) {
      auto f = split(argv[i], '|'); if (f.size() < 5) { fprintf(stderr, "bad ROW: %s\n", argv[i]); return 1; }
      std::vector<std::string> flags; if (f.size() > 5) flags = split(f[5], ' ');
      int prio = atoi(f[4].c_str());
      th.emplace_back([=, &lines]() { int rc = runRow(f[1], f[0], atof(f[2].c_str()), atof(f[3].c_str()), secs, "", flags, prio, lines[i - 4]);
        if (rc != 0 || lines[i - 4].empty()) { g_ready++;   // a failed row must not hold the barrier
          lines[i - 4] = "{\"name\":\"" + f[0] + "\",\"driver\":\"row_loop_cpp\",\"status\":\"FAILED\",\"rc\":" + std::to_string(rc) + ",\"n\":0,\"p50_ms\":null,\"p99_ms\":null,\"max_ms\":null,\"deadline_ms\":" + f[3] + ",\"miss_frac\":1.0,\"target_hz\":" + f[2] + ",\"achieved_hz\":0.0,\"overrun_frac\":1.0,\"stream_priority\":" + std::to_string(prio) + "}"; } });
    }
    for (int w = 0; w < 60000 && g_ready.load() < g_multi_rows; w++) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    long long v = ns_of(std::chrono::steady_clock::now() + std::chrono::seconds(1)); g_start_ns.store(v);
    fprintf(stderr, "[row_loop --multi] %d/%d rows warm; shared start armed (+1 s)\n", g_ready.load(), g_multi_rows);
    if (const char* rd = std::getenv("ROW_TRACE_DIR")) { std::string sp = std::string(rd) + "/row_start_ns.txt"; FILE* f = fopen(sp.c_str(), "w"); if (f) { fprintf(f, "%lld\n", v); fclose(f); } }
    for (auto& t : th) t.join();
    FILE* o = fopen(outp.c_str(), "w"); if (!o) { fprintf(stderr, "cannot write %s\n", outp.c_str()); return 1; }
    fprintf(o, "{\"mode\":\"single_process_multistream\",\"row_start_ns\":%lld,\"stream_priority_range\":[%d,%d],\"rows\":[", g_start_ns.load(), lo, hi);
    for (size_t k = 0; k < lines.size(); k++) fprintf(o, "%s%s", k ? "," : "", lines[k].c_str());
    fprintf(o, "]}\n"); fclose(o); return 0;
  }
  if (argc < 7) { fprintf(stderr, "usage: row_loop <engine> <name> <hz> <deadline_ms> <seconds> <out.json> [flags]   |   row_loop --multi <seconds> <out.json> ROW...   |   row_loop --prio-range\n"); return 1; }
  std::vector<std::string> flags(argv + 7, argv + argc); std::string line;
  return runRow(argv[1], argv[2], atof(argv[3]), atof(argv[4]), atof(argv[5]), argv[6], flags, 0, line);
}
