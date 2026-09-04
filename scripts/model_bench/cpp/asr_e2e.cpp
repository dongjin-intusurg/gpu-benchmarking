// Encoder-decoder ASR model end-to-end in C++ (TensorRT runtime API): encoder -> first decoder step -> greedy
// decoder-with-past loop. CUDA-event timing per stage; token ids written for the (offline) WER step.
// Build: g++ -O2 -std=c++17 asr_e2e.cpp -I/usr/include/$(gcc -dumpmachine) -I/usr/local/cuda/include \
//        -L/usr/local/cuda/lib64 -lnvinfer -lcudart -o asr_e2e
// Usage: ./asr_e2e <engines_dir> <prec: fp16|fp32_ref> <mel list.txt> <out.jsonl> [repeats=3] [realtime=0]
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <cstring>
using namespace nvinfer1;
struct Logger : public ILogger { void log(Severity s, const char* m) noexcept override { if (s <= Severity::kERROR) std::cerr << m << "\n"; } } gLogger;
static size_t dsize(DataType t){ switch(t){ case DataType::kFLOAT: return 4; case DataType::kHALF: return 2; case DataType::kINT32: return 4; case DataType::kINT64: return 8; case DataType::kINT8: return 1; case DataType::kBOOL: return 1; default: return 4; } }
static size_t vol(const Dims& d){ size_t v=1; for(int i=0;i<d.nbDims;i++) v*= (size_t)std::max(1, (int)d.d[i]); return v; }
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ std::cerr<<"CUDA "<<cudaGetErrorString(e)<<" @"<<__LINE__<<"\n"; exit(1);} }while(0)
struct Eng {
  ICudaEngine* eng=nullptr; IExecutionContext* ctx=nullptr; std::vector<std::string> in, out; std::map<std::string, DataType> dt;
  void load(IRuntime* rt, const std::string& path){
    std::ifstream f(path, std::ios::binary); std::vector<char> b((std::istreambuf_iterator<char>(f)), {});
    eng = rt->deserializeCudaEngine(b.data(), b.size()); if(!eng){ std::cerr<<"deserialize failed "<<path<<"\n"; exit(1);} ctx = eng->createExecutionContext();
    for(int i=0;i<eng->getNbIOTensors();i++){ const char* n=eng->getIOTensorName(i); dt[n]=eng->getTensorDataType(n); (eng->getTensorIOMode(n)==TensorIOMode::kINPUT? in: out).push_back(n); }
  }
};
static void* dalloc(size_t bytes){ void* p; CK(cudaMalloc(&p, bytes)); return p; }
int main(int argc, char** argv){
  if(argc<5){ std::cerr<<"usage: engines_dir prec mel_list out.jsonl [repeats] [realtime]\n"; return 1; }
  std::string ED=argv[1], P=argv[2], LIST=argv[3], OUT=argv[4]; int REP = argc>5? atoi(argv[5]):3; int RT = argc>6? atoi(argv[6]):0;
  const int L=32, H=20, DH=64, T=1500, VOCAB=51866, MAXLEN=448, EOS=50257; const std::vector<int64_t> PROMPT={50258,50259,50360,50364};
  IRuntime* rt = createInferRuntime(gLogger); Eng enc, dec1, decp;
  enc.load(rt, ED+"/encoder_"+P+".engine"); dec1.load(rt, ED+"/decoder_first_"+P+".engine"); decp.load(rt, ED+"/decoder_past_"+P+".engine");
  cudaStream_t st; CK(cudaStreamCreate(&st));
  // buffers (max sizes)
  void* mel = dalloc(128*3000*dsize(enc.dt["input_features"])); void* hid = dalloc((size_t)T*1280*dsize(enc.dt["last_hidden_state"]));
  void* ids = dalloc(8*8); void* logits1 = dalloc((size_t)8*VOCAB*dsize(dec1.dt["logits"])); void* logitsp = dalloc((size_t)VOCAB*dsize(decp.dt["logits"]));
  std::map<std::string, void*> encKV; std::map<std::string, void*> decA, decB;   // cross KV (fixed) ; self KV ping-pong
  for(int i=0;i<L;i++) for(std::string kv: {"key","value"}){
    encKV["present."+std::to_string(i)+".encoder."+kv] = dalloc((size_t)H*T*DH*4);
    decA["present."+std::to_string(i)+".decoder."+kv] = dalloc((size_t)H*MAXLEN*DH*4); decB["present."+std::to_string(i)+".decoder."+kv] = dalloc((size_t)H*MAXLEN*DH*4); }
  std::vector<float> hlog(VOCAB); std::vector<float> hlog1((size_t)8*VOCAB); float* pin; CK(cudaMallocHost((void**)&pin, (size_t)8*VOCAB*4));
  // read mel list
  std::ifstream lf(LIST); std::ofstream of(OUT); std::string line; int nclips=0;
  std::vector<std::tuple<int,double,std::string>> clips; while(std::getline(lf,line)){ std::istringstream ss(line); int id; double sec; std::string p; ss>>id>>sec>>p; clips.push_back({id,sec,p}); }
  cudaEvent_t e0,e1,e2,ea,eb; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&e2)); CK(cudaEventCreate(&ea)); CK(cudaEventCreate(&eb));
  auto run_clip=[&](const std::string& melpath, double sec, int id, bool print){
    std::vector<float> hm(128*3000); { std::ifstream mf(melpath, std::ios::binary); mf.read((char*)hm.data(), hm.size()*4); }
    if(enc.dt["input_features"]==DataType::kHALF){ std::vector<__half> hh(hm.size()); for(size_t i=0;i<hm.size();i++) hh[i]=__float2half(hm[i]); CK(cudaMemcpyAsync(mel, hh.data(), hh.size()*2, cudaMemcpyHostToDevice, st)); }
    else CK(cudaMemcpyAsync(mel, hm.data(), hm.size()*4, cudaMemcpyHostToDevice, st));
    auto w0=std::chrono::steady_clock::now();
    // encoder
    enc.ctx->setInputShape("input_features", Dims3(1,128,3000)); enc.ctx->setTensorAddress("input_features", mel); enc.ctx->setTensorAddress("last_hidden_state", hid);
    CK(cudaEventRecord(e0, st)); enc.ctx->enqueueV3(st); CK(cudaEventRecord(e1, st));
    // first step
    CK(cudaMemcpyAsync(ids, PROMPT.data(), PROMPT.size()*8, cudaMemcpyHostToDevice, st));
    dec1.ctx->setInputShape("input_ids", Dims2(1,(int)PROMPT.size())); dec1.ctx->setInputShape("encoder_hidden_states", Dims3(1,T,1280));
    dec1.ctx->setTensorAddress("input_ids", ids); dec1.ctx->setTensorAddress("encoder_hidden_states", hid); dec1.ctx->setTensorAddress("logits", logits1);
    for(auto& n: dec1.out){ if(n=="logits") continue; if(n.find(".encoder.")!=std::string::npos) dec1.ctx->setTensorAddress(n.c_str(), encKV[n]); else dec1.ctx->setTensorAddress(n.c_str(), decA[n]); }
    dec1.ctx->enqueueV3(st); CK(cudaEventRecord(e2, st));
    CK(cudaMemcpyAsync(pin, (char*)logits1 + (size_t)(PROMPT.size()-1)*VOCAB*4, VOCAB*4, cudaMemcpyDeviceToHost, st)); CK(cudaStreamSynchronize(st));
    float tenc, tfirst; CK(cudaEventElapsedTime(&tenc, e0, e1)); CK(cudaEventElapsedTime(&tfirst, e1, e2));
    int nxt = (int)(std::max_element(pin, pin+VOCAB)-pin); std::vector<int> outtok{nxt}; std::vector<float> dec_ms; int past=(int)PROMPT.size();
    std::map<std::string,void*>* cur=&decA; std::map<std::string,void*>* nxtb=&decB;
    while(nxt!=EOS && past<MAXLEN-1){
      int64_t t=nxt; CK(cudaMemcpyAsync(ids, &t, 8, cudaMemcpyHostToDevice, st));
      decp.ctx->setInputShape("input_ids", Dims2(1,1)); decp.ctx->setTensorAddress("input_ids", ids); decp.ctx->setTensorAddress("logits", logitsp);
      for(int i=0;i<L;i++) for(std::string kv: {"key","value"}){
        std::string pk="past_key_values."+std::to_string(i)+".decoder."+kv, ek="past_key_values."+std::to_string(i)+".encoder."+kv, pr="present."+std::to_string(i)+".decoder."+kv;
        decp.ctx->setInputShape(pk.c_str(), Dims4(1,H,past,DH)); decp.ctx->setTensorAddress(pk.c_str(), (*cur)[pr]);
        decp.ctx->setInputShape(ek.c_str(), Dims4(1,H,T,DH)); decp.ctx->setTensorAddress(ek.c_str(), encKV["present."+std::to_string(i)+".encoder."+kv]);
        decp.ctx->setTensorAddress(pr.c_str(), (*nxtb)[pr]); }
      CK(cudaEventRecord(ea, st)); decp.ctx->enqueueV3(st); CK(cudaEventRecord(eb, st));
      CK(cudaMemcpyAsync(pin, logitsp, VOCAB*4, cudaMemcpyDeviceToHost, st)); CK(cudaStreamSynchronize(st));
      float td; CK(cudaEventElapsedTime(&td, ea, eb)); dec_ms.push_back(td);
      nxt=(int)(std::max_element(pin, pin+VOCAB)-pin); outtok.push_back(nxt); past++; std::swap(cur, nxtb);
    }
    double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-w0).count();
    std::vector<float> s=dec_ms; std::sort(s.begin(), s.end()); float med = s.empty()?0:s[s.size()/2]; float p99 = s.empty()?0:s[std::min(s.size()-1,(size_t)(0.99*s.size()))];
    if(print){ long long w0ns=std::chrono::duration_cast<std::chrono::nanoseconds>(w0.time_since_epoch()).count(); of<<"{\"clip\":"<<id<<",\"t_start_ns\":"<<w0ns<<",\"seconds\":"<<sec<<",\"encoder_ms\":"<<tenc<<",\"first_step_ms\":"<<tfirst<<",\"ttft_ms\":"<<tenc+tfirst<<",\"decode_ms_median\":"<<med<<",\"decode_ms_p99\":"<<p99<<",\"decode_ms_max\":"<<(s.empty()?0:s.back())<<",\"n_tokens\":"<<outtok.size()<<",\"gpu_total_ms\":"<<tenc+tfirst+std::accumulate(dec_ms.begin(),dec_ms.end(),0.0f)<<",\"wall_ms\":"<<wall<<",\"rtf_wall\":"<<wall/1000.0/sec<<",\"tokens\":[";
      for(size_t i=0;i<outtok.size();i++) of<<(i?",":"")<<outtok[i]; of<<"]}\n"; of.flush();
      std::cout<<"clip "<<id<<" "<<sec<<"s enc "<<tenc<<" first "<<tfirst<<" dec med "<<med<<" p99 "<<p99<<" tok "<<outtok.size()<<" wall "<<wall<<" ms RTF "<<wall/1000.0/sec<<"\n"; }
    if(RT){ double sleep_ms = sec*1000.0 - wall; if(sleep_ms>0) std::this_thread::sleep_for(std::chrono::milliseconds((long)sleep_ms)); }
  };
  auto [id0,sec0,p0]=clips[0]; run_clip(p0,sec0,id0,false); run_clip(p0,sec0,id0,false);   // warm-up
  for(int r=0;r<REP;r++) for(auto& [id,sec,p]: clips) run_clip(p,sec,id,true);
  std::cout<<"DONE\n"; return 0;
}
