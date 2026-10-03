# Hardware: which mode on which GPU

**What these numbers are.** The tables below were measured with the **basal-1.0 weights and engine** (offline,
`basal-bench`). basal-1.5 (4.5B) has the architecture of basal-1.0-4.5B and basal-1.5-mini that of basal-1.0-1.5B
(each pair is fine-tuned from the same Bielik base model), so the tables show what each GPU does with these model
sizes and which mode to choose; they are not measurements of the 1.5 models. The *accuracy* and *agreement* columns are
those of the basal-1.0 weights on our speed sample; for the accuracy of the 1.5 models see the
[README](../README.md#performance). basal-1.5-max (11B) and Apple Silicon have their own sections below.

**basal-1.5 release check** (one H100, `basal-serve --mode fast`, bf16, HTTP load test): one question 13.7 ms p50 /
23.8 ms p95, 64.4 decisions/s with 32 concurrent clients (basal-1.0 was not re-measured on that machine).
**basal-1.5-mini, release check** (same setup): one question 10.9 ms p50 / 15.9 ms p95, 155.8 decisions/s with 32 concurrent clients (basal-1.0-1.5B was not re-measured on that machine). Do not compare these HTTP numbers with the offline tables below.

One decision = **both option orders** (the default), batch size 1 unless stated. *dec/s*: two-order decisions per
second with 32 option-order passes per forward (offline) or 32 concurrent clients (HTTP). *Agreement*: share of
decisions whose top option equals the fp32 reference on the same machine (RTX 5090 and RTX 4090: the bf16 reference,
marked ²). Measured with `basal-bench` / `basal-loadtest` (H100, RTX PRO 6000 and RTX 5090 with the equivalent research
harness) on a private 500-item sample of the Polish/English test split. Requests with several questions are much
cheaper than separate requests since basal-1.5: see [SOAM.md](SOAM.md).

## Engines

Which engine runs where, and what is verified on basal-1.5 (the per-GPU tables further down are basal-1.0 engine-only
measurements):

<!-- engines -->
| engine | `basal-serve` mode | hardware | what it is for | quick start | status (basal-1.5, 4.5B) |
|---|---|---|---|---|---|
| **basal engine** (PyTorch) | `fast` (also `fast-nocompile`, `fp8`, `nvfp4`); `eager --device mps\|cpu` | NVIDIA GPUs (sm80+); Apple Silicon (MPS) and CPU with `eager` | the primary server: packed requests (SOAM), evidence spans, `fp8` on Hopper and Blackwell | `basal-serve --model Remek/basal-1.5-4.5B --mode fast` | **verified**: served check on one H100, 13.7 ms p50 per question over HTTP, 64.4 decisions/s with 32 clients |
| **vLLM** | `vllm` | NVIDIA GPUs | an alternative CUDA server | `basal-serve --model Remek/basal-1.5-4.5B --mode vllm` (extra `basal[vllm]`) | **verified offline** (one H100, `basal-bench`, 1,000 development decisions): agreement 0.994 with the fp32 engine, accuracy 0.928 vs 0.930, 147 decisions/s |
| **SGLang** | `sglang` | NVIDIA GPUs | an alternative CUDA server (RadixAttention prefix sharing) | `basal-serve --model Remek/basal-1.5-4.5B --mode sglang` (own environment: `sglang[srt]==0.5.21`, then basal with `--no-deps`) | **verified offline** (one H100, `basal-bench`, 1,000 development decisions): agreement 0.995 with the fp32 engine, accuracy 0.925 vs 0.930, 104.6 decisions/s; on Werdykt the same answer as the basal engine on 98.3% of items (97.0% for states over 8k tokens) |
| **MLX** | `mlx` | Apple Silicon | Macs: `-MLX-8bit` (recommended) or `-MLX-fp4` (half the memory) | `basal-serve --model Remek/basal-1.5-4.5B-MLX-8bit --mode mlx` (extra `basal[mlx]`) | **verified** on an Apple M4 Pro (500 development decisions): agreement with bf16 0.992 (`-MLX-8bit`) / 0.970 (`-MLX-fp4`), 587 / 581 ms per question |
| **Ollama** | `ollama` | CPU, Apple Silicon, consumer GPUs | local and desktop use with Ollama (`-GGUF`) | 1. `hf download Remek/basal-1.5-4.5B-GGUF --local-dir basal-1.5-4.5B-GGUF` 2. in that folder: `ollama create basal-1.5-4.5b:q8_0 -f Modelfile.Q8_0` 3. `basal-serve --model ./basal-1.5-4.5B-GGUF --mode ollama --ollama-model basal-1.5-4.5b:q8_0` | **verified** on an Apple M4 Pro (500 development decisions): agreement with bf16 0.994 (`Q8_0`) / 0.980 (`Q4_K_M`), 518 / 542 ms per question |
| **llama.cpp** | `llamacpp` | CPU, Apple Silicon, consumer GPUs | `llama-server` with the `-GGUF` files; the engine's token ids are sent as is | 1. `llama-server -m basal-1.5-4.5B-GGUF/basal-1.5-4.5B-Q8_0.gguf -c 4096 --port 8080` 2. `basal-serve --model ./basal-1.5-4.5B-GGUF --mode llamacpp` | **verified** on an Apple M4 Pro (500 development decisions): agreement with bf16 0.994 (`Q8_0`) / 0.980 (`Q4_K_M`), 427 / 446 ms per question |

**`basal-serve` is the decision layer in every row.** vLLM, SGLang, MLX, Ollama and llama.cpp run the weights; the typed answers (a calibrated probability for each of your options, both option orders, the state shared by several questions, `multi`, `act` and `facts`) come from `basal-serve --mode <engine>` in front of them, with the same HTTP API in every mode. A plain `ollama run` or `llama-cli` gives free text only. Evidence spans need the PyTorch engine (`fast`, `eager`); vLLM and SGLang start with a 4,096-token context (longer states are refused); raise it with `--max-len` (e.g. 32768 for long documents). Install and first request for each engine: [quick start](../README.md#quick-start).
<!-- /engines -->

## Recommendation

| GPU class | recommended mode | why |
|---|---|---|
| Data-centre (B300, B200, H100, H200) | `fast` (bf16) | limited by kernel launches at batch 1: FP8 gains ≤ 10% on H100 and is slower on B300, bf16 keeps decisions unchanged |
| Workstation / consumer (RTX PRO 6000, RTX 5090) | `fp8` | compute-bound; FP8 gives 1.3–1.4× at 97% agreement |
| RTX 4090 | `fast` (bf16) | FP8 compilation stalled on our test machine; bf16 works (1.5B: 12.5 ms) |
| Desktop with unified memory (DGX Spark GB10) | `fp8` | memory-bandwidth-bound; FP8 halves latency |
| Batched high-throughput serving with vLLM | `vllm` + the `-FP8` checkpoint on Hopper / Ada, the `-NVFP4` checkpoint on Blackwell (RTX 50xx, RTX PRO 6000, B200, DGX Spark); see [QUANTIZATION](QUANTIZATION.md#basal-15-the-modelopt-checkpoints) | basal-1.0 NVFP4: 3.2× throughput on the DGX Spark, but −3 accuracy points; the basal-1.5 checkpoints' agreement and speed: FP8 agrees with the bf16 model on 0.979–0.989 of decisions; NVFP4 on 0.979 for max and 0.937 for basal-1.5 (−2.8 points); no NVFP4 for mini |
| A100 and other GPUs without FP8 (not measured) | `fast` | the bf16 path runs on any sm80+ GPU |
| Apple Silicon (M-series) | MLX port: `-MLX-8bit` (decisions as with bf16); `-MLX-fp4` for 8 GB Macs (about 1.4 accuracy points less on basal-1.5, 2.0 on max, 3.6 on mini: for mini prefer `-MLX-8bit` or GGUF) | see [Apple Silicon](#apple-silicon) |

With basal-1.0 models (which ship exit heads), add `--mode fast-exit` to let requests choose `"early_exit": "0.99"`
(about 1.15× faster, agreement ≥ 0.99). The basal-1.5 models do not ship exit heads; serve them with `fast`.

## Measured (4.5B architecture: basal-1.5, basal-1.0-4.5B)

| GPU | mode | 2 orders (ms) | dec/s | agreement | accuracy |
|---|---|---|---|---|---|
| **B300 SXM6** | `fast` (bf16) | **8.8** | 109 | 1.000 | 0.822 |
| | `fp8` | 9.7 | 102 | 0.974 | 0.824 |
| | `nvfp4` (torchao) | 13.9 | 63 | 0.904 | 0.790 |
| | `fast-exit`, 0.99 | **7.7** | **119** | 0.998 | 0.824 |
| | HTTP `fast` (p50) | **9.7** | 109 | – | – |
| | HTTP `fp8` (p50) | 10.4 | **119** | – | – |
| **H100 80GB** | `fast` (bf16) | 12.5 | 63 | 0.993 | 0.826 |
| | `fp8` | 11.3 | 80 | 0.976 | 0.830 |
| | `fast-exit`, 0.99 | 10.5 | 65 | 0.992 | 0.820 |
| | HTTP `fast` (p50) | 14.1 | 63 | – | – |
| | HTTP `fast-exit`, 0.99 (p50) | 12.1 | 77 | – | – |
| **RTX PRO 6000 Blackwell** | `fast` (bf16) | 18.9 | 39 | 0.994 | 0.822 |
| | `fp8` | 14.9 | 58 | 0.968 | 0.840 |
| | HTTP `fast` (p50) | 22.5 | 39 | – | – |
| **RTX 5090** | `fast` (bf16) | 27.3 (28.0 on a second machine) | 24 | 0.999² | 0.824 |
| | `fp8` | 19.4 | 40 | 0.968² | 0.828 |
| | HTTP `fast` (p50) | 32.3 | 23 | – | – |
| **DGX Spark (GB10)** | `eager` fp32 | 800.4 | 0.4 | reference | 0.822 |
| | `fast` (bf16) | 92.0 | 7.0 | 0.996 | 0.826 |
| | `fp8` | **44.6** | 8.3 | 0.970 | 0.836 |
| | `nvfp4` (torchao) | 41.0 | 10.4 | 0.910 | 0.792 |
| | `vllm` + NVFP4 checkpoint | 46.1 | **26.6** | 0.896¹ | 0.792 |
| | `fast-exit`, 0.99 | 76.9 | 7.6 | 0.994 | 0.828 |

¹ agreement with bf16 (which agrees with fp32 on 99.6%). ² agreement with bf16 eager (no fp32 run on this card).

## 1.5B architecture: basal-1.5-mini, basal-1.0-1.5B

| GPU | `fast` (bf16) | dec/s | `fp8` | dec/s | agreement bf16 / fp8 | vs 4.5B bf16 |
|---|---|---|---|---|---|---|
| B300 SXM6 | **4.7 ms** | **250** | 5.2 ms | 224 | 0.994 / 0.966 | 1.9× faster |
| H100 80GB | 6.2 ms | 157 | 6.3 ms | 177 | 0.992 / 0.964 | 2.0× |
| RTX PRO 6000 Blackwell | 8.8 ms | 101 | 7.6 ms | 128 | 0.998 / 0.968 | 2.2× |
| RTX 5090 | 12.7 ms | 67 | 9.3 ms | 99 | 0.996 / 0.960 | 2.2× |
| RTX 4090 | 12.5 ms | 51 | – | – | reference | – |
| DGX Spark (GB10) | 34.3 ms | 20 | **18.1 ms** | 31 | 0.996 / 0.962 | 2.7× |

Speed-up against the 4.5B model measured with the released engine on the same machine (RTX PRO 6000: 19.1 ms,
RTX 5090: 28.0 ms; the earlier runs above: 18.9 and 27.3 ms). Agreement with fp32 (RTX 4090: bf16 is the reference).

HTTP on H100 (`basal-serve --mode fast`): 7.7 ms p50, 147 decisions/s with 32 clients. NVFP4 on the DGX Spark: 16.5 ms
but agreement 0.87 — not recommended. RTX 4090: the FP8 compilation stalled on our test machine; bf16 works.

## 11B: basal-1.5-max

Measured on one H100; other GPUs were not measured for max.

| engine | measurement | result |
|---|---|---|
| basal engine, `fast` (bf16) | HTTP load test, one question per request, 32 concurrent clients | 27.3 ms p50 / 46.6 ms p95 for one question, 33.4 decisions/s; bf16 with SOAM agrees with the fp32 reference on 0.993 of 1,500 decisions |
| vLLM | – | not measured for max (verified on basal-1.5: agreement 0.994 with the fp32 engine) |
| SGLang (sglang 0.5.21) | Werdykt, 5,000 decisions, against the basal engine's `fast` mode | the same answer on 99.4% of decisions (98.4% for states over 8k tokens); not measured offline for max |

In-process engine, one H100, p50 per request: 30.7 ms for one question; 68 ms for 5 questions and 148 ms for 12 with
SOAM, against 201 and 422 ms as separate prompts. SOAM answers equal separate prompts in fp32 (1.000, largest probability
difference 1.5e-5). The HTTP and engine numbers are not comparable with each other. The 11B model needs about 22 GB
for the bf16 weights alone, plus activations and CUDA graphs: on 24 GB cards use the 4.5B model (`--mode fp8` may fit,
but it was not measured for max). On Apple Silicon, see the ports below.

## Apple Silicon

Two routes: the **MLX** ports (`-MLX-8bit`, `-MLX-fp4`), and the PyTorch reference path on the GPU of the Mac
(`--mode eager --device mps`, bf16 weights, no compilation); the GGUF ports run there too (Ollama, llama.cpp). The
reference is the bf16 model in PyTorch; for basal-1.5-max, which does not fit a 24 GB Mac in bf16, it is the GGUF
`Q8_0` port with llama.cpp (on basal-1.5 (4.5B), `Q8_0` agrees with bf16 on 0.994).

| machine | model | route | latency per decision | agreement with the reference | memory |
|---|---|---|---|---|---|
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | PyTorch MPS, bf16 | 817 ms | reference | 9.5 GB |
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | MLX `-MLX-8bit` | 587 ms | 0.992 | 5.2 GB |
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | MLX `-MLX-fp4` | 581 ms | 0.970 | 2.9 GB |
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | llama.cpp GGUF Q8_0 | 427 ms | 0.994 | 5.1 GB |
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | Ollama GGUF Q8_0 | 518 ms | as llama.cpp | 5.1 GB |
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | llama.cpp GGUF Q4_K_M | 446 ms | 0.980 | 2.9 GB |
| Apple M4 Pro, 24 GB | basal-1.5 (4.5B) | Ollama GGUF Q4_K_M | 542 ms | as llama.cpp | 2.9 GB |
| Apple M4 Pro, 24 GB | basal-1.5-max (11B) | llama.cpp GGUF Q8_0 | 1.16 s | reference | 11.9 GB |
| Apple M4 Pro, 24 GB | basal-1.5-max (11B) | Ollama GGUF Q8_0 | 1.10 s | as llama.cpp | 11.9 GB |
| Apple M4 Pro, 24 GB | basal-1.5-max (11B) | MLX `-MLX-8bit` | 1.36 s | 0.996 (vs GGUF Q8_0) | 12.1 GB |
| Apple M4 Pro, 24 GB | basal-1.5-max (11B) | MLX `-MLX-fp4` | 1.32 s | 0.972 (vs GGUF Q8_0) | 6.7 GB |
| Apple M4 Pro, 24 GB | basal-1.5-max (11B) | llama.cpp GGUF Q4_K_M | 1.22 s | 0.994 (vs GGUF Q8_0) | 6.8 GB |
| Apple M4 Pro, 24 GB | basal-1.5-max (11B) | Ollama GGUF Q4_K_M | 1.16 s | as llama.cpp | 6.8 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | PyTorch MPS, bf16 | 279 ms | reference | 3.2 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | MLX `-MLX-8bit` | 196 ms | 0.998 | 1.8 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | MLX `-MLX-fp4` | 197 ms | 0.908 | 1.0 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | llama.cpp GGUF Q8_0 | 160 ms | 0.998 | 1.7 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | Ollama GGUF Q8_0 | 186 ms | as llama.cpp | 1.7 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | llama.cpp GGUF Q4_K_M | 157 ms | 0.964 | 1.0 GB |
| Apple M4 Pro, 24 GB | basal-1.5-mini (1.5B) | Ollama GGUF Q4_K_M | 196 ms | as llama.cpp | 1.0 GB |

Agreement here is on a 500-decision sample, to check that each port matches its reference; to compare model sizes, see
the benchmark table ([README](../README.md#other-results)). For basal-1.5-mini use `-MLX-8bit` (1.8 GB, agreement 0.998) or GGUF `Q4_K_M` (1.0 GB, 0.964); its `-MLX-fp4` port costs about 3.6 accuracy points (agreement 0.908, accuracy 0.882 vs 0.918).

Serve with `basal-serve --model Remek/<model>-MLX-8bit --mode mlx` ([QUANTIZATION.md](QUANTIZATION.md#mlx-apple-silicon)). On Apple Silicon a decision is compute-bound (prefill): every MLX format takes about the same time, so the formats differ in memory, not speed; llama.cpp is about 25% faster than MLX on the M4 Pro and Ollama about 10% (basal-1.5). Five questions over one state take 2.1–2.2 s with MLX and 1.7–1.8 s with llama.cpp / Ollama, not five times one question, because the state is computed once.

For Ollama (GGUF), see [QUANTIZATION.md](QUANTIZATION.md#gguf-for-ollama-and-llamacpp).

## Notes per platform

- **DGX Spark (GB10, aarch64, CUDA 13).** In the README install steps use the CUDA 13 build of PyTorch
  (`uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`) instead of cu128. The GPU shares LPDDR5X memory (about 273 GB/s) with the CPU; every forward pass reads
  all weights, which is why FP8 (half the bytes) halves latency.
- **B300 / B200 (sm_100/sm_103).** PyTorch cu130 wheels work. The `vllm` mode needs a CUDA toolkit (nvcc) of version
  12.9 or newer on the machine: vLLM's FlashInfer kernels are compiled on first use, and nvcc 12.8 (found in some
  container images) cannot target the B300 (`Unsupported gpu architecture 'compute_103a'`). Compilation of `nvfp4` takes long (about 50 minutes on
  the B300) and is not recommended there.
- **RTX 5090 / RTX PRO 6000.** A driver with CUDA 12.8 is enough (`--index-url https://download.pytorch.org/whl/cu128`).
- **Start-up.** `fast` compiles and captures CUDA graphs for all input shapes before serving: 4–10 minutes the first
  time for the 4.5B model (289 s on an H100 with the weights on local disk for the 11B), much less with a warm compile cache. `fast-exit` captures one
  graph per segment and takes longer (14–22 minutes). `fast-nocompile` starts in seconds at about 1.4× the latency.
- **NVFP4 quality.** With or without calibration, 4-bit weights and activations change about 10% of decisions of this
  model and lower accuracy by about 3 points. Prefer bf16 or FP8 unless batch throughput matters more.
