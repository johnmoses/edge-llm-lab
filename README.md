# Edge LLM Lab

A hands-on lab for **on-device / edge LLM inference** with
[llama.cpp](https://github.com/ggml-org/llama.cpp) and GGUF, on Apple Silicon.

The goal: take a model through the full edge pipeline — **build the runtime from
source → convert to GGUF → quantize → profile the size/latency tradeoff → publish**
— and measure what actually happens on real consumer hardware (an M1 Pro).

> Reusable reference for any edge-inference use case. Full run-by-run details in
> [`BUILD_LOG.md`](./BUILD_LOG.md).

---

## Why

On-device inference removes the cloud from the loop: lower latency, privacy by
default, no server dependency. But the edge has no safety net — the runtime has to
be fast, lean, and stable on its own. This lab is about building the muscle for
that: working with the C++ inference engine directly, quantizing models to fit
constrained hardware, and measuring the real tradeoffs.

## Pipeline

```
HF model ──convert_hf_to_gguf──▶ FP16 GGUF ──llama-quantize──▶ Q8_0 / Q5_K_M / Q4_K_M
                                                                      │
                                                          llama-bench │ profile
                                                                      ▼
                                                       size · throughput · (quality)
```

## Results so far (GPT-2, pipeline verification)

Built llama.cpp from source (CMake + Metal), converted GPT-2 → GGUF, quantized, and
profiled on an M1 Pro (`llama-bench`, CPU path, 3 runs):

| Quant | Size | Prompt t/s | Gen t/s |
|-------|------|-----------|---------|
| FP16  | 239 MiB | 1972 | 321.6 |
| Q8_0  | 129 MiB | 6150 | 511.7 |
| Q5_K_M| 92 MiB  | 3832 | 548.1 |
| Q4_K_M| 85 MiB  | 4485 | 584.5 |

**Takeaway:** Q4_K_M is **2.8× smaller and 1.8× faster at generation** than FP16 —
fewer bytes per weight means less memory bandwidth per token. Exactly the "run
leaner + faster on edge" property that matters on-device.

> GPT-2 base is a weak 2019 model used here only to verify the *pipeline*. A
> fine-tuned small model (quality comparison across quant levels) is the next step.

## Notable findings

- **Converter bug fix:** llama.cpp's GPT-2 conversion handler forwarded the model's
  attention-mask buffers (`.attn.bias`) to the tensor mapper instead of skipping
  them, causing `ValueError: Can not map tensor 'h.0.attn.bias'`. Fixed by skipping
  those non-weight buffers.
- **Metal backend diagnosis:** GPU offload (`-ngl > 0`) aborts with
  `GGML_ASSERT(buf_dst)` in `ggml_metal_cpy_tensor_async`. Isolated to the
  async-copy path interacting with macOS 13.7 (Ventura) — reproduces on known-good
  models and across two releases, on native arm64, even at `-ngl 1`. CPU path works.
  See `BUILD_LOG.md` for the full elimination trail.

## Reproduce

```bash
# 1. build llama.cpp from source (Metal auto-enabled on Apple Silicon)
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build --config Release -j

# 2. python env for conversion
python3 -m venv .venv && ./.venv/bin/pip install \
  -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# 3. convert → quantize → profile
./.venv/bin/python llama.cpp/convert_hf_to_gguf.py <hf-model-dir> \
  --outfile models/model-fp16.gguf --outtype f16
llama.cpp/build/bin/llama-quantize models/model-fp16.gguf models/model-Q4_K_M.gguf Q4_K_M
llama.cpp/build/bin/llama-bench -m models/model-Q4_K_M.gguf -p 128 -n 64 -r 3
```

## Roadmap

- [x] Build llama.cpp from source (CMake + Metal)
- [x] HF → GGUF conversion (+ converter bug fix)
- [x] Quantize Q8_0 / Q5_K_M / Q4_K_M
- [x] Profile size / throughput tradeoff
- [ ] Fine-tune a small decoder-only model (Gemma-2-2B, LoRA) → run through pipeline
- [ ] Publish quantized GGUFs to HuggingFace with a model card
- [ ] Quality-vs-quantization comparison on the fine-tuned model

## Environment

Apple M1 Pro · 32 GB · macOS 13.7 · AppleClang 14 · CMake 4.2 · Python 3.10

## License

MIT
