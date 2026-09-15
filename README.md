# Edge LLM Lab

A hands-on lab for **on-device / edge LLM inference** with
[llama.cpp](https://github.com/ggml-org/llama.cpp) and GGUF, on Apple Silicon.

The goal: take a model through the full edge pipeline — **build the runtime from
source → fine-tune → convert to GGUF → quantize → profile → evaluate → publish** —
and measure what actually happens on real consumer hardware (an M1 Pro).

**Published artifact:** [johnmose/qwen2.5-0.5b-extraction-gguf](https://huggingface.co/johnmose/qwen2.5-0.5b-extraction-gguf)
— a fine-tuned Qwen2.5-0.5B extraction model, quantized (Q4/Q5/Q8) for the
llama.cpp edge runtime, with the eval + tradeoff metrics below.

> Reusable reference for any edge-inference use case. Full run-by-run details in
> [`BUILD_LOG.md`](./BUILD_LOG.md).

---

## Headline result — fine-tuned model, quantized for edge, measured

Fine-tuned **Qwen2.5-0.5B** (LoRA) on a structured-extraction task (transcript →
12-field JSON), quantized it, and **evaluated on a held-out test set** (100 unseen
examples). This quantifies the core edge question: *how much quality do you lose to
run leaner and faster?*

| | FP16 | Q4_K_M | Δ |
|---|---|---|---|
| **Size** | 948 MB | 374 MB | **2.5× smaller** |
| **Gen speed** (M1, CPU) | 90 t/s | 183 t/s | **~2× faster** |
| **Field accuracy** | 0.897 | 0.871 | −2.6% |
| **F1** | 0.897 | 0.894 | −0.3% |
| Valid-JSON rate | 1.00 | 0.96 | −0.04 |
| ROUGE-L | 0.976 | 0.944 | −0.032 |

**The finding: Q4_K_M is 2.5× smaller and ~2× faster for a 2.6% field-accuracy
cost** (F1 barely moves). Quantization also made the model slightly more
conservative — precision *rose* (0.92) as recall fell. That is the "run leaner on
the edge at an acceptable quality cost" tradeoff, measured end-to-end rather than
assumed.

> Honest scope: a 0.5B / 1-epoch / CPU **proof** model — it reliably emits valid
> JSON and nails easy fields, but isn't SOTA (a purpose-built seq2seq T5 on the
> same task reaches ~0.90 field-acc / 0.98 ROUGE-L). This lab proves the *edge
> deployment pipeline and measurement discipline*, not maximum model quality.

---

## Why

On-device inference removes the cloud from the loop: lower latency, privacy by
default, no server dependency. But the edge has no safety net — the runtime has to
be fast, lean, and stable on its own. This lab is about building the muscle for
that: working with the C++ inference engine directly, quantizing models to fit
constrained hardware, and measuring the real tradeoffs.

## Pipeline

```
HF base ──LoRA finetune──▶ merged HF ──convert_hf_to_gguf──▶ FP16 GGUF
                                                                 │
                                              llama-quantize ────┤──▶ Q8_0 / Q5_K_M / Q4_K_M
                                                                 │
                              llama-bench (profile) ◀────────────┤
                              evaluate.py (held-out metrics) ◀────┘
                                                                 │
                                                    publish ─────▶ HuggingFace
```

## GPT-2 run — pipeline verification (Tier 1)

Before fine-tuning, the pipeline was proven on **GPT-2 base** (a cheap, already-
local model) to de-risk the tooling. Built llama.cpp from source, converted →
quantized → profiled (`llama-bench`, CPU, 3 runs):

| Quant | Size | Prompt t/s | Gen t/s |
|-------|------|-----------|---------|
| FP16  | 239 MiB | 1972 | 321.6 |
| Q8_0  | 129 MiB | 6150 | 511.7 |
| Q5_K_M| 92 MiB  | 3832 | 548.1 |
| Q4_K_M| 85 MiB  | 4485 | 584.5 |

Same quantization tradeoff pattern (Q4_K_M ~2.8× smaller, ~1.8× faster gen). GPT-2
base output isn't coherent, so quality wasn't measured here — that's what the
fine-tuned Qwen model above is for.

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
# 0. clone this repo
git clone https://github.com/johnmoses/edge-llm-lab.git
cd edge-llm-lab

# 1. Python env — pick ONE:

#   (a) reusable conda env (recommended: install heavy deps once, reuse across projects;
#       miniforge = arm64-native on Apple Silicon)
conda create -y -n edge-ai python=3.10
conda activate edge-ai
pip install -r requirements.txt

#   (b) portable venv (no conda needed; self-contained per project)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. build llama.cpp from source (Metal auto-enabled on Apple Silicon)
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build
cmake --build llama.cpp/build --config Release -j

# 3. fine-tune a small model (LoRA, CPU) — optional; produces models/<name>/merged
python finetune_qwen.py --data data/crusade_synthetic.jsonl \
  --base Qwen/Qwen2.5-0.5B --out models/qwen0.5b-extract --epochs 1 --max-rows 2000

# 4. convert → quantize → profile
python llama.cpp/convert_hf_to_gguf.py models/qwen0.5b-extract/merged \
  --outfile models/model-fp16.gguf --outtype f16
llama.cpp/build/bin/llama-quantize models/model-fp16.gguf models/model-Q4_K_M.gguf Q4_K_M
llama.cpp/build/bin/llama-bench -m models/model-Q4_K_M.gguf -p 128 -n 64 -r 3

# 5. evaluate on held-out test set
python evaluate.py --model models/model-Q4_K_M.gguf --data data/crusade_test.jsonl --n 100

# 6. publish quantized GGUFs + model card to HuggingFace
python upload_to_hf.py --repo <your-username>/<repo-name>
```

> Every subsequent session, re-activate the env first: `conda activate edge-ai`
> (option a) or `source .venv/bin/activate` (option b). Then the commands above use
> plain `python` / `llama-*`. The conda env is reusable across projects (install the
> heavy deps once); the venv is self-contained if you don't use conda.

## Roadmap

- [x] Build llama.cpp from source (CMake + Metal)
- [x] HF → GGUF conversion (+ converter bug fix)
- [x] Quantize Q8_0 / Q5_K_M / Q4_K_M
- [x] Profile size / throughput tradeoff
- [x] Fine-tune a small decoder-only model (Qwen2.5-0.5B, LoRA) → run through pipeline
- [x] Quality-vs-quantization comparison on the fine-tuned model (held-out eval)
- [x] Publish quantized GGUFs to HuggingFace with a model card
- [ ] Contribute a small fix/benchmark to the llama.cpp/ggml C++ layer (next)

## Environment

Apple M1 Pro · 32 GB · macOS 13.7 · AppleClang 14 · CMake 4.2 · Python 3.10

Python deps are pinned in [`requirements.txt`](./requirements.txt). Two setup
options (see [Reproduce](#reproduce)):
- **conda env** (`edge-ai`, arm64-native via miniforge) — recommended for reuse:
  the heavy deps (torch, transformers, peft, datasets) install once and are shared
  across ML projects.
- **portable venv** — no conda required, fully self-contained per project.

Both install from the same `requirements.txt`, so results are reproducible either way.

## License

MIT
