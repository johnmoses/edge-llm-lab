---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B
tags:
  - gguf
  - llama.cpp
  - quantized
  - edge
  - structured-extraction
  - lora
language:
  - en
pipeline_tag: text-generation
---

# Qwen2.5-0.5B — Structured Extraction (GGUF, edge / on-device)

A small **decoder-only** model fine-tuned to turn a free-text or conversational
transcript into a **structured 12-field JSON** record, then quantized to GGUF for
the [llama.cpp](https://github.com/ggml-org/llama.cpp) runtime and profiled on
Apple Silicon.

This repository is an **edge-inference demonstration**: it shows the full pipeline
— fine-tune → convert to GGUF → quantize → profile → evaluate — and the real
size/speed/quality tradeoff across quantization levels. It is a proof of the
workflow, **not** a state-of-the-art extractor.

- **Base:** `Qwen/Qwen2.5-0.5B` (~0.5B params, decoder-only)
- **Method:** LoRA (r=8, α=16, q/k/v/o proj; 0.22% trainable params), 1 epoch,
  ~2k synthetic examples, CPU training on an M1 Pro (~42 min)
- **Task:** transcript → JSON with fields: `full_name, sex, mobile_number,
  address, decision_type, prayer_need, worship_place, invited_by, age_category,
  alias, email, additional_info`
- **Data:** fully **synthetic** (no real personal data), incl. noisy/partial and
  multilingual (Yoruba/Hausa) variants

## Files (quantized variants)

This repo ships the **quantized** (edge) variants. FP16 is listed as the baseline
for reference but is not published here — this is an on-device artifact.

| File | Quant | Size | Gen speed (M1 Pro, CPU) |
|------|-------|------|--------------------------|
| _(FP16 baseline — not published)_ | FP16 | 948 MB | 90 tok/s |
| `qwen0.5b-extract-Q8_0.gguf` | Q8_0 | 506 MB | 156 tok/s |
| `qwen0.5b-extract-Q5_K_M.gguf` | Q5_K_M | 401 MB | 175 tok/s |
| `qwen0.5b-extract-Q4_K_M.gguf` | Q4_K_M | 374 MB | 183 tok/s |

**Recommended:** `Q4_K_M` — smallest and fastest, with minimal quality loss (see below).

## Evaluation (held-out test set, 100 examples)

Task-appropriate metrics for structured extraction. ROUGE-L is included only for
cross-domain comparability and is **not** the primary metric for JSON extraction.

| Metric | FP16 | Q4_K_M |
|--------|------|--------|
| valid JSON rate | 1.00 | 0.96 |
| **field accuracy** | **0.897** | **0.871** |
| exact match | 0.46 | 0.41 |
| precision | 0.896 | 0.918 |
| recall | 0.897 | 0.871 |
| F1 | 0.897 | 0.894 |
| ROUGE-L | 0.976 | 0.944 |

**Key finding:** Q4_K_M is **2.5× smaller and ~2× faster** than FP16 while losing
only **~2.6% field accuracy** (F1 −0.3%). Quantization also made the model slightly
more conservative — precision *rose* (0.918) as recall fell. This is the core
"run leaner on the edge at an acceptable quality cost" tradeoff, measured end-to-end.

## Usage (llama.cpp)

```bash
./llama-simple -m qwen0.5b-extract-Q4_K_M.gguf -n 200 -ngl 0 \
"<|im_start|>system
You extract structured decision fields from a transcript and return JSON.<|im_end|>
<|im_start|>user
extract decision fields: Grace 08029876543 Kubwa salvation invited by Pastor Musa<|im_end|>
<|im_start|>assistant
"
```

Prompt format is Qwen ChatML. The model returns a JSON object; a 0.5B model may
append noise after the closing brace, so parse the first valid JSON object and cap
generation.

## Limitations (honest)

- **Small proof model.** 0.5B, 1 epoch, CPU. Reliable on easy fields (name, phone,
  address, inferred sex); occasional field misplacement or hallucination on sparse
  fields; does not always emit a clean stop token (trailing noise after the JSON).
- **Not a quality benchmark.** A purpose-built seq2seq T5 on the same task reaches
  ~0.90 field accuracy / 0.98 ROUGE-L; this artifact demonstrates the *edge
  deployment pipeline*, not maximum quality.
- **Synthetic domain data.** Trained on synthetic transcripts; real-world
  distribution will differ.

## How it was built

Full build/quantize/profile/eval log and scripts:
https://github.com/johnmoses/edge-llm-lab
