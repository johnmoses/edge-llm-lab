# Edge LLM Lab — Build & Profiling Log

Project: take a fine-tuned model → convert to GGUF → quantize → run on the
llama.cpp runtime → profile → publish, on Apple Silicon.
Purpose: hands-on lab for on-device / edge LLM inference — building the engine from
source, quantizing my own model, and measuring the size/latency/memory/quality
tradeoffs across quantization levels. Reusable reference for any edge-inference
use case.

**Machine:** Apple M1 Pro · 32 GB RAM · macOS · Apple clang 14 · CMake 4.2.3

---

## Step 1 — Build llama.cpp from source (Metal backend)

### Goal
Build the engine from source with CMake, confirm the Metal backend enables, run a
known model to verify the build works end to end. This is the real CMake + native
build exposure.

### Commands
```bash
# clone
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# configure (Metal is ON by default on Apple Silicon)
cmake -B build

# build (release, parallel)
cmake --build build --config Release -j
```

### Results
(to be filled in as the build runs)

- [x] Clone complete — llama.cpp @ commit `5f436dddb` (485M)
- [x] CMake configure complete — **Metal backend detected + included** ✅
- [x] Build complete — **129 binaries in `build/bin/`**, exit 0
- [x] Version / commit hash recorded: `5f436dddb`
- [x] Binary verified — `llama-cli` runs; Metal lib linked ✅

**Build result:**
- `cmake --build build --config Release -j 8` → **1:20 total** (395s user, 517% CPU / 8 cores)
- `llama-cli --version` → `0.4.0-dev (build 10948, commit 5f436dddb)`, `AppleClang 14.0.3 for Darwin arm64`
- Metal linked: `otool -L` shows `@rpath/libggml-metal.0.dylib` (v0.23.0) ✅
- Key downstream binaries present: `llama-cli`, `llama-quantize` (Step 3),
  `llama-bench` (Step 4 profiling).

**✅ STEP 1 COMPLETE** — engine built from source with Metal on M1 Pro. Real CMake +
native build exposure on the llama.cpp/ggml runtime.

Configure confirmed on M1 Pro:
- `Metal framework found -- Including METAL backend` ✅
- `Accelerate framework found` + BLAS via Accelerate.framework
- `ARM detected`, AppleClang 14.0.3, `-mcpu=native`
- Configuring done (5.1s)

### Notes / issues hit (real problems + fixes — interview talking points)
- **Issue:** first `cmake -B build` failed — `ggml-version.h.in does not exist`,
  `ggml-metal ... not an existing directory`, plus several add_subdirectory errors.
- **Diagnosis:** the earlier clone was aborted mid-checkout. Git objects downloaded
  fine (`git fsck` clean), but the working-tree files were never written — git saw
  the entire tree as staged-deleted.
- **Fix:** `git checkout -f HEAD` to restore the working tree from the intact object
  DB (no history touched), then re-configure from a clean `build/`. Resolved.
- **Takeaway:** distinguishing "repo corrupt" from "working tree not materialized"
  via `git fsck` + `git status` — object DB intact, checkout incomplete.

---

## Step 2 — Convert model to GGUF ✅

Model: `gpt2` (124M, decoder-only) — pipeline-verification model (Tier 1).
Source: local HF cache `models--gpt2` (548MB safetensors).
Command: `.venv/bin/python llama.cpp/convert_hf_to_gguf.py <gpt2> --outfile models/gpt2-fp16.gguf --outtype f16`

**Result:** `models/gpt2-fp16.gguf` — 241M, 148 tensors, FP16. Verified via CPU
inference: **prompt 511.7 t/s, generation 217.0 t/s.**

### Bug found + fixed (real llama.cpp converter defect — interview evidence)
- **Symptom:** `ValueError: Can not map tensor 'h.0.attn.bias'` during export.
- **Root cause:** `conversion/gpt2.py` intended to SKIP GPT-2's attention-mask
  buffers (`.attn.bias`, `.attn.masked_bias`) but mistakenly forwarded them to
  `super().modify_tensors(name)`, which then failed to map the non-weight tensor.
- **Fix:** replace the forward with a bare `return` (yield nothing) so the mask
  buffers are actually skipped. Conversion then succeeded.
- **Takeaway:** reading + modifying llama.cpp's Python conversion layer to fix a
  real tensor-mapping defect.

## Step 2.5 — Metal backend investigation (documented finding)

**Finding:** Metal GPU offload (`-ngl > 0`) crashes with
`GGML_ASSERT(buf_dst) failed` in `ggml_metal_cpy_tensor_async`
(`ggml-metal-context.m`). CPU path (`-ngl 0`) works perfectly.

**Systematic diagnosis (ruled out, in order):**
1. Not our model — fails identically on known-good `sqlcoder-7b-q5_k_m.gguf`.
2. Not a recent regression — fails on both b10948 (2026-09-13) and b10502 (2026-08-19).
3. Not arch/Rosetta — binary + Metal lib confirmed native `arm64`;
   shell native (`proc_translated=0`).
4. Not memory/model size — fails even at `-ngl 1` (single layer).
5. Traced to the `ggml_metal_cpy_tensor_async` (async tensor-copy) path, present
   since ≥ b9452 (June). Interacts badly with **macOS 13.7.8 (Ventura)** Metal;
   likely needs macOS 14+ (Sonoma) or a targeted backend fix.

**Decision:** proceed on the CPU path for the bridge (quantize/profile/publish all
work on CPU). Metal offload documented as an OS-interaction bug, not a blocker.
Honest note: prior `experiments/` Metal runs (LM Studio, earlier builds) DID use
Metal — so "ran models on the Metal backend" remains true; this from-source build
hit an OS-specific async-copy assertion.

## Step 3 — Quantize (Q8_0 / Q5_K_M / Q4_K_M) ✅

Tool: `llama-quantize` (built from source). Source: `gpt2-fp16.gguf`.

| Level | Size | BPW | Quantize time |
|---|---|---|---|
| FP16 (baseline) | 241M | 16 | — |
| Q8_0 | 130M | 8.67 | 0.32 s |
| Q5_K_M | 94M | 6.23 | 0.75 s |
| Q4_K_M | 87M | 5.75 | 0.78 s |

## Step 4 — Profile (size / throughput tradeoff) ✅

Tool: `llama-bench` (built from source). M1 Pro, 8 threads, pp128 (prompt) + tg64
(generation), 3 runs each. Note: bench harness reports backend `MTL,BLAS`.

| Level | Size (MiB) | Params | Prompt t/s (pp128) | Gen t/s (tg64) |
|---|---|---|---|---|
| FP16 | 239.1 | 124.4M | 1972 ± 126 | 321.6 ± 3.2 |
| Q8_0 | 128.6 | 124.4M | 6150 ± 137 | 511.7 ± 1.2 |
| Q5_K_M | 92.5 | 124.4M | 3832 ± 9 | 548.1 ± 6.1 |
| Q4_K_M | 85.3 | 124.4M | 4485 ± 23 | 584.5 ± 7.9 |

**Key insight (the edge tradeoff, quantified):**
- **Q4_K_M vs FP16: 2.8× smaller (241M→87M) AND 1.8× faster generation** (321→585 t/s).
- Quantization improves both size AND speed here — fewer bytes per weight = less
  memory-bandwidth per token. This is exactly the "run leaner + faster on edge"
  property QVAC targets.
- Sweet spot: Q4_K_M / Q5_K_M — smallest + fastest generation, minimal quality cost
  at this scale.

**⚠️ Quality note (honest):** GPT-2 base is a weak 2019 model — output is not
coherent regardless of quantization. This model verifies the PIPELINE
(convert→quantize→profile), NOT output quality. Quality comparison across levels is
covered by the fine-tuned model below (Tier 2).

---

# TIER 2 — Fine-tuned OWN model (Qwen2.5-0.5B) ✅

The GPT-2 work above proved the *pipeline*. Tier 2 runs a model **I fine-tuned
myself** through that same pipeline — the real "my own model → edge runtime" arc.

> Note: dropped from the originally-planned Gemma-2-2B to Qwen2.5-0.5B because
> **macOS 13.7 blocks torch 2.11's MPS backend** (needs macOS 14+), so training is
> CPU-only. A 0.5B model is CPU-trainable in ~40 min AND more edge-appropriate.

## T2.1 — Fine-tune (LoRA, CPU)
- Base: `Qwen/Qwen2.5-0.5B` (decoder-only). Task: transcript → 12-field JSON
  extraction (synthetic data only, reframed from a seq2seq task to decoder-only
  instruction format with completion-only loss masking).
- LoRA: r=8, alpha=16, target q/k/v/o proj → **1.08M trainable params (0.22%)**.
- Run: 2000 rows, 1 epoch, batch 4 × grad-accum 4, CPU. **~42 min** on M1 Pro.
- **Loss: 2.41 → 1.10 → 0.75 → 0.67 → 0.55** (clean monotonic decline).
- Merged LoRA → standalone HF checkpoint (`finetune_qwen.py`).

## T2.2 — Convert + Quantize
Converted merged model → GGUF (948M, 290 tensors; no converter bug — Qwen is
well-supported). Quantized:

| Level | Size | Gen t/s (tg32) | Prompt t/s (pp64) |
|---|---|---|---|
| FP16 | 942 MiB | 90.1 | 323.5 |
| Q8_0 | 501 MiB | 156.5 | 1709 |
| Q5_K_M | 395 MiB | 175.4 | 372.7 |
| Q4_K_M | 374 MiB | 183.2 | 506.4 |

**Q4_K_M vs FP16: 2.5× smaller (948M→374M) AND ~2× faster generation (90→183 t/s).**

## T2.3 — Output verification (does it actually work?)
Generation tests on held-out-style transcripts:
- **FP16:** `"Grace 08029876543 Kubwa salvation invited by Pastor Musa"` →
  valid JSON, correct name/phone/invited_by, **inferred sex=Female from name** ✅
  (one field misplacement: Kubwa → worship_place instead of address).
- **Q4_K_M** (smallest): `"...Daniel. Phone 08111222333. From Nyanya. healing"` →
  valid JSON, correct name/sex/phone/**address=Nyanya** ✅ (one hallucinated
  `invited_by`). Quantization preserved the extraction ability.

**Honest quality note:** a 0.5B / 1-epoch / CPU model — it reliably emits valid
12-field JSON and nails easy fields (name, phone, address, inferred sex), but has
occasional field misplacement / hallucination on sparse fields, and doesn't emit a
clean stop (trails into noise after the JSON; cap generation in use). This is a
PROOF of the fine-tune→quantize→edge-deploy arc, NOT a quality benchmark. The
purpose-built T5 (ROUGE-L 0.98) remains the quality reference.

## T2.3b — Evaluation on held-out test set (real metrics) ✅

Tool: `evaluate.py` on the crusade **test** split (497 rows, unseen in training;
100-row random sample per model). Metrics are task-appropriate for structured
extraction; ROUGE-L included for cross-domain comparability with the T5 models.

| Metric | FP16 | Q4_K_M | Δ (quant impact) |
|---|---|---|---|
| valid_json_rate | 1.00 | 0.96 | −0.04 |
| **field_accuracy** | **0.897** | **0.871** | **−0.026** |
| exact_match | 0.46 | 0.41 | −0.05 |
| field_precision | 0.896 | 0.918 | +0.022 |
| field_recall | 0.897 | 0.871 | −0.026 |
| field_f1 | 0.897 | 0.894 | −0.003 |
| rougeL (legacy) | 0.976 | 0.944 | −0.032 |

**Headline (the edge tradeoff, now MEASURED not asserted):**
- Q4_K_M is **2.5× smaller + ~2× faster generation** (from profiling) while losing
  only **~2.6% field accuracy** and **0.3% F1**. This is the "run leaner on edge at
  acceptable quality cost" tradeoff, quantified end-to-end.
- Nuance: Q4 precision *rose* (0.918) while recall fell — quantization made the
  model slightly more conservative (fills fewer fields, more accurate when it does).

**Honest context:** ~0.87–0.90 field accuracy for a 0.5B / 1-epoch / CPU proof
model is respectable and, correctly, BELOW the purpose-built T5 crusade model
(field_acc 0.90, ROUGE-L 0.981). This is a pipeline/edge PROOF, not a SOTA attempt.

### Eval tooling — debugging journey (real engineering evidence)
Building `evaluate.py` surfaced and fixed three real issues:
1. **Subprocess capture returned empty** — `llama-cli` interactive mode (`-st`)
   writes to the TTY, not to captured stdout/stderr pipes. Fix: switch to
   `llama-simple` (minimal one-shot generator, pipe-friendly).
2. **JSON extraction from noisy output** — the 0.5B appends garbage (incl. stray
   braces) after the JSON. Fix: `json.JSONDecoder().raw_decode()` scanning from
   each `{`, tolerating trailing data, keeping the object with the expected schema.
3. **UTF-8 decode crash** — noisy tail emitted invalid UTF-8 bytes; `subprocess`
   with `text=True` threw `UnicodeDecodeError` mid-run. Fix: `errors="replace"`.

## T2.4 — Publish GGUF + model card
(pending — HuggingFace upload)

**✅ ARC COMPLETE:** built engine from source → fine-tuned my own model →
GGUF → quantized Q4/Q5/Q8 → profiled → verified valid output on the llama.cpp
runtime. Unlocks the claim: *"fine-tuned my own small model on a structured-
extraction task, quantized it, and ran it on the llama.cpp edge runtime."*
