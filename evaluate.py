"""
evaluate.py — evaluate a GGUF extraction model on the held-out test set.

Runs the model via the llama.cpp CLI (built from source), parses each JSON
output, and reports task-appropriate metrics:

  - valid_json_rate : fraction of outputs that parse as JSON
  - field_accuracy  : of gold non-null fields, fraction predicted exactly right
  - exact_match     : fraction of examples where ALL gold fields match
  - precision/recall/f1 (non-null fields): did it fill fields it should (recall)
                       without inventing fields that should be null (precision)
  - rougeL          : text-overlap vs gold JSON string (LEGACY / cross-domain
                       comparability with the T5 models — NOT the headline metric
                       for structured extraction)

Usage:
    .venv/bin/python evaluate.py \
        --model models/qwen0.5b-extract-Q4_K_M.gguf \
        --data data/crusade_test.jsonl \
        --n 100

Compare two variants (e.g. FP16 vs Q4) by running twice and diffing the JSON reports.
"""
import argparse
import json
import random
import re
import subprocess
from pathlib import Path

from rouge_score import rouge_scorer

CLI = "llama.cpp/build/bin/llama-simple"
SYSTEM = "You extract structured decision fields from a transcript and return JSON."
FIELDS = [
    "full_name", "sex", "mobile_number", "address", "decision_type",
    "prayer_need", "worship_place", "invited_by", "age_category",
    "alias", "email", "additional_info",
]


def build_prompt(user_input: str) -> str:
    # Qwen chatml format, matching training
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_model(model: str, prompt: str, n_predict: int = 220) -> str:
    """Run one generation via llama-cli, return ONLY the assistant completion.

    llama-cli echoes the prompt + a banner in interactive mode, so we split on the
    final assistant marker and keep what follows. n_predict is generous so the full
    12-field JSON completes (short budgets truncate the closing brace)."""
    # llama-simple: minimal one-shot generator, pipe-friendly (llama-cli's
    # interactive mode writes to the TTY and isn't captured by subprocess).
    # It echoes the prompt then the completion; extract_json finds the model's
    # JSON (prompts contain no braces). -ngl 0 avoids the Metal async-copy crash.
    p = subprocess.run(
        [CLI, "-m", model, "-n", str(n_predict), "-ngl", "0", prompt],
        capture_output=True, text=True, timeout=180,
        errors="replace",  # small models can emit invalid UTF-8 bytes in the noise tail
    )
    return p.stdout or ""


def extract_json(text: str):
    """Extract the model's JSON object from noisy output.

    A 0.5B model appends garbage after the closing brace (sometimes with stray
    braces). Strategy: for every '{' start position, greedily try to parse the
    longest-then-shorter substring ending at a '}' via a JSON decoder that tolerates
    trailing data. Return the first object that both parses AND looks like our
    schema (has 'full_name'). Robust against unbalanced trailing braces."""
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        start = m.start()
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if isinstance(obj, dict) and "full_name" in obj:
            return obj
    return None


def evaluate(model: str, data: Path, n: int, seed: int = 42):
    rows = [json.loads(l) for l in open(data) if l.strip()]
    random.seed(seed)
    if n and n < len(rows):
        rows = random.sample(rows, n)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    valid = 0
    exact = 0
    field_hits = 0
    field_total = 0          # gold non-null fields (recall denom)
    pred_nonnull = 0         # predicted non-null fields (precision denom)
    pred_nonnull_correct = 0 # predicted non-null AND correct
    rouge_l = []

    for i, r in enumerate(rows):
        out = run_model(model, build_prompt(r["input"]))
        pred = extract_json(out)
        gold = json.loads(r["target"])

        # rouge-L on the raw JSON strings (legacy comparability)
        pred_str = json.dumps(pred, ensure_ascii=False) if pred else out.strip()[:400]
        rouge_l.append(scorer.score(r["target"], pred_str)["rougeL"].fmeasure)

        if pred is None:
            field_total += sum(1 for f in FIELDS if gold.get(f) is not None)
            print(f"  [{i+1}/{len(rows)}] INVALID JSON")
            continue
        valid += 1

        all_match = True
        for f in FIELDS:
            g, p = gold.get(f), pred.get(f)
            if g is not None:
                field_total += 1
                if p == g:
                    field_hits += 1
                else:
                    all_match = False
            if p is not None and str(p).strip() != "":
                pred_nonnull += 1
                if p == g:
                    pred_nonnull_correct += 1
        if all_match:
            exact += 1
        print(f"  [{i+1}/{len(rows)}] ok")

    n_rows = len(rows)
    precision = pred_nonnull_correct / pred_nonnull if pred_nonnull else 0.0
    recall = field_hits / field_total if field_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    report = {
        "model": model,
        "n_examples": n_rows,
        "valid_json_rate": round(valid / n_rows, 4),
        "field_accuracy": round(field_hits / field_total, 4) if field_total else 0.0,
        "exact_match": round(exact / n_rows, 4),
        "field_precision": round(precision, 4),
        "field_recall": round(recall, 4),
        "field_f1": round(f1, 4),
        "rougeL": round(sum(rouge_l) / n_rows, 4),
        "_note": "rougeL is legacy/cross-domain comparability only; field_accuracy "
                 "is the primary metric for structured extraction.",
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/crusade_test.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=None, help="write JSON report to this path")
    args = ap.parse_args()

    print(f"Evaluating {args.model} on {args.n} examples from {args.data}\n")
    report = evaluate(args.model, Path(args.data), args.n)

    print("\n=== REPORT ===")
    print(json.dumps(report, indent=2))

    out = args.out or f"eval-{Path(args.model).stem}.json"
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
