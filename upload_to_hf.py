"""
upload_to_hf.py — publish the quantized GGUF extraction model + model card to the
HuggingFace Hub.

Run this yourself (it pushes ~2.2GB to YOUR HF account):

    .venv/bin/python upload_to_hf.py --repo johnmose/qwen2.5-0.5b-extraction-gguf

Requires a valid HF token (already present at ~/.cache/huggingface/token, or run
`.venv/bin/huggingface-cli login`).
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


# Publish the QUANTIZED (edge) variants only — this is an edge/on-device artifact.
# FP16 stays local as the eval baseline; add it later only if a need arises.
GGUFS = [
    "models/qwen0.5b-extract-Q8_0.gguf",
    "models/qwen0.5b-extract-Q5_K_M.gguf",
    "models/qwen0.5b-extract-Q4_K_M.gguf",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="e.g. johnmoses/qwen2.5-0.5b-extraction-gguf")
    ap.add_argument("--private", action="store_true",
                    help="create as private (default: public)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would happen, upload nothing")
    args = ap.parse_args()

    api = HfApi()

    # sanity: files exist
    missing = [f for f in GGUFS if not Path(f).exists()]
    if missing:
        raise SystemExit(f"Missing GGUF files: {missing}")
    if not Path("MODEL_CARD.md").exists():
        raise SystemExit("MODEL_CARD.md not found")

    print(f"Repo: {args.repo}  (private={args.private})")
    print("Will upload:")
    print("  README.md  <- MODEL_CARD.md")
    for f in GGUFS:
        print(f"  {Path(f).name}  ({Path(f).stat().st_size // (1024*1024)} MB)")

    if args.dry_run:
        print("\n[dry-run] nothing uploaded.")
        return

    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    print("\nRepo ready. Uploading model card...")
    api.upload_file(
        path_or_fileobj="MODEL_CARD.md",
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )
    for f in GGUFS:
        print(f"Uploading {Path(f).name} ...")
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=Path(f).name,
            repo_id=args.repo,
            repo_type="model",
        )
    print(f"\nDone → https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
