"""
push_model.py — Standalone HuggingFace upload script for the training kit.

Usage:
    set HUGGINGFACE_TOKEN=hf_xxxxx
    python push_model.py

Or create a .env file with:
    HUGGINGFACE_TOKEN=hf_xxxxx
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "Lukeed/Predictor-Models"

MODEL_DIR = Path(__file__).parent / "models"

# Model files to upload (produced by training)
MODEL_FILES = [
    "nexus_small_jamba_v1.pth",
    "mamba_scaler.pkl",
    "mamba_revin.pth",
]


def main():
    token = os.environ.get("HUGGINGFACE_TOKEN", "").strip()

    # Try .env file
    if not token:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("HUGGINGFACE_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break

    if not token:
        print("❌ HUGGINGFACE_TOKEN not found!")
        print("   Set it via environment variable or create a .env file")
        sys.exit(1)

    # Check model files exist
    files_to_upload = []
    for f in MODEL_FILES:
        fp = MODEL_DIR / f
        if fp.exists():
            files_to_upload.append(fp)
            print(f"  ✅ Found: {f} ({fp.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  ⚠️  Missing: {f} (skipping)")

    if not files_to_upload:
        print("❌ No model files found! Run training first.")
        sys.exit(1)

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, exist_ok=True, private=False)

    print(f"\n🚀 Uploading {len(files_to_upload)} files to {REPO_ID}...")
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        commit_message="BaseJamba v1 — SmallJamba trained model",
        repo_type="model",
        allow_patterns=[f.name for f in files_to_upload],
    )
    print(f"✅ Upload complete! → https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
