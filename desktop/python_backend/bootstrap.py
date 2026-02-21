"""
Nexus Shadow-Quant — Dependency Bootstrap
==========================================
Runs BEFORE api_server.py on every launch.
  1) Checks which pip packages from requirements.txt are missing
  2) Installs missing packages (with special handling for PyTorch + CUDA)
  3) Emits JSON progress for Electron splash screen
  4) Exits with code 0 on success, 1 on failure

Progress JSON format (one per line to stdout):
  {"stage":"install","progress":50,"message":"Installing torch..."}
"""

import importlib
import json
import os
import subprocess
import sys

# ── Constants ─────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REQ_FILE = os.path.join(SCRIPT_DIR, "requirements.txt")

# Map pip package names → Python import names (when they differ)
IMPORT_MAP = {
    "scikit-learn": "sklearn",
    "python-dotenv": "dotenv",
    "beautifulsoup4": "bs4",
    "Pillow": "PIL",
}

# PyTorch needs special install (CUDA wheels from pytorch.org)
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu128"


def emit(stage: str, progress: float, message: str):
    """Print JSON progress line for Electron to parse."""
    line = json.dumps({
        "stage": stage,
        "progress": round(progress, 1),
        "message": message,
    })
    print(line, flush=True)


def parse_requirements(path: str) -> list[str]:
    """Parse requirements.txt, skip comments and blanks."""
    pkgs = []
    if not os.path.exists(path):
        return pkgs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip version specifiers for the import check
            pkg = line.split(">=")[0].split("==")[0].split("<")[0].split("[")[0].strip()
            if pkg:
                pkgs.append(pkg)
    return pkgs


def can_import(pkg: str) -> bool:
    """Check if a package is importable."""
    import_name = IMPORT_MAP.get(pkg, pkg.replace("-", "_"))
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def main():
    emit("check", 0, "Checking dependencies...")

    # 1. Parse requirements
    all_pkgs = parse_requirements(REQ_FILE)
    if not all_pkgs:
        emit("done", 100, "No requirements found")
        return 0

    # 2. Find missing packages
    emit("check", 5, f"Checking {len(all_pkgs)} packages...")
    missing = []
    for pkg in all_pkgs:
        if not can_import(pkg):
            missing.append(pkg)

    if not missing:
        emit("done", 100, "All dependencies satisfied ✓")
        return 0

    if os.environ.get("NEXUS_IS_PACKAGED", "0") == "1":
        emit("warning", 95, f"Packaged mode: missing {len(missing)} packages. Proceeding with bundled environment.")
        emit("done", 100, "Proceeding.")
        return 0

    # 3. Display error and instruct manual installation (Dynamic pip install disabled - Security 1.3)
    missing_str = ', '.join(missing)
    emit("error", 100, f"Missing {len(missing)} packages: {missing_str}. Please run 'pip install -r requirements.txt'")
    return 1


if __name__ == "__main__":
    try:
        code = main()
        sys.exit(code)
    except Exception as e:
        emit("error", 0, f"Bootstrap failed: {e}")
        sys.exit(1)
