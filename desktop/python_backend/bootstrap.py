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


def install_package(pkg: str, use_pytorch_index: bool = False) -> bool:
    """Install a single package via pip."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", pkg]
    if use_pytorch_index:
        cmd.extend(["--index-url", PYTORCH_INDEX])
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for large packages
        )
        return result.returncode == 0
    except Exception as e:
        emit("error", 0, f"Failed to install {pkg}: {e}")
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
        emit("warning", 95, f"Packaged mode: skipping runtime installation of {len(missing)} packages.")
        emit("done", 100, "Proceeding with bundled environment.")
        return 0

    emit("install", 10, f"Need to install {len(missing)} packages: {', '.join(missing)}")

    # 3. Ensure pip is available and up to date
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, timeout=10,
        )
    except Exception:
        emit("error", 0, "pip is not available. Please install Python with pip.")
        return 1

    # 4. Install missing packages with progress
    total = len(missing)
    for i, pkg in enumerate(missing):
        pct = 10 + (i / total) * 85  # 10% to 95%
        
        # PyTorch special handling: install with CUDA index
        is_torch = pkg in ("torch", "torchvision", "torchaudio")
        
        if is_torch:
            emit("install", pct, f"⬇ Installing {pkg} with CUDA support (this may take a while)...")
        else:
            emit("install", pct, f"⬇ Installing {pkg}... ({i + 1}/{total})")

        ok = install_package(pkg, use_pytorch_index=is_torch)
        if not ok:
            # Retry without special index for torch
            if is_torch:
                emit("install", pct, f"⬇ Retrying {pkg} (standard PyPI)...")
                ok = install_package(pkg, use_pytorch_index=False)
            if not ok:
                emit("error", pct, f"Failed to install {pkg}")
                # Continue anyway — some packages may be optional
                continue

    # 5. Final verification
    emit("check", 95, "Verifying installation...")
    still_missing = [pkg for pkg in missing if not can_import(pkg)]
    
    if still_missing:
        emit("warning", 98, f"Could not install: {', '.join(still_missing)}")
        # Don't block — the app may still partially work
    
    emit("done", 100, f"Dependencies ready ✓ (installed {total - len(still_missing)}/{total})")
    return 0


if __name__ == "__main__":
    try:
        code = main()
        sys.exit(code)
    except Exception as e:
        emit("error", 0, f"Bootstrap failed: {e}")
        sys.exit(1)
