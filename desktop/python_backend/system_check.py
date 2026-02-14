"""
Nexus Shadow-Quant — System Check
===================================
Validates GPU, disk space, and RAM before first launch.
Returns JSON result for Electron to parse.
"""

import json
import sys
import os
import shutil


def check_system():
    """Run all system checks and return JSON result."""
    result = {
        "gpu_name": "Not detected",
        "vram_gb": 0,
        "gpu_ok": False,
        "gpu_compute": 0,
        "disk_free_gb": 0,
        "disk_ok": False,
        "ram_gb": 0,
        "ram_ok": True,  # soft requirement
        "cuda_version": "N/A",
        "errors": [],
        "warnings": [],
    }

    # ── GPU Check (soft gate — XGBoost runs on CPU) ──────
    try:
        import torch

        if not torch.cuda.is_available():
            result["warnings"].append(
                "No NVIDIA GPU detected. XGBoost predictions will run on CPU (fine). "
                "GPU is only needed for the optional Transformer/LSTM modules."
            )
            result["gpu_ok"] = True  # soft — XGBoost doesn't need GPU
        else:
            props = torch.cuda.get_device_properties(0)
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["vram_gb"] = round(props.total_memory / (1024 ** 3), 1)
            result["gpu_compute"] = float(f"{props.major}.{props.minor}")
            result["cuda_version"] = torch.version.cuda or "N/A"

            if props.major < 8:
                result["warnings"].append(
                    f"GPU {result['gpu_name']} (compute {result['gpu_compute']}) is below recommended. "
                    f"RTX 3060+ recommended for Transformer/LSTM acceleration."
                )
            elif result["vram_gb"] < 5.5:
                result["warnings"].append(
                    f"GPU {result['gpu_name']} has only {result['vram_gb']} GB VRAM. "
                    f"6 GB+ recommended for Transformer/LSTM modules."
                )

            result["gpu_ok"] = True
    except ImportError:
        result["warnings"].append("PyTorch not installed. GPU acceleration unavailable.")
        result["gpu_ok"] = True  # soft — core engine runs without GPU
    except Exception as e:
        result["warnings"].append(f"GPU check failed: {str(e)}")
        result["gpu_ok"] = True  # soft

    # ── Disk Space Check ───────────────────────────────
    try:
        # Check the drive where data will be stored
        if sys.platform == "win32":
            import platformdirs
            data_dir = platformdirs.user_data_dir("nexus-shadow-quant", "Nexus")
            drive = os.path.splitdrive(data_dir)[0] or "C:"
        else:
            drive = "/"

        usage = shutil.disk_usage(drive)
        result["disk_free_gb"] = round(usage.free / (1024 ** 3), 1)
        result["disk_ok"] = result["disk_free_gb"] >= 20

        if not result["disk_ok"]:
            result["errors"].append(
                f"Only {result['disk_free_gb']} GB free on {drive}. "
                f"At least 20 GB required."
            )
    except Exception as e:
        result["warnings"].append(f"Disk check failed: {str(e)}")
        result["disk_ok"] = True  # Don't block on failed check

    # ── RAM Check ──────────────────────────────────────
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            result["ram_gb"] = round(mem.ullTotalPhys / (1024 ** 3), 1)
        else:
            import psutil
            result["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)

        if result["ram_gb"] < 16:
            result["warnings"].append(
                f"System has {result['ram_gb']} GB RAM. "
                f"16 GB recommended for optimal performance."
            )
            result["ram_ok"] = True  # soft warning only
    except Exception as e:
        result["warnings"].append(f"RAM check failed: {str(e)}")

    return result


def main():
    """Run checks and output JSON to stdout."""
    result = check_system()
    print(json.dumps(result, indent=2))
    # Exit with error code if critical checks failed
    sys.exit(0 if not result["errors"] else 1)


if __name__ == "__main__":
    main()
