import GPUtil
import logging
import time
import os
from datetime import datetime

# Setup Logging
os.makedirs("logs", exist_ok=True)

class MemoryProfiler:
    def __init__(self, log_file="logs/hardware_stress.log"):
        self.log_file = log_file

    def log_vram_usage(self):
        """Logs VRAM usage of all detected GPUs."""
        try:
            gpus = GPUtil.getGPUs()
            with open(self.log_file, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for gpu in gpus:
                    log_entry = f"{timestamp} | GPU: {gpu.name} | VRAM: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)\n"
                    f.write(log_entry)
            logging.info("VRAM utilization logged.")
        except Exception as e:
            logging.error(f"Hardware profiling error: {e}")

if __name__ == "__main__":
    profiler = MemoryProfiler()
    profiler.log_vram_usage()
