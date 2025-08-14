import time
import json
import psutil

class DeviceAdapter:
    """Captures CPU and RAM info (CPU-only runner)."""
    def snapshot_static(self):
        cpu_info = psutil.cpu_freq()
        return {
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "cpu_freq_base_mhz": cpu_info.min,
            "cpu_freq_max_mhz": cpu_info.max,
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu_name": None,
            "gpu_mem_total_gb": None
        }

    def current_sample(self):
        return {
            "cpu_util": psutil.cpu_percent(interval=None),
            "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "gpu_util": None,
            "gpu_mem_used_gb": None,
            "gpu_temp_c": None,
            "gpu_power_w": None
        }

class JSONLLogger:
    """Simple JSONL logger."""
    def __init__(self, path, run_meta):
        self.f = open(path, "a", buffering=1)
        self.run_meta = run_meta

    def event(self, name, attrs=None):
        rec = {"ts": time.time(), "event": name, **self.run_meta, **(attrs or {})}
        self.f.write(json.dumps(rec) + "\n")
