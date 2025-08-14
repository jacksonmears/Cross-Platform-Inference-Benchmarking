import time
import json
import uuid
import psutil
import torch
# import pynvml

# ---------- Device Adapter for Windows PC with NVIDIA ----------
class DeviceAdapter:
    def __init__(self):
        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def snapshot_static(self):
        cpu_info = psutil.cpu_freq()
        return {
            "cpu_name": "Unknown on Windows (use wmic if needed)",
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True),
            "cpu_freq_base_mhz": cpu_info.min,
            "cpu_freq_max_mhz": cpu_info.max,
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu_name": pynvml.nvmlDeviceGetName(self.gpu_handle).decode(),
            "gpu_mem_total_gb": round(pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).total / (1024**3), 2)
        }

    def current_sample(self):
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
        power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # W

        return {
            "cpu_util": psutil.cpu_percent(interval=None),
            "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "gpu_util": util.gpu,
            "gpu_mem_used_gb": round(mem.used / (1024**3), 2),
            "gpu_temp_c": temp,
            "gpu_power_w": power
        }

# ---------- JSONL Logger ----------
class JSONLLogger:
    def __init__(self, path, run_meta):
        self.f = open(path, "a", buffering=1)
        self.run_meta = run_meta

    def event(self, name, attrs):
        rec = {"ts": time.time(), "event": name, **self.run_meta, **(attrs or {})}
        self.f.write(json.dumps(rec) + "\n")

# ---------- Example Inference Function ----------
def run_inference(model, input_tensor, device: DeviceAdapter, logger):
    # Preprocess (skip if already tensor)
    t0 = time.perf_counter()
    input_tensor = input_tensor.to("cuda")
    t1 = time.perf_counter()

    # Inference timing with CUDA events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.no_grad():
        output = model(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    infer_ms = start_event.elapsed_time(end_event)

    # Postprocess (skip for now)
    t2 = time.perf_counter()

    logger.event("inference", {
        "preprocess_ms": (t1 - t0) * 1000,
        "infer_ms": infer_ms,
        "postprocess_ms": (t2 - t1) * 1000,
        "system": device.current_sample()
    })
    return output

# ---------- Main Runner ----------
if __name__ == "__main__":
    # 1) Init device + logger
    device = DeviceAdapter()
    run_meta = {
        "run_id": str(uuid.uuid4()),
        "device_kind": "edge_pc",
        **device.snapshot_static()
    }
    logger = JSONLLogger("inference_logs.jsonl", run_meta)

    # 2) Load your model (replace this with your own)
    model = torch.load("your_model.pth", map_location="cuda")
    model.eval()

    # 3) Create dummy input (replace with real input)
    dummy_input = torch.randn(1, 3, 224, 224)  # example shape for ResNet

    # 4) Warmup
    for _ in range(5):
        _ = run_inference(model, dummy_input, device, logger)

    # 5) Measured runs
    for _ in range(10):
        _ = run_inference(model, dummy_input, device, logger)

    print("Done. Logs saved to inference_logs.jsonl")
