import os
import glob
import uuid
import time
import torch

from runner_functions import DeviceAdapter, JSONLLogger
from training_model.config import K, NUM_POINTS
from training_model.create_graphs import create_graph_from_point_cloud
from model.GNN_autoencoder_model import GNNAutoencoder
import numpy as np
from model_inferencing.inferencing import *
from torch_geometric.data import Data
from training_model.chamfer import chamfer_distance
from training_model.config import *

# -------------------- Config --------------------
SCAN = "000001"
CHECKPOINT = "../model/gnn_autoencoder.pth"
GROUND_TRUTH = f"../original_scans/{SCAN}.xyz"
INPUT_GLOB = f"../synthetic_scans/{SCAN}.xyz_localhole_1.xyz"
# INPUT_GLOB_SUFFIX = ["noise", "globaldropout", "localhole", "occlusionplane"]
# END = "_1.xyz"
DEVICE = torch.device("cpu")  # CPU-only runner for now
OUTPUT_DIR = "inpainting"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOGS_DIR = "inference_logs"
os.makedirs(LOGS_DIR, exist_ok=True)


def run_inference_on_file(model, input_path, device):
    """Returns reconstruction path, Chamfer distance, and timings."""
    # Load & preprocess
    t0 = time.perf_counter()
    original_pts = load_points_np(input_path)
    input_pts = fixed_size_points_np(original_pts, NUM_POINTS)
    data = make_data_from_np(input_pts)
    data = data.to(device)
    t1 = time.perf_counter()

    # Inference
    with torch.no_grad():
        output_pts = model(data)
    t2 = time.perf_counter()

    # Postprocess
    out_np = output_pts.cpu().numpy()
    if out_np.ndim == 3:
        out_np = out_np[0]

    # Chamfer distance
    gt_pts = load_points_np(GROUND_TRUTH)
    gt_pts = fixed_size_points_np(gt_pts, NUM_POINTS)
    gt_tensor = torch.from_numpy(gt_pts).unsqueeze(0).to(device)
    reconstructed_tensor = torch.from_numpy(out_np).unsqueeze(0).to(device)
    cd = chamfer_distance(reconstructed_tensor, gt_tensor).item()
    bbox_size = gt_pts.max(axis=0) - gt_pts.min(axis=0)
    cd_norm = cd / np.linalg.norm(bbox_size)

    # Save reconstruction
    filename = os.path.basename(input_path)
    base_name = os.path.splitext(filename)[0]
    recon_path = os.path.join(OUTPUT_DIR, base_name + "_recon.xyz")
    save_points_np(recon_path, out_np)

    return recon_path, cd, cd_norm, (t1 - t0) * 1000, (t2 - t1) * 1000  # preprocess_ms, infer_ms



# -------------------- Main Runner --------------------
if __name__ == "__main__":
    device_adapter = DeviceAdapter()
    run_id = str(uuid.uuid4())
    run_meta = {"run_id": run_id, "device_kind": "edge_pc_cpu", **device_adapter.snapshot_static()}
    log_path = os.path.join(LOGS_DIR, f"inference_log_{run_id}.jsonl")
    logger = JSONLLogger(log_path, run_meta)

    model = load_checkpoint(CHECKPOINT, DEVICE)

    file = sorted(glob.glob(INPUT_GLOB))
    if file:
        for i, fpath in enumerate(file[:5]):
            print(f"\n---- File {i+1}: {fpath} ----")
            recon_path, cd, cd_norm, preprocess_ms, infer_ms = run_inference_on_file(model, fpath, DEVICE)

            logger.event(
                "file_inference",
                {
                    "input_file": fpath,
                    "recon_file": recon_path,
                    "chamfer_distance": cd,
                    "chamfer_distance_norm": cd_norm,
                    "preprocess_ms": preprocess_ms,
                    "infer_ms": infer_ms,
                    "system": device_adapter.current_sample()
                }
            )

        print(f"Done. Logs saved to {log_path}")


    else:
        print("No files found for pattern:", )

