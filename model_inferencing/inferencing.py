import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)
import glob
import numpy as np
import torch
from training_model.config import K, NUM_POINTS
from training_model.create_graphs import create_graph_from_point_cloud
from model.GNN_autoencoder_model import GNNAutoencoder

CHECKPOINT = "model/gnn_autoencoder.pth"
# INPUT_GLOB = "../synthetic_scans/000006.xyz"
# INPUT_GLOB_SUFFIX = ["noise", "globaldropout", "localhole", "occlusionplane"]
# END = "_1.xyz"
# INPUT_GLOB = "C:/Users/jacks/PycharmProjects/GreenReader//test_space.xyz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = GNNAutoencoder()
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print("Loaded checkpoint: epoch", ckpt.get('epoch'), "loss", ckpt.get('loss'))
    return model

def load_points_np(path):
    pts = np.loadtxt(path, delimiter=None).astype(np.float32)
    return pts

def fixed_size_points_np(points, num_points=NUM_POINTS):
    n = points.shape[0]
    if n == num_points:
        return points.copy()
    if n > num_points:
        idx = np.random.choice(n, num_points, replace=False)
        return points[idx]
    pad = np.repeat(points[-1:], num_points - n, axis=0)
    return np.vstack([points, pad])

def make_data_from_np(points_np):
    import torch
    N = points_np.shape[0]
    k_effective = min(K, max(1, N-1))
    x = torch.from_numpy(points_np).float()
    data = create_graph_from_point_cloud(x, k=k_effective)
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    return data

def save_points_np(path, points_np):
    np.savetxt(path, points_np, fmt="%.6f")

def run_inference_on_file(model, path, device):
    pts = load_points_np(path)
    input_pts = fixed_size_points_np(pts, NUM_POINTS)
    data = make_data_from_np(input_pts)
    data = data.to(device)
    with torch.no_grad():
        out = model(data)
    out_np = out.cpu().numpy()
    if out_np.ndim == 3:
        out_np = out_np[0]

    output_dir = os.path.join("model_inferencing\inferences")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.basename(path)
    base_name = os.path.splitext(filename)[0]
    recon_path = os.path.join(output_dir, base_name + "_recon.xyz")

    save_points_np(recon_path, out_np)
    print("Saved reconstruction to:", recon_path)


if __name__ == "__main__":
    model = load_checkpoint(CHECKPOINT, DEVICE)

    # for suffix in INPUT_GLOB_SUFFIX:
    #     pattern = INPUT_GLOB+suffix+END
    #     files = sorted(glob.glob(pattern))
    #
    #     if not files:
    #         print("No files found for pattern:", pattern)
    #     for i, fpath in enumerate(files[:5]):
    #         print(f"\n---- File {i+1}: {fpath} ----")
    #         run_inference_on_file(model, fpath, DEVICE)
    #

    file_path = "model_inferencing\cleaned_scans"
    files = [file for file in os.listdir(file_path)]

    for file in files:
        full_path = os.path.join(file_path, file)
        run_inference_on_file(model, full_path, DEVICE)

    # files = sorted(glob.glob(INPUT_GLOB))
    #
    # if not files:
    #     print("No files found for pattern:", INPUT_GLOB)
    # for i, fpath in enumerate(files[:5]):
    #     print(f"\n---- File {i+1}: {fpath} ----")
    #     run_inference_on_file(model, fpath, DEVICE)
