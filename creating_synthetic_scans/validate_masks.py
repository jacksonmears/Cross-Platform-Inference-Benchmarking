import os
import numpy as np

SYNTHETIC_FOLDER = "../synthetic_scans"
MASK_FOLDER = "../masks"

def validate_masks(synthetic_folder=SYNTHETIC_FOLDER, mask_folder=MASK_FOLDER):
    synthetic_files = [f for f in os.listdir(synthetic_folder) if f.endswith('.xyz')]
    synthetic_files.sort()

    for syn_file in synthetic_files:
        syn_path = os.path.join(synthetic_folder, syn_file)
        points = np.loadtxt(syn_path)
        N = points.shape[0]

        mask_file = os.path.join(mask_folder, syn_file.replace('.xyz', '_mask.npy'))
        if not os.path.exists(mask_file):
            print(f"WARNING: Mask file missing for {syn_file}")
            continue

        mask = np.load(mask_file)

        if mask.ndim != 1:
            print(f"ERROR: Mask {mask_file} is not 1D, shape={mask.shape}")
            continue
        if mask.shape[0] != N:
            print(f"ERROR: Mask {mask_file} length {mask.shape[0]} != number of points {N}")
            continue
        if not np.any(mask):
            print(f"WARNING: Mask {mask_file} has no masked points (all False)")

        print(f"{syn_file}: OK, {mask.sum()} points masked out of {N}")

if __name__ == "__main__":
    validate_masks()
