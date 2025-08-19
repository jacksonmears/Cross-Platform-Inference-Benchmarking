import os
import numpy as np

mask_folder = "masks"
points_folder = "synthetic_scans"

def check_masks(mask_folder, points_folder):
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith("_mask.npy")]

    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        mask = np.load(mask_path)

        # Find corresponding xyz file
        xyz_file = mask_file.replace("_mask.npy", ".xyz")
        xyz_path = os.path.join(points_folder, xyz_file)
        if not os.path.exists(xyz_path):
            print(f"Warning: No matching point file for {mask_file}")
            continue

        points = np.loadtxt(xyz_path)

        # Check dimensions
        if mask.ndim != 1:
            print(f"Error: Mask {mask_file} is not 1D")
        elif len(mask) != len(points):
            print(f"Error: Mask {mask_file} length {len(mask)} does not match points {len(points)}")
        elif not np.any(mask):
            print(f"Warning: Mask {mask_file} has no masked points")
        else:
            print(f"{mask_file} OK. {mask.sum()} points masked out of {len(mask)}")

check_masks(mask_folder, points_folder)
