import os
import numpy as np
import sys

ground_folder = "ground_truths_test"
synthetic_folder = "synthetic_scans_test"
mask_folder = "masks_test"

def load_xyz_file(filepath):
    return np.loadtxt(filepath, delimiter=None)

def build_pointcloud_lists():
    ground_dict = {os.path.splitext(f)[0]: load_xyz_file(os.path.join(ground_folder, f))
                   for f in os.listdir(ground_folder) if f.endswith('.xyz')}

    synthetic_files = sorted([f for f in os.listdir(synthetic_folder) if f.endswith('.xyz')])
    ground_points_list, synthetic_points_list, mask_points_list = [], [], []

    for syn_file in synthetic_files:
        base_name = syn_file.split('_')[0]
        syn_path = os.path.join(synthetic_folder, syn_file)
        syn_pts = load_xyz_file(syn_path)
        synthetic_points_list.append(syn_pts)

        gt_pts = ground_dict.get(base_name, syn_pts.copy())
        ground_points_list.append(gt_pts)

        mask_file = os.path.join(mask_folder, syn_file.replace('.xyz', '_mask.npy'))
        if os.path.exists(mask_file):
            mask = np.load(mask_file)
            if mask.ndim != 1 or mask.shape[0] != syn_pts.shape[0]:
                # fallback: zero mask aligned to synthetic points
                mask = np.zeros(syn_pts.shape[0], dtype=bool)
        else:
            mask = np.zeros(syn_pts.shape[0], dtype=bool)
        mask_points_list.append(mask.astype(bool))
        

    assert len(ground_points_list) == len(synthetic_points_list) == len(mask_points_list)
    return ground_points_list, synthetic_points_list, mask_points_list
