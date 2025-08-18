import os
import numpy as np


def load_xyz_file(filepath):
    return np.loadtxt(filepath, delimiter=None)


ground_truths_folder = "../ground_truths"
synthetic_folder = "../synthetic_scans"
mask_folder = "../masks"

ground_dict = {f: load_xyz_file(os.path.join(ground_truths_folder, f)) for f in os.listdir(ground_truths_folder) if f[-1]=='z'}

ground_points_list = []
synthetic_points_list = []
mask_points_list = []

for syn_file in os.listdir(synthetic_folder):
    print(syn_file)
    base_ground_name = syn_file.split('_')[0]

    if base_ground_name in ground_dict:
        synthetic_points_list.append(load_xyz_file(os.path.join(synthetic_folder, syn_file)))
        ground_points_list.append(ground_dict[base_ground_name])

        mask_file = os.path.join(mask_folder, syn_file.replace('.xyz', '_mask.npy'))
        if os.path.exists(mask_file):
            mask_points_list.append(np.load(mask_file))
        else:
            mask_points_list.append(np.zeros(len(ground_dict[base_ground_name]), dtype=bool))
