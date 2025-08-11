import os
import numpy as np


def load_xyz_file(filepath):
    return np.loadtxt(filepath, delimiter=None)


original_folder = "../original-hallucinations"
synthetic_folder = "../synthetic_scans"

original_dict = {f: load_xyz_file(os.path.join(original_folder, f)) for f in os.listdir(original_folder)}

original_points_list = []
synthetic_points_list = []

for syn_file in os.listdir(synthetic_folder):
    print(syn_file)
    base_original_name = syn_file.split('_')[0]
    if base_original_name in original_dict:
        synthetic_points_list.append(load_xyz_file(os.path.join(synthetic_folder, syn_file)))
        original_points_list.append(original_dict[base_original_name])
