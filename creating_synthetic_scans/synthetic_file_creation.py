import os
import numpy as np
from add_noise import add_noise
from occlusion_plane import occlusion_plane
from random_global_dropout import random_global_dropout
from random_local_hole import random_local_hole
from save_functions import save_points_as_xyz, save_mask


def generate_synthetic_scans(ground_truth_points: np.ndarray, base_filename: str, corrupt_dir: str, mask_dir: str, number_files: int = 3):
    os.makedirs(corrupt_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # local holes
    for i in range(number_files):
        corrupted, mask = random_local_hole(ground_truth_points, num_holes=3)
        fname = f"{base_filename}_localhole_{i+1}.xyz"
        mname = f"{base_filename}_localhole_{i+1}_mask.npy"
        save_points_as_xyz(corrupted, os.path.join(corrupt_dir, fname))
        save_mask(mask, os.path.join(mask_dir, mname))

    # global dropout
    for i in range(number_files):
        dropout_ratio = float(np.random.uniform(0.05, 0.2))
        corrupted, mask = random_global_dropout(ground_truth_points, dropout_ratio=dropout_ratio)
        fname = f"{base_filename}_globaldropout_{i+1}.xyz"
        mname = f"{base_filename}_globaldropout_{i+1}_mask.npy"
        save_points_as_xyz(corrupted, os.path.join(corrupt_dir, fname))
        save_mask(mask, os.path.join(mask_dir, mname))

    # occlusion planes
    for i in range(number_files):
        corrupted, mask = occlusion_plane(ground_truth_points)
        fname = f"{base_filename}_occlusionplane_{i+1}.xyz"
        mname = f"{base_filename}_occlusionplane_{i+1}_mask.npy"
        save_points_as_xyz(corrupted, os.path.join(corrupt_dir, fname))
        save_mask(mask, os.path.join(mask_dir, mname))

    # noise (masked subset are noised)
    for i in range(number_files + 1):
        noise_std = float(np.random.uniform(0.005, 0.02))
        corrupted, mask = add_noise(ground_truth_points, noise_std=noise_std)
        fname = f"{base_filename}_noise_{i+1}.xyz"
        mname = f"{base_filename}_noise_{i+1}_mask.npy"
        save_points_as_xyz(corrupted, os.path.join(corrupt_dir, fname))
        save_mask(mask, os.path.join(mask_dir, mname))


def process_all_scans(input_folder: str, output_folder: str, masked_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(masked_folder, exist_ok=True)
    files = sorted([file for file in os.listdir(input_folder) if file.endswith('.xyz')])
    for file in files:
        filepath = os.path.join(input_folder, file)
        points = np.loadtxt(filepath, delimiter=None)
        base = os.path.splitext(file)[0]
        generate_synthetic_scans(points, base, output_folder, masked_folder, 3)


if __name__ == "__main__":
    input_folder = "ground_truths"
    corrupted_folder = "synthetic_scans"
    masked_folder = "masks"
    process_all_scans(input_folder, corrupted_folder, masked_folder)
    print("All synthetic data saved.")
