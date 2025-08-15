import os
from add_noise import add_noise
from occlusion_plane import occlusion_plane
from random_global_dropout import random_global_dropout
from random_local_hole import random_local_hole
import numpy as np

def save_points_as_xyz(points, filename):
    np.savetxt(filename, points, delimiter=' ', fmt='%.6f')

def save_mask(filename, mask):
    np.save(filename, mask)

def generate_synthetic_scans(ground_truth_points, base_filename, corrupt_dir, mask_dir, number_files):
    os.makedirs(corrupt_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(base_filename, corrupt_dir, mask_dir)

    for i in range(number_files):
        print("localhole #", i)
        corrupted, mask = random_local_hole(ground_truth_points, radius=0.5, num_holes=3)
        save_points_as_xyz(corrupted, os.path.join(corrupt_dir, f"{base_filename}_localhole_{i+1}.xyz"))
        save_mask(mask, os.path.join(mask_dir, f"{base_filename}_localhole_{i+1}_mask.npy"))

    for i in range(number_files):
        print("globaldropout #", i)
        dropout_ratio = np.random.uniform(0.05, 0.2)
        corrupted, mask = random_global_dropout(ground_truth_points, dropout_ratio=dropout_ratio)
        save_points_as_xyz(corrupted, os.path.join(corrupt_dir, f"{base_filename}_globaldropout_{i+1}.xyz"))
        save_mask(mask, os.path.join(mask_dir, f"{base_filename}_globaldropout_{i+1}_mask.npy"))

    for i in range(number_files):
        print("occlusionplane #", i)
        corrupted, mask = occlusion_plane(ground_truth_points)
        save_points_as_xyz(corrupted, os.path.join(corrupt_dir, f"{base_filename}_occlusionplane_{i+1}.xyz"))
        save_mask(mask, os.path.join(mask_dir, f"{base_filename}_occlusionplane_{i + 1}_mask.npy"))

    for i in range(number_files):
        print("noise #", i)
        noise_std = np.random.uniform(0.005, 0.02)
        corrupted, mask = add_noise(ground_truth_points, noise_std=noise_std)
        save_points_as_xyz(corrupted, os.path.join(corrupt_dir, f"{base_filename}_noise_{i+1}.xyz"))
        save_mask(mask, os.path.join(mask_dir, f"{base_filename}_noise_{i+1}_mask.npy"))

def process_all_scans(input_folder, output_folder, masked_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        print(f"Processing {filename}...")
        filepath = os.path.join(input_folder, filename)
        points = np.loadtxt(filepath, delimiter=None)
        generate_synthetic_scans(points, filename, output_folder, masked_folder, 10)


if __name__ == "__main__":
    input_folder = "../ground_truths"
    corrupted_folder = "../synthetic_scans"
    masked_folder = "../masks"
    process_all_scans(input_folder, corrupted_folder, masked_folder)
    print("All inpainting processed and synthetic data saved.")



