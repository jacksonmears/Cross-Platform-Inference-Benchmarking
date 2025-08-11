import os
import numpy as np
from functions import *

def save_points_as_xyz(points, filename):
    np.savetxt(filename, points, delimiter=' ', fmt='%.6f')

def generate_synthetic_scans(original_points, base_filename, save_dir, number_files):
    os.makedirs(save_dir, exist_ok=True)

    print(base_filename, save_dir)

    for i in range(number_files):
        corrupted = random_local_hole(original_points, radius=0.5, num_holes=3)
        save_points_as_xyz(corrupted, os.path.join(save_dir, f"{base_filename}_localhole_{i+1}.xyz"))

    for i in range(number_files):
        dropout_ratio = np.random.uniform(0.05, 0.2)
        corrupted = random_global_dropout(original_points, dropout_ratio=dropout_ratio)
        save_points_as_xyz(corrupted, os.path.join(save_dir, f"{base_filename}_globaldropout_{i+1}.xyz"))

    for i in range(number_files):
        corrupted = occlusion_plane(original_points)
        save_points_as_xyz(corrupted, os.path.join(save_dir, f"{base_filename}_occlusionplane_{i+1}.xyz"))

    for i in range(number_files):
        noise_std = np.random.uniform(0.005, 0.02)
        corrupted = add_noise(original_points, noise_std=noise_std)
        save_points_as_xyz(corrupted, os.path.join(save_dir, f"{base_filename}_noise_{i+1}.xyz"))


def process_all_scans(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        print(f"Processing {filename}...")
        filepath = os.path.join(input_folder, filename)
        points = np.loadtxt(filepath, delimiter=None)
        generate_synthetic_scans(points, filename, output_folder, 10)


if __name__ == "__main__":
    input_folder = "../original-scans"
    output_folder = "../synthetic-scans"
    process_all_scans(input_folder, output_folder)
    print("All scans processed and synthetic data saved.")



