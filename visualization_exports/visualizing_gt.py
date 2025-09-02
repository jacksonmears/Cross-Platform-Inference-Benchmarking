import os
import numpy as np
import open3d as o3d



file_path = "C:\\Users\\jacks\\Cross-Platform-Inference-Benchmarking\\ground_truths"
specific_files = ["000001.xyz", "000002.xyz", "000025.xyz"]

for file in specific_files:
    full_path = os.path.join(file_path,file)

    pcd = o3d.io.read_point_cloud(full_path, format='xyz')
    o3d.visualization.draw_geometries([pcd])
