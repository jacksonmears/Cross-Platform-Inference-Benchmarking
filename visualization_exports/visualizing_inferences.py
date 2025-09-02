import os
import numpy as np
import open3d as o3d



file_path = "C:\\Users\\jacks\\Cross-Platform-Inference-Benchmarking\\model_inferencing\\inferences"
files = [file for file in os.listdir(file_path)]

for file in files:
    print(file)
    full_path = os.path.join(file_path, file)
    pcd = o3d.io.read_point_cloud(full_path, format='xyz')
    o3d.visualization.draw_geometries([pcd])