import open3d as o3d
import os

# folder_path = "original-scans"
#
# files = [file for file in os.listdir(folder_path)]
#
#
# for file in files:
#     print(file)
#     pcd = o3d.io.read_point_cloud(os.path.join(folder_path, file), format="xyz")
#
#     o3d.visualization.draw_geometries([pcd])




file = "original-scans/000001.xyz"
pcd = o3d.io.read_point_cloud(file, format='xyz')
o3d.visualization.draw_geometries([pcd])






folder_path = "synthetic-scans"

files = [file for file in os.listdir(folder_path)]


for file in files:
    print(file)
    pcd = o3d.io.read_point_cloud(os.path.join(folder_path, file), format="xyz")

    o3d.visualization.draw_geometries([pcd])
