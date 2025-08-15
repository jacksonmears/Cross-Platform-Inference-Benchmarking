# import numpy as np
#
# def random_local_hole(points, radius=0.5, num_holes=1):
#     points = points.copy()
#     for _ in range(num_holes):
#         center = points[np.random.choice(len(points))]
#         distances = np.linalg.norm(points - center, axis=1)
#         mask = distances > radius # slightly redundant code but adds clarity
#         points = points[mask]
#     return points
#
# def random_global_dropout(points, dropout_ratio=0.1):
#     n_points = len(points)
#     keep_num = int(n_points * (1 - dropout_ratio))
#     indices = np.random.choice(n_points, keep_num, replace=False)
#     return points[indices]
#
# def occlusion_plane(points, plane_normal=None, plane_point=None):
#     if plane_normal is None:
#         plane_normal = np.random.randn(3)
#     plane_normal /= np.linalg.norm(plane_normal)
#     if plane_point is None:
#         plane_point = points[np.random.choice(len(points))]
#
#     points = points.copy()
#     distances = np.dot(points - plane_point, plane_normal)
#     mask = distances > 0
#     return points[mask]
#
# def add_noise(points, noise_std=0.01):
#     noise = np.random.normal(scale=noise_std, size=points.shape)
#     noisy_points = points + noise
#     return noisy_points
