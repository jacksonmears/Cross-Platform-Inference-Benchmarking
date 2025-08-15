import numpy as np

def occlusion_plane(points, plane_normal=None, plane_point=None):
    if plane_normal is None:
        plane_normal = np.random.randn(3)
    plane_normal /= np.linalg.norm(plane_normal)
    if plane_point is None:
        plane_point = points[np.random.choice(len(points))]

    distances = np.dot(points - plane_point, plane_normal)
    mask = distances <= 0  # points behind the plane are occluded
    return points[~mask], mask  # points kept, mask indicates occluded points
