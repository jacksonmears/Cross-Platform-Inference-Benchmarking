import numpy as np

def random_local_hole(points, radius=0.5, num_holes=1):
    """
    Remove points inside randomly selected spheres (holes).
    radius: radius of each hole sphere
    num_holes: how many holes to create per scan
    """
    points = points.copy()
    for _ in range(num_holes):
        center = points[np.random.choice(len(points))]
        distances = np.linalg.norm(points - center, axis=1)
        mask = distances > radius
        points = points[mask]
    return points

def random_global_dropout(points, dropout_ratio=0.1):
    """
    Randomly drop a % of points globally.
    dropout_ratio: fraction of points to drop
    """
    n_points = len(points)
    keep_num = int(n_points * (1 - dropout_ratio))
    indices = np.random.choice(n_points, keep_num, replace=False)
    return points[indices]

def occlusion_plane(points, plane_normal=None, plane_point=None):
    """
    Remove points on one side of a plane.
    plane_normal: normal vector of the plane (3D)
    plane_point: a point on the plane (3D)
    If None, random plane is created.
    """
    if plane_normal is None:
        plane_normal = np.random.randn(3)
    plane_normal /= np.linalg.norm(plane_normal)
    if plane_point is None:
        plane_point = points[np.random.choice(len(points))]

    points = points.copy()
    # Compute signed distance from plane
    distances = np.dot(points - plane_point, plane_normal)
    # Keep points on one side (e.g. positive side)
    mask = distances > 0
    return points[mask]

def add_noise(points, noise_std=0.01):
    """
    Add Gaussian jitter noise to points.
    noise_std: standard deviation of noise
    """
    noise = np.random.normal(scale=noise_std, size=points.shape)
    noisy_points = points + noise
    return noisy_points
