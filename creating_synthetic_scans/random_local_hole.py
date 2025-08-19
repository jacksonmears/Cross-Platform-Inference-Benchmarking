# random_local_hole.py
import numpy as np

def random_local_hole(points: np.ndarray, num_holes: int = 1, hole_scale_min: float = 0.05, hole_scale_max: float = 0.1, sentinel=1e6):
    """
    Create random spherical holes in the pointcloud. Masked points are replaced by sentinel.
    """
    pts = points.copy()
    N = pts.shape[0]
    mask = np.zeros(N, dtype=bool)

    min_bounds = pts.min(axis=0)
    max_bounds = pts.max(axis=0)
    bbox_diag = np.linalg.norm(max_bounds - min_bounds)

    for _ in range(num_holes):
        scale = np.random.uniform(hole_scale_min, hole_scale_max)
        radius = scale * bbox_diag
        center_idx = np.random.choice(N)
        center = pts[center_idx]
        distances = np.linalg.norm(pts - center, axis=1)
        hole_mask = distances <= radius
        mask = mask | hole_mask

    corrupted = pts.copy()
    corrupted[mask] = np.array([sentinel, sentinel, sentinel], dtype=pts.dtype)

    return corrupted, mask
