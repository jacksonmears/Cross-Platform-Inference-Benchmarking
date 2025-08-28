import numpy as np
from training_model.config import LEARNING_RATE
def occlusion_plane(points: np.ndarray, plane_normal=None, plane_point=None, sentinel=LEARNING_RATE):

    if plane_normal is None:
        plane_normal = np.random.randn(3)
        plane_normal /= np.linalg.norm(plane_normal)
        
    if plane_point is None:
        plane_point = points[np.random.choice(len(points))]

    distances = np.dot(points - plane_point, plane_normal)
    mask = distances <= 0  # Points behind the plane are masked

    corrupted = points.copy()
    
    # replace masked points with sentinel (so file still has same length)
    corrupted[mask] = np.array([sentinel, sentinel, sentinel], dtype=points.dtype)

    return corrupted, mask
