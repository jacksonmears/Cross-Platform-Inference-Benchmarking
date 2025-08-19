# random_global_dropout.py
import numpy as np

def random_global_dropout(points: np.ndarray, dropout_ratio: float = 0.1, sentinel=1e6):
    """
    Mark a random subset of points as dropped (masked) and set a sentinel in synthetic points.
    Returns (corrupted_points, mask_bool_1d).
    """
    n_points = len(points)
    drop_num = int(n_points * dropout_ratio)
    drop_num = max(1, drop_num)
    drop_indices = np.random.choice(n_points, drop_num, replace=False)

    mask = np.zeros(n_points, dtype=bool)
    mask[drop_indices] = True

    corrupted = points.copy()
    corrupted[drop_indices] = np.array([sentinel, sentinel, sentinel], dtype=points.dtype)
    return corrupted, mask
