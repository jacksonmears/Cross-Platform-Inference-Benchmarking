# add_noise.py
import numpy as np

def add_noise(points: np.ndarray, noise_std: float = 0.01, mask_ratio: float = 0.1):
    """
    Add Gaussian noise only to masked points. Return synthetic points and a 1-D boolean mask
    aligned to the synthetic point ordering.
    """
    pts = points.copy()
    N = pts.shape[0]
    mask = np.zeros(N, dtype=bool)

    num_mask = max(1, int(N * mask_ratio))
    mask_indices = np.random.choice(N, num_mask, replace=False)
    mask[mask_indices] = True

    # apply noise only to masked points
    pts[mask_indices] += np.random.normal(scale=noise_std, size=(num_mask, pts.shape[1]))

    return pts, mask
