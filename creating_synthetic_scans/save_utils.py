# save_utils.py
import numpy as np
import os

def save_points_as_xyz(points: np.ndarray, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, points, delimiter=' ', fmt='%.6f')

def save_mask(mask: np.ndarray, filename: str):
    """
    Save mask as 1-D boolean array.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np_mask = np.asarray(mask).astype(np.bool_)
    np.save(filename, np_mask)
