import numpy as np

def random_global_dropout(points, dropout_ratio=0.1):
    n_points = len(points)
    keep_num = int(n_points * (1 - dropout_ratio))
    keep_indices = np.random.choice(n_points, keep_num, replace=False)

    mask = np.ones(n_points, dtype=bool)
    mask[keep_indices] = False  # points we keep are not corrupted

    return points[keep_indices], mask
