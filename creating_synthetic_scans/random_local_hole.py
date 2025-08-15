import numpy as np

def random_local_hole(points, radius=0.5, num_holes=1):
    points = points.copy()
    mask = np.zeros(len(points), dtype=bool)

    for _ in range(num_holes):
        center_idx = np.random.choice(len(points))
        center = points[center_idx]
        distances = np.linalg.norm(points - center, axis=1)
        hole_mask = distances <= radius  # points inside the hole
        mask[hole_mask] = True          # mark these points as corrupted
        points = points[~hole_mask]     # remove points inside the hole

    return points, mask
