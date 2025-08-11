import torch
from config import NUM_POINTS


def fixed_size_points(points):
    if points.shape[0] > NUM_POINTS:
        # Randomly sample without replacement
        indices = torch.randperm(points.shape[0])[:NUM_POINTS]
        points = points[indices]
    elif points.shape[0] < NUM_POINTS:
        # Pad by repeating last point
        pad_size = NUM_POINTS - points.shape[0]
        pad = points[-1].repeat(pad_size, 1)
        points = torch.cat([points, pad], dim=0)
    return points
