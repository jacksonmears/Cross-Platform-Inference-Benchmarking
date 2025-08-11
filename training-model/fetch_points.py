import torch

def fixed_size_points(points, num_points=1024):
    if points.shape[0] > num_points:
        # Randomly sample without replacement
        indices = torch.randperm(points.shape[0])[:num_points]
        points = points[indices]
    elif points.shape[0] < num_points:
        # Pad by repeating last point
        pad_size = num_points - points.shape[0]
        pad = points[-1].repeat(pad_size, 1)
        points = torch.cat([points, pad], dim=0)
    return points
