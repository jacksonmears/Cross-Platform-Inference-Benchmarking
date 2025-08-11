import torch

def chamfer_distance(points1, points2):
    # points1, points2: [B, N, 3]
    x, y = points1, points2
    x = x.unsqueeze(2)  # [B, N, 1, 3]
    y = y.unsqueeze(1)  # [B, 1, M, 3]
    dist = torch.norm(x - y, dim=3)  # [B, N, M]
    min_dist_x, _ = dist.min(dim=2)  # [B, N]
    min_dist_y, _ = dist.min(dim=1)  # [B, M]
    loss = min_dist_x.mean(dim=1) + min_dist_y.mean(dim=1)  # [B]
    return loss.mean()  # scalar
