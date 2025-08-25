import torch

def masked_chamfer_distance(pred, gt, mask):
    x = pred.unsqueeze(2)  # [B, N, 1, 3]
    y = gt.unsqueeze(1)  # [B, 1, N, 3]
    distance = torch.norm(x - y, dim=3)  # [B, N, N]

    min_dist_x, _ = distance.min(dim=2)  # [B, N]
    min_dist_y, _ = distance.min(dim=1)  # [B, N]

    masked_x = min_dist_x * mask.float()
    masked_y = min_dist_y * mask.float()

    # normalize by number of points in mask
    loss_x = masked_x.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    loss_y = masked_y.sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    return (loss_x + loss_y).mean()