import torch
import torch.nn.functional as F

def laplacian_loss(points, k=8):
    """
    points: [B, N, 3]
    k: number of neighbors
    Encourages smooth surfaces by penalizing deviation from local mean
    """
    B, N, _ = points.size()
    total_loss = 0.0
    for b in range(B):
        # Compute pairwise distances
        dists = torch.cdist(points[b:b+1], points[b:b+1], p=2)[0]  # [N, N]
        knn_idx = dists.topk(k+1, largest=False).indices[:, 1:]  # skip self (index 0)

        neighbor_mean = points[b][knn_idx].mean(dim=1)  # [N, 3]
        total_loss += F.mse_loss(points[b], neighbor_mean)
    return total_loss / B