import torch
import torch.nn.functional as F
from training_model.config import K

def laplacian_loss(points):

    B, _, _ = points.size()
    total_loss = 0.0
    for b in range(B):
        # Compute pairwise distances
        dists = torch.cdist(points[b:b+1], points[b:b+1], p=2)[0]  # [N, N]
        knn_idx = dists.topk(K+1, largest=False).indices[:, 1:]  # skip self (index 0)

        neighbor_mean = points[b][knn_idx].mean(dim=1)  # [N, 3]
        total_loss += F.mse_loss(points[b], neighbor_mean)

    return total_loss / B