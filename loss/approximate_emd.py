import torch
from scipy.optimize import linear_sum_assignment

def approximate_emd(pred, gt):
    B, N, _ = pred.size()
    total_loss = 0.0

    for b in range(B):
        # compute pairwise distances
        distances = torch.cdist(pred[b:b+1], gt[b:b+1], p=2)[0]  # [N, N]
        # Hungarian matching (minimum cost perfect matching)
        row_ind, col_ind = linear_sum_assignment(distances.detach().cpu().numpy())
        total_loss += distances[row_ind, col_ind].sum() / N

    return total_loss / B