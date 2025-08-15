import torch
from scipy.optimize import linear_sum_assignment

def approximate_emd(pred, gt):
    """
    pred, gt: [B, N, 3]
    returns scalar loss
    """
    B, N, _ = pred.size()
    total_loss = 0.0
    for b in range(B):
        # compute pairwise distances
        dists = torch.cdist(pred[b:b+1], gt[b:b+1], p=2)[0]  # [N, N]
        # Hungarian matching (minimum cost perfect matching)
        row_ind, col_ind = linear_sum_assignment(dists.detach().cpu().numpy())
        total_loss += dists[row_ind, col_ind].sum() / N
    return total_loss / B