from chamfer import *
from masked_chamfer import *
from laplacian_loss import *
from scipy.optimize import linear_sum_assignment
from approximate_emd import *

def inpainting_loss(pred, gt, mask=None, lambda_cd=1.0, lambda_emd=0.1, lambda_lap=0.01):
    """
    pred, gt: [B, N, 3]
    mask: [B, N] boolean tensor for corrupted points (optional)
    """
    if mask is not None:
        loss_cd = masked_chamfer_distance(pred, gt, mask)
    else:
        loss_cd = chamfer_distance(pred, gt)

    loss_lap = laplacian_loss(pred)

    # loss_emd = approximate_emd(pred, gt)                  # Uncomment if N is small or approximate EMD available   # EMD (optional, slow)
    loss_emd = 0.0

    return lambda_cd*loss_cd + lambda_emd*loss_emd + lambda_lap*loss_lap