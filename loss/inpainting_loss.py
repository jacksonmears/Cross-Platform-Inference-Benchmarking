from loss.chamfer import chamfer_distance
from loss.masked_chamfer import masked_chamfer_distance
from loss.laplacian_loss import laplacian_loss
from loss.approximate_emd import approximate_emd

def inpainting_loss(pred, gt, mask=None, lambda_cd=0.75, lambda_emd=0.1, lambda_lap=0.05):

    if mask is not None:
        loss_cd = masked_chamfer_distance(pred, gt, mask)
    else:
        loss_cd = chamfer_distance(pred, gt)

    loss_lap = laplacian_loss(pred)

    # loss_emd = approximate_emd(pred, gt)                    # Uncomment if N is small or approximate EMD available   # EMD (optional, slow, probably never gonna use it until GPU implementation)
    loss_emd = 0.0                                       # Uncomment if we chose to NOT use loss_emd 

    return lambda_cd*loss_cd + lambda_emd*loss_emd + lambda_lap*loss_lap