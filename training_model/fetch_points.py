import torch
from config import NUM_POINTS
import os
import random
import string


def fixed_size_points(points: torch.Tensor):
    """Simple down/up sampling to NUM_POINTS with optional index saving"""
    N = points.size(0)
    if N > NUM_POINTS:
        idx = torch.randperm(N)[:NUM_POINTS]

        return points[idx]
    elif N < NUM_POINTS:
        pad_size = NUM_POINTS - N
        pad = points[-1].unsqueeze(0).repeat(pad_size, 1)
        return torch.cat([points, pad], dim=0)
    return points

def fixed_size_points_with_mask_torch(points: torch.Tensor, mask: torch.Tensor):
    """
    Downsample or pad a point cloud with mask to exactly NUM_POINTS.

    Args:
        points: Tensor [N, 3]
        mask: BoolTensor [N], True = masked/corrupted points

    Returns:
        points_down: Tensor [NUM_POINTS, 3]
        mask_down: BoolTensor [NUM_POINTS], True where the point was originally masked
    """
    if not torch.is_tensor(points):
        raise TypeError("points must be a torch.Tensor")
    if not torch.is_tensor(mask):
        raise TypeError("mask must be a torch.Tensor")
    if mask.dtype != torch.bool:
        mask = mask.bool()
    N = points.size(0)
    if mask.numel() != N:
        raise ValueError(f"Mask length {mask.numel()} != points length {N}")

    masked_points = points[mask]       # [M, 3]
    unmasked_points = points[~mask]    # [N-M, 3]

    M = masked_points.size(0)

    if M >= NUM_POINTS:
        # Only sample from masked points
        perm = torch.randperm(M)[:NUM_POINTS]
        points_down = masked_points[perm]
        mask_down = torch.ones(NUM_POINTS, dtype=torch.bool)
    else:
        remaining = NUM_POINTS - M

        # Sample unmasked points if available
        if unmasked_points.size(0) >= remaining:
            perm_u = torch.randperm(unmasked_points.size(0))[:remaining]
            sampled_unmasked = unmasked_points[perm_u]
        else:
            sampled_unmasked = unmasked_points

        # Concatenate masked and unmasked points
        points_down = torch.cat([masked_points, sampled_unmasked], dim=0)
        mask_down = torch.cat([
            torch.ones(masked_points.size(0), dtype=torch.bool),
            torch.zeros(points_down.size(0) - masked_points.size(0), dtype=torch.bool)
        ], dim=0)

        # Pad if still too short
        if points_down.size(0) < NUM_POINTS:
            pad_size = NUM_POINTS - points_down.size(0)
            pad_point = points_down[-1].unsqueeze(0).repeat(pad_size, 1)
            points_down = torch.cat([points_down, pad_point], dim=0)
            mask_down = torch.cat([mask_down, torch.zeros(pad_size, dtype=torch.bool)], dim=0)

    # Final shuffle to remove ordering bias
    perm_all = torch.randperm(points_down.size(0))
    points_down = points_down[perm_all]
    mask_down = mask_down[perm_all]

    # Safety check
    assert points_down.size(0) == NUM_POINTS
    assert mask_down.size(0) == NUM_POINTS

    return points_down, mask_down

