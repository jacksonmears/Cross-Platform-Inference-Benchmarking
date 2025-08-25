from torch.utils.data import Dataset
import torch
from create_graphs import create_graph_from_point_cloud
from fetch_points import fixed_size_points_with_mask_torch, fixed_size_points

class PointCloudDataset(Dataset):
    def __init__(self, ground_points_list, synthetic_points_list, mask_points_list, mask_fraction_schedule=None):
        self.ground_points_list = ground_points_list
        self.synthetic_points_list = synthetic_points_list
        self.mask_points_list = mask_points_list
        self.mask_fraction_schedule = mask_fraction_schedule

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def __len__(self):
        return len(self.ground_points_list)

    def __getitem__(self, idx):
        input_points = torch.tensor(self.synthetic_points_list[idx], dtype=torch.float)
        mask = torch.tensor(self.mask_points_list[idx], dtype=torch.bool)

        # Curriculum masking
        if self.mask_fraction_schedule is not None:
            frac = self.mask_fraction_schedule(self.current_epoch)

            masked_idx = torch.nonzero(mask).squeeze(1)
            num_to_keep = int(len(masked_idx) * frac)

            if num_to_keep < len(masked_idx):
                keep_idx = torch.randperm(len(masked_idx))[:num_to_keep]
                new_mask = torch.zeros_like(mask)
                new_mask[masked_idx[keep_idx]] = True
                mask = new_mask

        # Downsample points + mask
        input_points, mask = fixed_size_points_with_mask_torch(input_points, mask)

        # Build graph
        input_graph = create_graph_from_point_cloud(input_points)
        input_graph.batch = torch.zeros(input_graph.x.size(0), dtype=torch.long)

        # Target GT points
        target_points = torch.tensor(self.ground_points_list[idx], dtype=torch.float)
        target_points = fixed_size_points(target_points)

        return input_graph, target_points, mask
