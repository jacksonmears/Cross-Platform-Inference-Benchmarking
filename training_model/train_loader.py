from torch.utils.data import Dataset
import torch
from create_graphs import create_graph_from_point_cloud
from fetch_points import fixed_size_points_with_mask_torch, fixed_size_points

class PointCloudDataset(Dataset):
    def __init__(self, ground_points_list, synthetic_points_list, mask_points_list):
        self.ground_points_list = ground_points_list
        self.synthetic_points_list = synthetic_points_list
        self.mask_points_list = mask_points_list

    def __len__(self):
        return len(self.ground_points_list)

    def __getitem__(self, idx):
        # Convert to torch tensors
        input_points = torch.tensor(self.synthetic_points_list[idx], dtype=torch.float)
        mask = torch.tensor(self.mask_points_list[idx], dtype=torch.bool)

        # Downsample points while preserving masked points
        input_points, mask = fixed_size_points_with_mask_torch(input_points, mask)

        # Build graph
        input_graph = create_graph_from_point_cloud(input_points)
        input_graph.batch = torch.zeros(input_graph.x.size(0), dtype=torch.long)

        # Prepare target points
        target_points = torch.tensor(self.ground_points_list[idx], dtype=torch.float)
        target_points = fixed_size_points(target_points)

        # Debug: check mask stats
        # print("Mask stats:", mask.shape, "masked:", mask.sum().item(), "kept:", (~mask).sum().item())

        return input_graph, target_points, mask
