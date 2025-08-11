import torch
from create_graphs import create_graph_from_point_cloud  # make sure this function exists and returns torch_geometric.data.Data
from fetch_points import *

class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, original_points_list, synthetic_points_list, k=16):
        self.original_points_list = original_points_list
        self.synthetic_points_list = synthetic_points_list
        self.k = k

    def __len__(self):
        return len(self.original_points_list)

    def __getitem__(self, idx):
        # Convert synthetic partial scan to graph
        input_points = torch.tensor(self.synthetic_points_list[idx], dtype=torch.float)
        input_graph = create_graph_from_point_cloud(input_points, k=self.k)

        # Target original points (pad or sample to fixed size)
        target_points = torch.tensor(self.original_points_list[idx], dtype=torch.float)
        target_points = fixed_size_points(target_points, num_points=1024)

        # For batch processing, add dummy batch vector (all zeros for single graph)
        input_graph.batch = torch.zeros(input_graph.x.size(0), dtype=torch.long)

        return input_graph, target_points




