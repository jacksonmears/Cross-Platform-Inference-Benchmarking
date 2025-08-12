from create_graphs import create_graph_from_point_cloud
from fetch_points import *
from config import K


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, original_points_list, synthetic_points_list):
        self.original_points_list = original_points_list
        self.synthetic_points_list = synthetic_points_list
        self.k = K

    def __len__(self):
        return len(self.original_points_list)

    def __getitem__(self, idx):
        input_points = torch.tensor(self.synthetic_points_list[idx], dtype=torch.float)
        input_points = fixed_size_points(input_points)  # reduce first to save memory!
        input_graph = create_graph_from_point_cloud(input_points)

        target_points = torch.tensor(self.original_points_list[idx], dtype=torch.float)
        target_points = fixed_size_points(target_points)

        input_graph.batch = torch.zeros(input_graph.x.size(0), dtype=torch.long)

        return input_graph, target_points
