from create_graphs import create_graph_from_point_cloud
from fetch_points import *
from config import K


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, ground_points_list, synthetic_points_list, mask_points_list):
        self.ground_points_list = ground_points_list
        self.synthetic_points_list = synthetic_points_list
        self.mask_points_list = mask_points_list
        self.k = K

    def __len__(self):
        return len(self.ground_points_list)

    def __getitem__(self, idx):
        input_points = torch.tensor(self.synthetic_points_list[idx], dtype=torch.float)
        input_points = fixed_size_points(input_points)  # reduce first to save memory!
        input_graph = create_graph_from_point_cloud(input_points)
        input_graph.batch = torch.zeros(input_graph.x.size(0), dtype=torch.long)

        target_points = torch.tensor(self.ground_points_list[idx], dtype=torch.float)
        target_points = fixed_size_points(target_points)

        mask = torch.tensor(self.mask_points_list[idx], dtype=torch.bool)
        mask = fixed_size_points(mask)

        return input_graph, target_points, mask
