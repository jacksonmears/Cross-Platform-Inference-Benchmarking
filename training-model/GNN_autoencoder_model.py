import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_max_pool
from torch import nn

class GNNEncoder(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        # EdgeConv layers
        self.conv1 = EdgeConv(nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64)))
        self.conv2 = EdgeConv(nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128)))
        self.conv3 = EdgeConv(nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256)))

        self.fc = nn.Linear(256, 128)

    def forward(self, x, edge_index, batch):
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)

        # Global pooling to get a fixed-length embedding per graph
        x_pool = global_max_pool(x3, batch)
        embedding = self.fc(x_pool)
        return embedding


class GNNDecoder(nn.Module):
    def __init__(self, num_points=1024):
        super().__init__()
        self.num_points = num_points
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_points * 3)  # reconstruct xyz of all points
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(-1, self.num_points, 3)  # batch_size x num_points x 3
        return out


class GNNAutoencoder(nn.Module):
    def __init__(self, k=16, num_points=1024):
        super().__init__()
        self.encoder = GNNEncoder(k)
        self.decoder = GNNDecoder(num_points)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        z = self.encoder(x, edge_index, batch)
        reconstructed = self.decoder(z)
        return reconstructed
