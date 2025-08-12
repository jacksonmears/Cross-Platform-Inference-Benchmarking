from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from training_model.config import K


def create_graph_from_point_cloud(points, k=K):
    edge_index = knn_graph(points, k, loop=False)  # [2, num_edges]

    x = points  # technically redundant code but important to understand
                # and emphasize what x's role is

    data = Data(x=x, edge_index=edge_index)
    return data
