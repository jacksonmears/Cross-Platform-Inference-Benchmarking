from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from config import K

def create_graph_from_point_cloud(points, k=K):
    """
    points: torch.Tensor [N,3]
    returns: PyG Data object with edge_index and x=points
    """
    edge_index = knn_graph(points, k=k, loop=False)  # [2, num_edges]
    data = Data(x=points, edge_index=edge_index)
    return data
