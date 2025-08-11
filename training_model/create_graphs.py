from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from config import K


def create_graph_from_point_cloud(points, k=K):  # oringally 16 !!!
    """
    points: torch.Tensor of shape [N, 3] (xyz coordinates)
    k: number of neighbors for edges
    """
    # Build edge_index with k-NN
    edge_index = knn_graph(points, k, loop=False)  # [2, num_edges]

    # Node features can just be the xyz coords here (or add reflectance if you want)
    x = points

    # Create PyG graph data object
    data = Data(x=x, edge_index=edge_index)
    return data
