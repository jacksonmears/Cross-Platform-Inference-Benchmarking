from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from config import K

def create_graph_from_point_cloud(points, k=K):
    
    edge_index = knn_graph(points, k=k, loop=False)  # [2, num_edges]
    data = Data(x=points, edge_index=edge_index)

    return data
