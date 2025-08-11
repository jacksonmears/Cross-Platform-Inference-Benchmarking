import torch
from train_loader import PointCloudDataset
from torch_geometric.loader import DataLoader
from GNN_autoencoder_model import GNNAutoencoder
from chamfer import chamfer_distance
from fetching_files import original_points_list, synthetic_points_list

dataset = PointCloudDataset(original_points_list, synthetic_points_list, k=8) # originally 16 but downsampled because memeory couldn't handle 16
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNAutoencoder(k=16, num_points=1024).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

save_path = '../model/gnn_autoencoder.pth'

for epoch in range(100):
    model.train()
    total_loss = 0
    for input_graph, target_points in train_loader:
        input_graph = input_graph.to(device)
        target_points = target_points.to(device).float()

        optimizer.zero_grad()
        output_points = model(input_graph)

        loss = chamfer_distance(output_points, target_points)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    # Save model checkpoint every N epochs or based on loss improvement
    if (epoch + 1) % 10 == 0:  # e.g. every 10 epochs
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

