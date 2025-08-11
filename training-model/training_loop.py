import torch
from GNN_autoencoder_model import *
from chamfer import *
from main import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GNNAutoencoder(k=16, num_points=1024).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    total_loss = 0
    for input_graph, target_points in train_loader:
        input_graph = input_graph.to(device)
        target_points = target_points.to(device).float()

        optimizer.zero_grad()
        output_points = model(input_graph)

        # Chamfer loss expects [B, N, 3] tensors
        loss, _ = chamfer_distance(output_points, target_points)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
