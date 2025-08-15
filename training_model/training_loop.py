from main import *
from loss.inpainting_loss import inpainting_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GNNAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    total_loss = 0

    for input_graph, target_points, mask in train_loader:
        input_graph = input_graph.to(device)
        target_points = target_points.to(device).float()
        mask = mask.to(device)

        optimizer.zero_grad()
        output_points = model(input_graph)

        loss = inpainting_loss(
            pred=output_points,
            gt=target_points,
            mask=mask,
            lambda_cd=1.0,
            lambda_emd=0.1,
            lambda_lap=0.01
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
