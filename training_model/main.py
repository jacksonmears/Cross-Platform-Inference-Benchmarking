import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from train_loader import PointCloudDataset
from torch_geometric.loader import DataLoader
from model.GNN_autoencoder_model import GNNAutoencoder
from fetching_files import build_pointcloud_lists
import os
from loss.inpainting_loss import inpainting_loss

# CLEAR overfitting right now omg we NEED WAY more sample data and maybe less synthetic scans per ground truth.
# it looks like all model scans are coming out almost identical to the GT
# which would be sweet if I wasn't 100% it should be impossible
# given the alterations we made while creating the synthetic scans


best_loss_file = "best_loss.txt"
save_path = 'model/gnn_autoencoder.pth'

ground_points_list, synthetic_points_list, mask_points_list = build_pointcloud_lists()
dataset = PointCloudDataset(
    ground_points_list, 
    synthetic_points_list, 
    mask_points_list,
    mask_fraction_schedule=lambda epoch: linear_mask_schedule(epoch, max_epoch=TOTAL_NEW_EPOCHS)
    )
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

TOTAL_NEW_EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fetch_best_loss():
    if not os.path.exists(best_loss_file):
        return float('inf')
    with open(best_loss_file, "r") as file:
        try:
            return float(file.read().strip())
        except:
            return float('inf')


def save_best_loss(loss):
    with open(best_loss_file, "w") as file:
        file.write(str(loss))


def linear_mask_schedule(epoch, max_epoch, start_frac=0.1, end_frac=0.7):
    """
    Returns a mask fraction for the given epoch, linearly increasing from start_frac to end_frac.
    """
    if epoch >= max_epoch:
        return end_frac
    return start_frac + (end_frac - start_frac) * (epoch / max_epoch)



def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_loss = checkpoint['loss']
    print(f"Resuming from epoch {start_epoch}, best loss {best_loss:.6f}")
    return start_epoch, best_loss


def main():
    model = GNNAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = fetch_best_loss()
    loss_threshold = 0.01

    # ask user if they want to resume training if checkpoint exists
    user_input = input(f"Checkpoint found with best loss {best_loss:.5f}. Resume training? (y/n): ").strip().lower()
    if os.path.exists(save_path) and user_input == 'y':
        start_epoch, best_loss = load_checkpoint(model, optimizer, save_path)
    else:
        print("Either checkpoint wasn't found or user didn't want to resume. Starting training from scratch.")
        best_loss = float('inf')
        save_best_loss(best_loss)
        start_epoch = 0

    for epoch in range(start_epoch, start_epoch+TOTAL_NEW_EPOCHS):
        dataset.set_epoch(epoch)
        print(epoch, dataset.mask_fraction_schedule)
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

        if avg_loss < best_loss - loss_threshold:
            best_loss = avg_loss
            save_best_loss(best_loss)
            loss_threshold = avg_loss * 0.1     # MAY NEED TO ADJUST LIL BRO
            print("New threshold value: ", best_loss - loss_threshold)

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
