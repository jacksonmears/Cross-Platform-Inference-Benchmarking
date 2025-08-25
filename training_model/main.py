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
from training_model.loss_json_functions import save_best_losses, fetch_best_losses, update_losses, update_model
from training_model.model_functions import update_model, load_checkpoint

# CLEAR overfitting right now omg we NEED WAY more sample data and maybe less synthetic scans per ground truth.
# it looks like all model scans are coming out almost identical to the GT
# which would be sweet if I wasn't 100% it should be impossible
# given the alterations we made while creating the synthetic scans


all_time_path = "model/all_time.pth"
strategy_path = "model/strategy.pth"
last_run_path = "model/last_run.pth"
current_run_path = "model/current_run.pth"


TOTAL_NEW_EPOCHS = 100

ground_points_list, synthetic_points_list, mask_points_list = build_pointcloud_lists()
dataset = PointCloudDataset(
    ground_points_list, 
    synthetic_points_list, 
    mask_points_list,
    mask_fraction_schedule=lambda epoch: linear_mask_schedule(epoch, max_epoch=TOTAL_NEW_EPOCHS)
    )
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def linear_mask_schedule(epoch, max_epoch, start_frac=0.1, end_frac=0.7): # originally 0.2 and 0.7
    if epoch >= max_epoch:
        return end_frac
    return start_frac + (end_frac - start_frac) * (epoch / max_epoch)

def first_epoch_check(epoch, total_epochs):
    return epoch <= total_epochs

def second_epoch_check(epoch, total_epochs, has_saved):
    return epoch <= total_epochs*10 and not has_saved




def main():
    model = GNNAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_losses = fetch_best_losses()

    # ask user if they want to resume training if checkpoint exists
    user_input = input(f"Checkpoint found with current strategy with best loss {best_losses['strategy']:.5f}. Resume training? (y/n): ").strip().lower()
    if os.path.exists(strategy_path) and user_input == 'y':
        start_epoch = load_checkpoint(model, optimizer, strategy_path)
        best_losses["last_run"] = best_losses["current_run"]
        update_model(start_epoch, model, optimizer, best_losses["last_run"], "last_run")
        best_losses["current_run"] = float("inf")
    else:
        print("Either checkpoint wasn't found or user didn't want to resume. Starting training from scratch.")
        best_losses = {
            "all_time": best_losses["all_time"],
            "strategy": float("inf"),
            "last_run": float("inf"),
            "current_run": float("inf")
        }
        start_epoch = 0
    
    best_losses["strategy"] *= 1.6      # made the 1.4 up so the model can have an easier time updating strategy for the first time after a new run has begun but gets harder after that with const threshold 
    save_best_losses(best_losses)
    
    has_saved = False
    epoch = start_epoch
    total_epochs = start_epoch + TOTAL_NEW_EPOCHS

    while first_epoch_check(epoch, total_epochs) or second_epoch_check(epoch, total_epochs, has_saved):
    # for epoch in range(start_epoch, start_epoch+TOTAL_NEW_EPOCHS):
        dataset.set_epoch(epoch)
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

        if update_losses(avg_loss, best_losses, epoch, model, optimizer):
            has_saved = True

        epoch += 1


if __name__ == "__main__":
    main()
