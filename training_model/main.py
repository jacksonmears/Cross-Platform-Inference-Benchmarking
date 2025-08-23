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
import json

# CLEAR overfitting right now omg we NEED WAY more sample data and maybe less synthetic scans per ground truth.
# it looks like all model scans are coming out almost identical to the GT
# which would be sweet if I wasn't 100% it should be impossible
# given the alterations we made while creating the synthetic scans


best_losses_file = "best_losses.json"
all_time_path = "model/all_time.pth"
strategy_path = "model/strategy.pth"
last_run_path = "model/last_run.pth"
current_run_path = "model/current_run.pth"

ALL_TIME_THRESHOLD_CONST = 0.9
STRATEGY_THRESHOLD_CONST = 0.8
CURRENT_RUN_THRESHOLD_CONST = 1
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

        
def fetch_best_losses():
    defaults = {
        "all_time": float("inf"),
        "strategy": float("inf"),
        "last_run": float("inf"),
        "current_run": float("inf"),
    }

    if not os.path.exists(best_losses_file):
        return defaults

    try:
        with open(best_losses_file, "r") as file:
            data = json.load(file)
        defaults.update(data)
        print(defaults)
        return defaults
    except Exception:
        return defaults


def save_best_losses(losses):
    with open(best_losses_file, "w") as file:
        json.dump(losses, file, indent=2)

def update_model(epoch, model, optimizer, avg_loss, path):
    full_path = os.path.join("model", f"{path}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, full_path)
    print(f"Saved checkpoint to {full_path}")


def update_losses(avg_loss, best_losses, epoch, model, optimizer):
    data = {"all_time": ALL_TIME_THRESHOLD_CONST, "strategy": STRATEGY_THRESHOLD_CONST, "current_run": CURRENT_RUN_THRESHOLD_CONST}
    updated = False
    for title, threshold in data.items():
        if avg_loss < best_losses[title]*threshold:
            best_losses[title] = avg_loss
            save_best_losses(best_losses)
            update_model(epoch, model, optimizer, avg_loss, title)
            if title == "strategy":
                updated = True

    return updated




def linear_mask_schedule(epoch, max_epoch, start_frac=0.1, end_frac=0.7): # originally 0.2 and 0.7
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
    
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch
    
def first_check(epoch, total_epochs):
    return epoch <= total_epochs

def second_check(epoch, total_epochs, has_saved):
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

    while first_check(epoch, total_epochs) or second_check(epoch, total_epochs, has_saved):
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
