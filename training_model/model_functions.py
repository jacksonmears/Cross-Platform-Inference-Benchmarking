import torch
import os

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    
    print(f"Resuming from epoch {start_epoch}")
    
    return start_epoch

# had to make some adjustments because windows throws a fit if you access a file too often in a short span :(
def update_model(epoch, model, optimizer, avg_loss, path):
    os.makedirs("model", exist_ok=True)

    final_path = os.path.join("model", f"{path}.pth")
    tmp_path = final_path + ".tmp"  # temporary file for safe writing

    # Save to temporary file first
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, tmp_path)

    # Atomically replace the old checkpoint with the new one
    os.replace(tmp_path, final_path)

    print(f"Saved checkpoint to {final_path}")
