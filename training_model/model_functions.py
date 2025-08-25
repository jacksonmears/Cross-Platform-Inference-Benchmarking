import torch
import os

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    
    print(f"Resuming from epoch {start_epoch}")
    return start_epoch


def update_model(epoch, model, optimizer, avg_loss, path):
    full_path = os.path.join("model", f"{path}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, full_path)
    print(f"Saved checkpoint to {full_path}")