import os
import json
from training_model.model_functions import update_model
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

best_losses_file = "best_losses.json"
ALL_TIME_THRESHOLD_CONST = 0.9
STRATEGY_THRESHOLD_CONST = 0.8
CURRENT_RUN_THRESHOLD_CONST = 1

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
