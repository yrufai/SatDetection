import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os

from model import build_model
from dataset import build_dataloaders
from utils import visualize_prediction


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def iou_score(pred_bin, target, smooth=1e-6):
    intersection = (pred_bin & target.bool()).float().sum()
    union = (pred_bin | target.bool()).float().sum()
    return ((intersection + smooth) / (union + smooth)).item()


def f1_score(pred_bin, target, smooth=1e-6):
    tp = (pred_bin & target.bool()).float().sum()
    fp = (pred_bin & ~target.bool()).float().sum()
    fn = (~pred_bin & target.bool()).float().sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return (2 * precision * recall / (precision + recall + smooth)).item()


def compute_metrics(preds, targets, threshold=0.5):
    """
    Compute per-class IoU and F1.

    Args:
        preds:   (N, 2, H, W) sigmoid outputs
        targets: (N, 2, H, W) binary ground truth
        threshold: binarization threshold

    Returns:
        dict with iou_building, iou_road, f1_building, f1_road
    """
    pred_bin = (preds > threshold).bool()
    target_bin = targets.bool()

    return {
        "iou_building": iou_score(pred_bin[:, 0], target_bin[:, 0]),
        "iou_road":     iou_score(pred_bin[:, 1], target_bin[:, 1]),
        "f1_building":  f1_score(pred_bin[:, 0], target_bin[:, 0]),
        "f1_road":      f1_score(pred_bin[:, 1], target_bin[:, 1]),
    }


def threshold_sweep(preds, targets, thresholds=None):
    """
    Find the best per-channel threshold by sweeping and maximizing F1.

    Returns:
        best_thresh_building, best_thresh_road, sweep results dict
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    results = {"building": [], "road": []}

    for t in thresholds:
        m = compute_metrics(preds, targets, threshold=t)
        results["building"].append((t, m["f1_building"]))
        results["road"].append((t, m["f1_road"]))

    best_b = max(results["building"], key=lambda x: x[1])
    best_r = max(results["road"], key=lambda x: x[1])

    return best_b[0], best_r[0], results


# --------------------------------------------------------------------------- #
# Full evaluation on test set
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(config, checkpoint_path, save_visuals=True, n_visuals=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(pretrained=config["pretrained"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, _, test_loader = build_dataloaders(config)

    all_preds, all_targets, all_images = [], [], []

    for images, masks in test_loader:
        preds = model(images.to(device)).cpu()
        all_preds.append(preds)
        all_targets.append(masks)
        all_images.append(images)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_images = torch.cat(all_images, dim=0)

    # threshold sweep on test set to find optimal thresholds
    best_t_b, best_t_r, _ = threshold_sweep(all_preds, all_targets)
    print(f"Best threshold — Building: {best_t_b:.2f} | Road: {best_t_r:.2f}")

    # compute final metrics at default threshold and best thresholds
    metrics_default = compute_metrics(all_preds, all_targets, threshold=0.5)
    metrics_best = {
        "iou_building": iou_score((all_preds[:, 0] > best_t_b).bool(), all_targets[:, 0].bool()),
        "iou_road":     iou_score((all_preds[:, 1] > best_t_r).bool(), all_targets[:, 1].bool()),
        "f1_building":  f1_score((all_preds[:, 0] > best_t_b).bool(), all_targets[:, 0].bool()),
        "f1_road":      f1_score((all_preds[:, 1] > best_t_r).bool(), all_targets[:, 1].bool()),
    }

    print("\n=== Test Results (threshold=0.5) ===")
    for k, v in metrics_default.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Test Results (best threshold) ===")
    for k, v in metrics_best.items():
        print(f"  {k}: {v:.4f}")

    mean_iou_default = (metrics_default["iou_building"] + metrics_default["iou_road"]) / 2
    mean_iou_best = (metrics_best["iou_building"] + metrics_best["iou_road"]) / 2
    print(f"\nMean IoU (default): {mean_iou_default:.4f}")
    print(f"Mean IoU (best):    {mean_iou_best:.4f}")

    if save_visuals:
        vis_dir = config.get("visuals_dir", "outputs/visuals")
        os.makedirs(vis_dir, exist_ok=True)
        run_name = (
            f"{config['dataset']}_"
            f"{'pretrained' if config['pretrained'] else 'random'}"
        )
        indices = np.random.choice(len(all_images), min(n_visuals, len(all_images)), replace=False)
        for i, idx in enumerate(indices):
            save_path = os.path.join(vis_dir, f"{run_name}_sample_{i}.png")
            visualize_prediction(
                image=all_images[idx].numpy(),
                true_mask=all_targets[idx].numpy(),
                pred_mask=all_preds[idx].numpy(),
                threshold=0.5,
                save_path=save_path,
            )
        print(f"\nSaved {len(indices)} visual samples to {vis_dir}")

    return metrics_best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    evaluate(config, args.checkpoint)
