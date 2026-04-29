import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from model import build_model
from dataset import build_dataloaders
from evaluate import compute_metrics


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Loss functions
# --------------------------------------------------------------------------- #

def dice_loss(pred, target, smooth=1.0):
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt  = torch.exp(-bce)
    return (alpha * (1 - pt) ** gamma * bce).mean()


def combined_loss(pred, target, w_building=1.0, w_road=2.0):
    """
    Buildings (ch 0): BCE + Dice
    Roads     (ch 1): Focal + Dice
    """
    pred_b, pred_r = pred[:, 0], pred[:, 1]
    tgt_b,  tgt_r  = target[:, 0], target[:, 1]

    loss_building = F.binary_cross_entropy(pred_b, tgt_b) + dice_loss(pred_b, tgt_b)
    loss_road     = focal_loss(pred_r, tgt_r) + dice_loss(pred_r, tgt_r)

    return w_building * loss_building + w_road * loss_road


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, optimizer, device, w_building, w_road):
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss  = combined_loss(preds, masks, w_building=w_building, w_road=w_road)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, w_building, w_road):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        preds = model(images)
        loss  = combined_loss(preds, masks, w_building=w_building, w_road=w_road)
        total_loss += loss.item()

        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics     = compute_metrics(all_preds, all_targets)

    return total_loss / len(loader), metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def train(config):
    device = get_device()
    print(f"Using device: {device}")

    run_name = "pretrained" if config["pretrained"] else "random_init"

    wandb.init(
        project=config.get("wandb_project", "SatDetection"),
        name=run_name,
        config=config,
    )

    train_loader, val_loader, _ = build_dataloaders(config)
    model = build_model(pretrained=config["pretrained"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5, verbose=True
    )

    best_iou        = 0.0
    patience_counter = 0
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["checkpoint_dir"], f"{run_name}_best.pth")

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            w_building=config["w_building"],
            w_road=config["w_road"],
        )
        val_loss, metrics = validate(
            model, val_loader, device,
            w_building=config["w_building"],
            w_road=config["w_road"],
        )

        mean_iou = (metrics["iou_building"] + metrics["iou_road"]) / 2

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"IoU Building: {metrics['iou_building']:.4f} | "
            f"IoU Road: {metrics['iou_road']:.4f} | "
            f"Mean IoU: {mean_iou:.4f}"
        )

        wandb.log({
            "epoch":        epoch,
            "train_loss":   train_loss,
            "val_loss":     val_loss,
            "iou_building": metrics["iou_building"],
            "iou_road":     metrics["iou_road"],
            "mean_iou":     mean_iou,
            "f1_building":  metrics["f1_building"],
            "f1_road":      metrics["f1_road"],
            "lr":           optimizer.param_groups[0]["lr"],
        })

        scheduler.step(mean_iou)

        if mean_iou > best_iou:
            best_iou         = mean_iou
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best checkpoint (mean IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch}.")
                break

    wandb.finish()
    print(f"Training complete. Best mean IoU: {best_iou:.4f}")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str, default="configs/train_config.yaml")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.pretrained:
        config["pretrained"] = True

    train(config)
