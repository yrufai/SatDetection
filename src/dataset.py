import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


LANDCOVERAI_STATS = {
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
}


def get_augmentation_pipeline(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ElasticTransform(alpha=120, sigma=6, p=0.3),
        ])
    return None


class SatDataset(Dataset):
    """
    PyTorch Dataset for 256x256 pre-tiled LandCover.ai patches.

    Expects:
        root/images/tile_XXXXX.npy  — (C, H, W) float32
        root/masks/tile_XXXXX.npy   — (2, H, W) float32 [buildings, roads]
    """

    def __init__(self, root, train=True):
        self.root = root
        self.train = train

        image_dir = os.path.join(root, "images")

        if not os.path.exists(image_dir):
            raise FileNotFoundError(
                f"Tiles directory not found: {image_dir}\n"
                f"Run the tiling step in Section 6 of the notebook first."
            )

        self.image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".npy")
        ])

        if len(self.image_files) == 0:
            raise ValueError(
                f"No .npy tiles found in: {image_dir}\n"
                f"The tiling step may have failed or saved files to a different path.\n"
                f"Check that tiles exist at: {image_dir}"
            )

        self.augment = get_augmentation_pipeline(train)
        self.normalize = A.Normalize(
            mean=LANDCOVERAI_STATS["mean"],
            std=LANDCOVERAI_STATS["std"]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        image = np.load(os.path.join(self.root, "images", fname))  # (C, H, W)
        mask  = np.load(os.path.join(self.root, "masks",  fname))  # (2, H, W)

        # albumentations expects (H, W, C)
        image = np.transpose(image[:3], (1, 2, 0)).astype(np.float32)
        mask  = np.transpose(mask, (1, 2, 0))  # (H, W, 2)

        if image.max() > 1.0:
            image = image / 255.0

        if self.augment is not None:
            augmented = self.augment(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = self.normalize(image=image)["image"]

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask  = torch.from_numpy(np.transpose(mask,  (2, 0, 1))).float()

        return image, mask


def build_dataloaders(config):
    """
    Build train/val/test DataLoaders from LandCover.ai tiles.
    Split: 70% train / 15% val / 15% test.

    Uses three separate dataset instances so setting val augment=False
    does not affect the train instance (they would share state otherwise).
    """
    from torch.utils.data import Subset

    root        = config["landcoverai_root"]
    batch_size  = config["batch_size"]
    num_workers = config.get("num_workers", 2)

    n = len(SatDataset(root=root, train=False))
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    indices   = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_ds = Subset(SatDataset(root=root, train=True),  train_idx)
    val_ds   = Subset(SatDataset(root=root, train=False), val_idx)
    test_ds  = Subset(SatDataset(root=root, train=False), test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Dataset split — Train: {n_train} | Val: {n_val} | Test: {n_test}")
    return train_loader, val_loader, test_loader
