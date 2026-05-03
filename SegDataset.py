from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# grabs tiled images and masks from disk
class SegDataset(Dataset):
    EXTS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg'}

    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir   = Path(img_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform
        self.imgs = sorted(p for p in self.img_dir.iterdir()
                           if p.suffix.lower() in self.EXTS)

    def __len__(self):
        return len(self.imgs)

    def _find_mask(self, img_path):
        for ext in self.EXTS:
            m = self.mask_dir / (img_path.stem + ext)
            if m.exists():
                return m
        raise FileNotFoundError(f'No mask for {img_path.name}')

    def __getitem__(self, idx):
        img_path  = self.imgs[idx]
        mask_path = self._find_mask(img_path)

        image = np.array(Image.open(img_path).convert('RGB'))
        mask  = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        image = torch.tensor(image).permute(2,0,1).float()
        mask  = torch.tensor(mask).long()

        return image, mask