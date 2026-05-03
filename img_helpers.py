from PIL import Image
from pathlib import Path
from torchgeo.datasets import LandCoverAI
import matplotlib.pyplot as plt
import numpy as np

# tile 512x512 images into 4 256x256 images and save to disk
def save_tiling():
    # create output directories
    Path('data/tiles/images').mkdir(parents=True, exist_ok=True)
    Path('data/tiles/masks').mkdir(parents=True, exist_ok=True)

    base_dataset = LandCoverAI(root='data/', download=True)

    for idx in range(len(base_dataset)):
        sample = base_dataset[idx]
        image  = sample['image']   # (3, 512, 512)
        mask   = sample['mask']    # (512, 512)

        for tile_idx, (row, col) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            img_tile  = image[:, row*256:(row+1)*256, col*256:(col+1)*256]
            mask_tile = mask[row*256:(row+1)*256, col*256:(col+1)*256]

            Image.fromarray(img_tile.permute(1,2,0).byte().numpy()).save(f'data/tiles/images/{idx}_{tile_idx}.png')
            Image.fromarray(mask_tile.byte().numpy()).save(f'data/tiles/masks/{idx}_{tile_idx}.png')

        if idx % 100 == 0:
            print(f'{idx}/{len(base_dataset)} done')

    print('Tiling complete!')

# rescale dataset images to [0, 1] for clipping
def clippingRescale(img):
  return (img - img.min()) / (img.max() - img.min())

# view an nxn image
def preview_img(dataset, idx):
    # Permuting for image and mask visualization from [3, M, N] to [M, N, 3]
    img = dataset[idx][0].permute(1, 2, 0).numpy()
    msk = dataset[idx][1].numpy()
    print(img.shape)
    print(msk.shape)
    print(np.max(msk), np.min(msk))
    img = clippingRescale(img)
    # msk = clippingRescale(msk)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img)
    axes[0].axis('off')

    axes[1].imshow(msk)
    axes[1].axis('off')

    plt.show()

def plot_loss_miou(x, y1_miou, y2_loss):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(y1_miou)
    axes[0].set_title("Training Mean IoU")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(x, y2_loss, c="orange")
    axes[1].set_title("Training Loss")
    axes[1].set_xlabel("Epoch")
    
    plt.show()