import os
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# --------------------------------------------------------------------------- #
# Tiling: LandCover.ai GeoTIFF → 256×256 patches
# --------------------------------------------------------------------------- #

def tile_landcoverai(image_path, mask_path, out_dir, tile_size=256, stride=256):
    """
    Tile a LandCover.ai image + mask pair into fixed-size patches.

    LandCover.ai masks are single-channel with pixel values:
      0 = background, 1 = building, 2 = road

    Converts to 2-channel float mask: [buildings, roads].
    Skips tiles that are entirely background.
    """
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "masks"),  exist_ok=True)

    with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as msk_src:
        height, width = img_src.height, img_src.width
        count = len(os.listdir(os.path.join(out_dir, "images")))  # continue numbering

        for row in range(0, height - tile_size + 1, stride):
            for col in range(0, width - tile_size + 1, stride):
                window = Window(col, row, tile_size, tile_size)

                image_tile = img_src.read(window=window)       # (C, H, W)
                raw_mask   = msk_src.read(1, window=window)    # (H, W)

                if image_tile.shape[1] != tile_size or image_tile.shape[2] != tile_size:
                    continue

                building_tile = (raw_mask == 1).astype(np.float32)
                road_tile     = (raw_mask == 2).astype(np.float32)

                if building_tile.max() == 0 and road_tile.max() == 0:
                    continue

                mask_tile = np.stack([building_tile, road_tile], axis=0)

                np.save(os.path.join(out_dir, "images", f"tile_{count:05d}.npy"), image_tile)
                np.save(os.path.join(out_dir, "masks",  f"tile_{count:05d}.npy"), mask_tile)
                count += 1

    return count


def tile_all(image_dir, mask_dir, out_dir, tile_size=256, stride=256):
    """
    Tile all LandCover.ai GeoTIFF pairs.

    LandCover.ai naming convention:
      image: M-33-20-C-c-3-4.tif
      mask:  M-33-20-C-c-3-4_m.tif

    Supports two layouts:
      Layout A — separate dirs: image_dir/X.tif + mask_dir/X_m.tif
      Layout B — same dir:      image_dir/X.tif + image_dir/X_m.tif
    """
    import glob

    all_tifs    = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    image_files = [f for f in all_tifs if not f.endswith("_m.tif")]

    if len(image_files) == 0:
        raise FileNotFoundError(
            f"No image .tif files found in: {image_dir}\n"
            f"Make sure LandCover.ai images are uploaded to Drive."
        )

    pairs = []
    for img_path in image_files:
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = f"{stem}_m.tif"

        # try mask_dir first, then same dir as image
        msk_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(msk_path):
            msk_path = os.path.join(image_dir, mask_name)
        if not os.path.exists(msk_path):
            print(f"  Warning: no mask found for {stem}, skipping")
            continue
        pairs.append((img_path, msk_path))

    print(f"Found {len(pairs)} image/mask pairs")

    for img_path, msk_path in pairs:
        n = tile_landcoverai(img_path, msk_path, out_dir, tile_size=tile_size, stride=stride)
        print(f"  {os.path.basename(img_path)} -> {n} total tiles so far")

    total = len(os.listdir(os.path.join(out_dir, "images")))
    print(f"\nTotal tiles saved: {total}")
    return total


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #

def visualize_prediction(image, true_mask, pred_mask, threshold=0.5, save_path=None):
    """
    Overlay predicted 2-channel mask on the input image.

    Args:
        image:     (C, H, W) numpy array
        true_mask: (2, H, W) ground truth — [buildings, roads]
        pred_mask: (2, H, W) sigmoid model output
        threshold: binarization threshold
        save_path: if provided, save figure instead of showing
    """
    rgb = np.transpose(image[:3], (1, 2, 0)).astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0, 1)

    pred_building = (pred_mask[0] > threshold)
    pred_road     = (pred_mask[1] > threshold)
    true_building = true_mask[0] > 0
    true_road     = true_mask[1] > 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    gt_overlay = rgb.copy()
    gt_overlay[true_building] = [1.0, 0.3, 0.3]
    gt_overlay[true_road]     = [0.3, 0.3, 1.0]
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    pred_overlay = rgb.copy()
    pred_overlay[pred_building] = [1.0, 0.3, 0.3]
    pred_overlay[pred_road]     = [0.3, 0.3, 1.0]
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    building_patch = mpatches.Patch(color=(1.0, 0.3, 0.3), label="Building")
    road_patch     = mpatches.Patch(color=(0.3, 0.3, 1.0), label="Road")
    fig.legend(handles=[building_patch, road_patch], loc="lower center", ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()
