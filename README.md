# SatDetection

Multi-class semantic segmentation of aerial imagery using U-Net + ResNet-50.

**Course:** CSCI 4366/6366 — Neural Networks and Deep Learning, Spring 2026  
**Team:** Rufai Yakubu, Aaron Tyler  
**Affiliation:** The George Washington University

---

## Overview

Pixel-level land cover classification from high-resolution aerial imagery using the [LandCover.ai](https://landcover.ai/) dataset.

**Classes:**

| Class | Mask Value |
|------------|------------|
| Background | 0 |
| Building | 1 |
| Woodland | 2 |
| Water | 3 |
| Road | 4 |

---

## Dataset

- **Source:** LandCover.ai — aerial imagery from Poland, 25 cm/pixel resolution
- **Size:** 41 GeoTIFF images, auto-downloaded via `torchgeo`
- **Tiling:** Each 512×512 image is split into four 256×256 tiles → stored under `data/tiles/`

---

## Project Structure

```
SatDetection/
├── main.py              # Entry point — tiling, training, evaluation
├── SegDataset.py        # Dataset class (loads tiled images and masks)
├── SegModel.py          # Training loop, validation, testing, visualization
├── UNetResNet50.py      # Custom U-Net with ResNet-50 encoder
├── img_helpers.py       # Tiling, preview, and plotting utilities
├── scr.py               # Scripts / scratch
├── test.py              # Standalone test script
├── data/
│   └── tiles/
│       ├── images/      # Tiled RGB patches (.png)
│       └── masks/       # Tiled label masks (.png)
├── out/                 # Output visualizations
├── models/              # Saved model checkpoints (.pt)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

On first run this will:
1. Download the LandCover.ai dataset via `torchgeo`
2. Tile images into 256×256 patches under `data/tiles/`
3. Train the model and save the best checkpoint to `models/`
4. Run evaluation and display a sample prediction

**Key flags in `main.py`:**

| Flag | Default | Description |
|------|---------|-------------|
| `load_model` | `False` | Load a saved model instead of training |
| `custom_unet` | `False` | Use custom `UNetResNet50` instead of `smp.Unet` |
| `weights` | `'imagenet'` | Encoder weights (`'imagenet'` or `None`) |
| `model_name` | `'best_model_untrained'` | Name used for saving/loading |

---

## Model

Two model options are supported:

- **`smp.Unet`** (default) — U-Net with ResNet-50 encoder from `segmentation-models-pytorch`
- **`UNetResNet50`** — custom U-Net with ResNet-50 backbone built with `torchvision`

Both support optional ImageNet pretrained weights on the encoder.

**Architecture summary:**
- Encoder: ResNet-50 (5 stages)
- Decoder: transposed convolutions with skip connections
- Head: 1×1 conv → 5-class logits
- Loss: Dice loss (multiclass)
- Optimizer: Adam (lr=3e-4)
- Metric: Mean IoU

---

## Results

Best checkpoint saved to `models/` based on validation mIoU. Sample predictions are visualized with per-class color overlays:

| Class | Color |
|------------|-------|
| Background | Black |
| Building | Red |
| Woodland | Green |
| Water | Blue |
| Road | Gray |
