from SegDataset import SegDataset
import segmentation_models_pytorch as smp
from SegModel import SegModel
import img_helpers
import torch
from UNetResNet50 import UNetResNet50

# road = 4, water = 3, woodlands = 2, building = 1, background = 0
img_helpers.save_tiling()

# load LandCover.ai dataset
dataset = SegDataset(
    img_dir='data/tiles/images',
    mask_dir='data/tiles/masks',
)

for i in range(2, 15):
    img_helpers.preview_img(dataset, i)

model = None
load_model = False
custom_unet = False
model_name = 'best_model_untrained'
weights = 'imagenet'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if load_model:
    model = torch.load(f'models/{model_name}.pt', map_location=DEVICE, weights_only=False)
else:
    # --- Model ---
    if custom_unet:
        if weights:
            model = UNetResNet50(num_classes=5, pretrained=True)
        else:
            model = UNetResNet50(num_classes=5, pretrained=False)
    else:
        model = smp.Unet(
            encoder_name='resnet50',
            encoder_weights=weights,
            in_channels=3,
            classes=5,
        )

seg_model = SegModel(EPOCHS=30, BATCH_SIZE=32, NUM_CLASSES=5, 
                     model=model, dataset=dataset, DEVICE=DEVICE)

seg_model.train(train_split=0.15, val_split=0.15, saved_model_name='best_model_untrained')
seg_model.test()
seg_model.load_example(90)

