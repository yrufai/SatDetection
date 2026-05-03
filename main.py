from SegDataset import SegDataset
import segmentation_models_pytorch as smp
from SegModel import SegModel
import img_helpers
import torch

img_helpers.save_tiling()

# load LandCover.ai dataset
dataset = SegDataset(
    img_dir='data/tiles/images',
    mask_dir='data/tiles/masks',
)


model = None
load_model = False
weights = 'imagenet'

if load_model:
    model_name = 'best_model_untrained'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(f'models/{model_name}.pt', map_location=DEVICE, weights_only=False)
else:
    # --- Model ---
    model = smp.Unet(
        encoder_name='resnet50',
        encoder_weights=weights,
        in_channels=3,
        classes=5,
    )
seg_model = SegModel(EPOCHS=30, BATCH_SIZE=32, NUM_CLASSES=5, 
                     model=model, dataset=dataset)

# seg_model.train(train_split=0.15, val_split=0.15, saved_model_name='best_model_untrained')
seg_model.test()
seg_model.load_example(90)