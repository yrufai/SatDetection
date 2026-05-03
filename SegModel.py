import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchmetrics.segmentation import MeanIoU
from torch.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


class SegModel():
    def __init__(self, EPOCHS, BATCH_SIZE, NUM_CLASSES, model, dataset):
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_CLASSES = NUM_CLASSES
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset = dataset
        self.model = model
        self.model.to(self.DEVICE)
    
    def train(self, train_split, val_split, saved_model_name):
        Path('models').mkdir(parents=True, exist_ok=True)
        model_save_path = f"models/{saved_model_name}.pt"

        # split data into train, validation, and test sets
        n = len(self.dataset)
        n_train = int(train_split * n)
        n_val = int(val_split * n)
        n_test = n - n_train - n_val

        train_ds, val_ds, self.test_ds = random_split(self.dataset, [n_train, n_val, n_test])

        train_loader = DataLoader(train_ds, batch_size=self.BATCH_SIZE, num_workers=4, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.BATCH_SIZE, num_workers=4, shuffle=False)

        # --- Loss, metric, optimizer ---
        self.criterion     = smp.losses.DiceLoss(mode='multiclass')
        self.criterion.__name__ = 'dice_loss'
        self.metric = MeanIoU(num_classes=self.NUM_CLASSES).to(self.DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        scaler = GradScaler()

        # --- Training loop ---
        best_miou = 0.0
        for epoch in range(1, self.EPOCHS + 1):
            self.model.train()
            total_loss = 0.0
            loop = tqdm(train_loader, desc=f'Epoch {epoch}/{self.EPOCHS}')
            for images, masks in loop:
                images, masks = images.to(self.DEVICE), masks.to(self.DEVICE)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type='cuda'):
                    logits = self.model(images)
                    loss   = self.criterion(logits, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                loop.set_postfix(loss=f'{loss.item():.4f}')

            # validate
            self.model.eval()
            self.metric.reset()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(self.DEVICE), masks.to(self.DEVICE)
                    with autocast(device_type='cuda'):
                        logits = self.model(images)
                        val_loss += self.criterion(logits, masks).item()
                    self.metric.update(logits.argmax(dim=1), masks)

            val_miou = self.metric.compute().item()
            print(f'Epoch {epoch} | Loss {total_loss/len(train_loader):.4f} | Val Loss {val_loss/len(val_loader):.4f} | Val mIoU {val_miou:.4f}')

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(self.model, model_save_path)
                print('Model saved!')

        print(f'Best mIoU: {best_miou:.4f}')
    
    def test(self):
        self.model.eval()
        self.metric.reset()
        test_loss = 0.0
        test_loader  = DataLoader(self.test_ds,  batch_size=self.BATCH_SIZE, num_workers=4, shuffle=False)

        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc='Testing'):
                images, masks = images.to(self.DEVICE), masks.to(self.DEVICE)
                with autocast(device_type='cuda'):
                    logits = self.model(images)
                    test_loss += self.criterion(logits, masks).item()
                self.metric.update(logits.argmax(dim=1), masks)

        test_miou = self.metric.compute().item()
        print(f'Test Loss {test_loss/len(test_loader):.4f} | Test mIoU {test_miou:.4f}')

    def load_example(self, idx):
        CLASS_COLORS = np.array([
            [0,   0,   0  ],   # background
            [255, 0,   0  ],   # building
            [0,   128, 0  ],   # woodland
            [0,   0,   255],   # water
            [128, 128, 128],   # road
        ], dtype=np.uint8)

        # load indexed example from dataset
        image, mask = self.dataset[idx]

        # calculate model prediction
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image.unsqueeze(0).to(self.DEVICE))
            pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        # color masks
        pred_rgb = CLASS_COLORS[pred]
        true_rgb = CLASS_COLORS[mask.numpy()]

        # plot/display results
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(image.permute(1,2,0) / 255)
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(true_rgb)
        axes[1].set_title('Ground truth')
        axes[1].axis('off')

        axes[2].imshow(pred_rgb)
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()



# --- Training loop ---
# import time
# data_time, gpu_time = 0, 0
# t = time.time()
# for images, masks in train_loader:
#     data_time += time.time() - t
#     images, masks = images.to(DEVICE), masks.to(DEVICE)
#     t = time.time()
#     with autocast(device_type='cuda'):
#         logits = model(images)
#         loss = criterion(logits, masks)
#     loss.backward()
#     gpu_time += time.time() - t
#     t = time.time()
#     break

# print(f'Data loading: {data_time:.2f}s | GPU: {gpu_time:.2f}s')