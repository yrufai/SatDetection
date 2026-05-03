import torch

# wrapper class to remove mask/image dictionary in dataset
# allows for use of smp epochloader (no longer used)
class UnpackedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image  = sample['image'].float()
        mask   = sample['mask'].float()
        return image, mask.long()