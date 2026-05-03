import torch

# splits dataset images of 512x512 into 4 256x256 images
# on call of __getitem__()
class TiledDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset) * 4

    def __getitem__(self, idx):
        sample_idx = idx // 4   # which original image
        tile_idx   = idx % 4    # which of the 4 tiles (0-3)

        sample = self.dataset[sample_idx]
        image  = sample[0].float()
        mask   = sample[1].float()

        # tile positions: (row, col)
        positions = [(0,0), (0,1), (1,0), (1,1)]
        row, col  = positions[tile_idx]

        image = image[:, row*256:(row+1)*256, col*256:(col+1)*256]
        mask  =  mask[row*256:(row+1)*256, col*256:(col+1)*256]

        return image, mask.long()