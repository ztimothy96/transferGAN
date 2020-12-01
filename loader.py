import torch
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageFile

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://github.com/naoto0804/pytorch-AdaIN/blob/master/train.py

train_transform = transforms.Compose([
    transforms.Resize(size=(512, 512)),
    transforms.RandomCrop(256),
    transforms.ToTensor()])

class FlatFolderDataset(Dataset):
    def __init__(self, root, transform = None):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.paths= []
        for path, dirs, files in os.walk(root):
            for name in files:
                if name.endswith('.jpg'):
                    self.paths.append(os.path.join(path, name))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


if __name__ == '__main__':
    DATA_DIR = 'data0/lsun/bedroom/0/0/'



    dataset = FlatFolderDataset(DATA_DIR, train_transform)
    data_iter = iter(DataLoader(dataset))
    next(data_iter)
