import torch
import os
import random

from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://github.com/naoto0804/pytorch-AdaIN/blob/master/train.py

class FlatFolderDataset(Dataset):
    def __init__(self, root, transform = None, n_examples = None):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.paths= []
        for path, dirs, files in os.walk(root):
            for name in files:
                if name.endswith('.jpg'):
                    self.paths.append(os.path.join(path, name))
                    
        if n_examples:
            if n_examples > len(self.paths):
                raise ValueError('only {} examples in folder: {} needed'.format(
                    len(self.paths), n_examples))
            # grab a random subset of size n_examples
            random.shuffle(self.paths)
            self.paths = self.paths[:n_examples]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

