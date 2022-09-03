import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob


class CustomDataset(Dataset):

    def __init__(
        self,
        path,
        subset,
        transforms_=None,
    ):
        assert subset in ('train', 'valid', 'test'), \
            'you should be choose between train, valid and test'
        self.class0_files = glob(path+'/'+subset+'/class0/*.jpg')
        self.class1_files = glob(path+'/'+subset+'/class1/*.jpg')
        self.transforms_ = transforms_
        self.images = self.class0_files + self.class1_files
        self.labels = [0] * len(self.class0_files) + [1] * len(self.class1_files)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = Image.open(self.images[idx]).convert('RGB')
        labels = self.labels[idx]
        if self.transforms_ is not None:
            images = self.transforms_(images)
        return images, labels