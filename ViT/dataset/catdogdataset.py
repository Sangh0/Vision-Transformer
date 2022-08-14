import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob

"""
path : data/
├── train_set
│    ├─dogs
│       ├─ dog1.jpg
│       ├─ ...
│    ├─cats
│       ├─ cat1.jpg 
│       ├─ ...
├── valid_set
│    ├─dogs
│       ├─ dog8.jpg
│       ├─ ...
│    ├─cats
│       ├─ cat8.jpg
│       ├─ ... 
├── test_set
│    ├─dogs
│       ├─ dog4001.jpg
│       ├─ ...
│    ├─cats
│       ├─ cat4001.jpg 
│       ├─ ...
"""

class CatDogDataset(Dataset):
    def __init__(self, 
                 path, 
                 subset, 
                 transforms_=None):
        assert subset in ('train', 'valid', 'test')
        self.cat_files = glob(path+subset+'_set/cats/*.jpg')
        self.dog_files = glob(path+subset+'_set/dogs/*.jpg')
        self.transforms_ = transforms_
        self.images = self.cat_files + self.dog_files
        self.labels = [0]*len(self.cat_files) + [1]*len(self.dog_files)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        images = Image.open(self.images[idx]).convert('RGB')
        labels = self.labels[idx]
        if self.transforms_ is not None:
            images = self.transforms_(images)
        return images, labels
    
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    path = 'C:/Users/user/MY_DL/classification/data/'
    batch_size = 32
    transforms_ = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    
    train_loader = DataLoader(
        CatDogDataset(path=path, subset='train', transforms_=transforms_),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    valid_loader = DataLoader(
        CatDogDataset(path=path, subset='valid', transforms_=transforms_),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )