# Swin Transformer Implementation  
### paper link : https://arxiv.org/abs/2103.14030  
### [Paper Review](https://github.com/Sangh0/Vision-Transformer/tree/main/SwinTransformer/swin_transformer_paper_review.ipynb)
### Swin Transformer Architecture  
<img src = "https://github.com/Sangh0/Vision-Transformer/blob/main/SwinTransformer/figure/figure3.JPG?raw=true">

### Code Implementation Reference: [Official Github](https://github.com/microsoft/Swin-Transformer)

## Train
```
usage: main.py [-h] [--use_pretrained USE_PRETRAINED] [--data_dir DATA_DIR] [--lr LR] [--epochs EPOCHS] \
               [--batch_size BATCH_SIZE] [--weight_decay WEIGHT_DECAY] [--num_classes NUM_CLASSES] \
               [--img_size IMG_SIZE] [--lr_scheduling LR_SCHEDULING] [--check_point CHECK_POINT] \
               [--early_stop EARLY_STOP] [--train_log_step TRAIN_LOG_STEP] [--valid_log_step VALID_LOG_STEP]

example: python main.py --data_dir ./dataset --lr 1e-3 --num_classes 2 --img_size 224
```

## Evaluate
```
usage: evaluate.py [-h] [--data_dir DATA_DIR] [--weight WEIGHT] [--num_classes NUM_CLASS] [--img_size IMG_SIZE]

example: python evaluate.py --data_dir ./dataset --weight ./weights/best_weight.pt --num_classes 2
```

## Run on Jupyter Notebook for training model
```python
# Load Packages
import torchvision.transforms as transforms
from torchsummary import summary

from train import TrainModel
from util.dataset import CustomDataset

# Set hyperparameters
Config = {
    'use_pretrained': False,
    'data_dir': './dataset',
    'lr': 1e-3,
    'weight_decay': 5e-2,
    'batch_size': 32,
    'epochs': 300,
    'num_classes': 2,
    'img_size': 224,
    'lr_scheduling': True,
    'check_point': True,
    'early_stop': False,
    'train_log_step': 40,
    'valid_log_step': 20,
}

# Load Datasets
transforms_ = transforms.Compose([
    transforms.Resize((Config['img_size'], Config['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

train_loader = DataLoader(
    CustomDataset(path=Config['data_dir'], subset='train', transforms_=transforms_),
    batch_size=Config['batch_size'],
    shuffle=True,
    drop_last=True,
)

valid_loader = DataLoader(
    CustomDataset(path=Config['data_dir'], subset='valid', transforms_=transforms_),
    batch_size=Config['batch_size'],
    shuffle=True,
    drop_last=True,
)

# Load Swin Transformer
if Config['use_pretrained']:
    from pretrained_model import get_pretrained_model
    swin_transformer = get_pretrained_model(num_classes=Config['num_classes'], pretrained=True)

else:
    from model import SwinTransformer
    model_config = {
        'img_size': Config['img_size'],
        'patch_size': 4,
        'in_dim': 3,
        'num_classes': Config['num_classes'],
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
        'mlp_ratio': 4.,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.1,
        'drop_path_rate': 0.1,
        'norm_layer': nn.LayerNorm,
        'ape': False,
        'patch_norm': True,
    }
    swin_transformer = SwinTransformer(**model_config)

# Check summary of model
summary(swin_transformer, (3, Config['img_size'], Config['img_size']), device='cpu')

# Training model
model = TrainModel(
    model=swin_transformer,
    lr=Config['lr'],
    epochs=Config['epochs'],
    weight_decay=Config['weight_decay'],
    lr_scheduling=Config['lr_scheduling'],
    check_point=Config['check_point'],
    early_stop=Config['early_stop'],
    train_log_step=Config['train_log_step'],
    valid_log_step=Config['valid_log_step'],
)

history = model.fit(train_loader, valid_loader)
```

## Run on Jupyter Notebook to evaluate model for test set
```python
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import SwinTransformer
from util.dataset import CustomDataset
from evaluate import eval

Config = {
    'data_dir': './dataset',
    'weight': './weights/best_weight.pt',
    'num_classes': 2,
    'img_size': 224,
}

transforms_ = transforms.Compose([
    transforms.Resize((Config['img_size'], Config['img_size'])),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_loader = DataLoader(
    CustomDataset(path=Config['data_dir'], subset='test', transforms_=transforms_),
    batch_size=1,
    shuffle=False,
    drop_last=False,
)

model_config = {
    'use_pretrained': False,
    'img_size': Config['img_size'],
    'patch_size': 4,
    'in_dim': 3,
    'num_classes': Config['num_classes'],
    'embed_dim': 96,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': 7,
    'mlp_ratio': 4.,
    'qkv_bias': True,
    'qk_scale': None,
    'drop_rate': 0.1,
    'attn_drop_rate': 0.1,
    'drop_path_rate': 0.1,
    'norm_layer': nn.LayerNorm,
    'ape': False,
    'patch_norm': True,
}

model = SwinTransformer(**model_config)
model.load_state_dict(torch.load(Config['weight']))

result = eval(model, test_loader)
```