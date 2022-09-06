import argparse
from webbrowser import get

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary

from train import TrainModel
from util.dataset import CustomDataset

def get_args_parser():
    parser = argparse.ArgumentParser(description='Training PIDNet', add_help=False)
    parser.add_argument('--use_pretrained', type=bool, default=False,
                        help='use pre-trained weight of swin transformer in timm package')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory where your dataset is located')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--epochs', default=300, type=int,
                        help='Epochs for training model')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch Size for training model')
    parser.add_argument('--weight_decay', default=5e-2, type=float,
                        help='weight decay of optimizer SGD')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='class number of dataset')
    parser.add_argument('--img_size', type=int, default=224,
                        help='image size')
    parser.add_argument('--lr_scheduling', default=True, type=bool,
                        help='apply learning rate scheduler')
    parser.add_argument('--check_point', default=True, type=bool,
                        help='apply check point for saving weights of model')
    parser.add_argument('--early_stop', default=False, type=bool,
                        help='apply early stopping')
    parser.add_argument('--train_log_step', default=40, type=int,
                        help='print log of training phase')
    parser.add_argument('--valid_log_step', default=20, type=int,
                        help='print log of validating phase')
    return

def main(args):
    
    path = args.data_dir
    batch_size = args.batch_size
    img_size = args.img_size

    transforms_ = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_loader = DataLoader(
        CustomDataset(path=path, subset='train', transforms_=transforms_),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        CustomDataset(path=path, subset='valid', transforms_=transforms_),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    if args.use_pretrained:

        from pretrained_model import get_pretrained_model
        model = get_pretrained_model()

    else:

        from model import SwinTransformer
        model_config = {
            'img_size': img_size,
            'patch_size': 4,
            'in_dim': 3,
            'num_classes': args.num_classes,
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

    summary(model, (3, img_size, img_size), device='cpu')

    model = TrainModel(
        model=model,
        lr=args.lr,
        epochs=args.epochs,
        lr_scheduling=args.lr_scheduling,
        check_point=args.check_point,
        early_stop=args.early_stop,
        train_log_step=args.train_log_step,
        valid_log_step=args.valid_log_step,
    )

    history = model.fit(train_loader, valid_loader)