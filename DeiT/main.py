import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchsummary import summary

from train import TrainModel
from model import DistilledVisionTransformer


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training DeiT', add_help=False)
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

    train_data = ImageFolder(
        root=path+'/train',
        transform=transforms_,
    )

    valid_data = ImageFolder(
        root=path+'/valid',
        transform=transforms_,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = DistilledVisionTransformer(num_classes=args.num_classes)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)