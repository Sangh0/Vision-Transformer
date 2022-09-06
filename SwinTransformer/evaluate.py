import argparse
import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import SwinTransformer
from util.dataset import CustomDataset

@torch.no_grad()
def eval(model, dataset, loss_func=nn.CrossEntropyLoss()):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    batch_loss, batch_acc = 0, 0
    pbar = tqdm(enumerate(dataset), total=len(dataset))
    
    start = time.time()
    for batch, (images, labels) in pbar:
        images, labels = images.to(device), labels.to(device)
    
        outputs = model(images)
        loss = loss_func(outputs, labels)
        output_index = torch.argmax(outputs, dim=1)
        acc = (output_index==labels).sum()/len(outputs)

        batch_loss += loss.item()
        batch_acc += acc.item()

        del images; del labels; del outputs
        torch.cuda.empty_cache()

    end = time.time()

    print(f'\nTotal time for testing is {end-start:.2f}s')
    print(f'\nAverage loss: {batch_loss/(batch+1):.3f}  accuracy: {batch_acc/(batch+1):.3f}')
    return {
        'loss': batch_loss/(batch+1),
        'accuracy': batch_acc/(batch+1),
    }


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training PIDNet', add_help=False)
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory where your dataset is located')
    parser.add_argument('--weight', type=str, required=True,
                        help='load weight file of trained model')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='class number of dataset')
    parser.add_argument('--img_size', type=int, default=224,
                        help='image size used when training')
    return parser


def main(args):

    transforms_ = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_loader = DataLoader(
        CustomDataset(path=args.data_dir, subset='test', transforms_=transforms_),
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    model_config = {
        'img_size': args.img_size,
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
        'use_checkpoint': False,
    }

    model = SwinTransformer(**model_config)
    model.load_state_dict(torch.load(args.weight))

    result = eval(model, test_loader)