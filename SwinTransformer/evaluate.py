import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn

@torch.no_grad()
def eval(model, dataset, loss_func=nn.CrossEntropyLoss()):
    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    batch_loss, batch_acc = 0, 0
    pbar = tqdm(enumerate(dataset), total=len(dataset))
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