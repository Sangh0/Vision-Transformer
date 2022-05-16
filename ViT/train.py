import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

from torchcallback import EarlyStopping, CheckPoint
from torchscheduler import PolynomialLRDecay
from model import VisionTransformer

lr = 1e-3
EPOCH = 500
device = torch.device('cuda')

model_config = {
    'img_size': 224,
    'in_chans': 3,
    'n_classes': 2,
    'patch_size': 16,
    'embed_dim': 768,
    'depth': 12,
    'n_heads': 12,
    'qkv_bias': True,
    'mlp_ratio': 4,
}

model = VisionTransformer(**model_config).to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.1)

es_save_path = './model/es_checkpoint.pt'
cp_save_path = './model/cp_checkpoint.pt'
checkpoint = CheckPoint(verbose=True, path=cp_save_path)
early_stopping = EarlyStopping(patience=20, verbose=True, path=es_save_path)
lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=EPOCH)

def valid_step(model,
               validation_data):
    model.eval()
    with torch.no_grad():
        vbatch_loss, vbatch_acc = 0, 0
        for vbatch, (val_images, val_labels) in enumerate(validation_data):
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            
            val_outputs = model(val_images)
            val_loss = loss_func(val_outputs, val_labels)
            output_index = torch.argmax(val_outputs, dim=1)
            val_acc = (output_index==val_labels).sum()/batch_size
            
            vbatch_loss += val_loss.item()
            vbatch_acc += val_acc.item()
            
            del val_images; del val_labels; del val_outputs
            torch.cuda.empty_cache()
            
    return vbatch_loss/(vbatch+1), vbatch_acc/(vbatch+1)

def train_step(model,
               train_data,
               validation_data,
               epochs,
               learning_rate_scheduler=False,
               check_point=False,
               early_stop=False,
               last_epoch_save_path='./model/last_checkpoint.pt'):
    
    loss_list, acc_list = [], []
    val_loss_list, val_acc_list = [], []
    
    print('Start Model Training...!')
    start_training = time.time()
    for epoch in tqdm(range(epochs)):
        init_time = time.time()
        batch_loss, batch_acc = 0, 0
        for batch, (train_images, train_labels) in enumerate(train_data):
            model.train()
            train_images, train_labels = train_images.to(device), train_labels.to(device)
            
            optimizer.zero_grad()
            
            train_outputs = model(train_images)
            loss = loss_func(train_outputs, train_labels)
            output_index = torch.argmax(train_outputs, dim=1)
            acc = (output_index==train_labels).sum()/batch_size
            
            batch_loss += loss.item()
            batch_acc += acc.item()
            
            loss.backward()
            optimizer.step()
            
            del train_images; del train_labels; del train_outputs
            torch.cuda.empty_cache()
        
        loss_list.append(batch_loss/(batch+1))
        acc_list.append(batch_acc/(batch+1))
            
        val_loss, val_acc = valid_step(model, validation_data)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        
        end_time = time.time()
        
        print(f'\n[Epoch {epoch+1}/{epochs}]'
              f'  [time: {end_time-init_time:.3f}s]'
              f'  [lr = {optimizer.param_groups[0]["lr"]}]')
        print(f'[train loss: {batch_loss/(batch+1):.3f}]'
              f'  [train acc: {batch_acc/(batch+1):.3f}]'
              f'  [valid loss: {val_loss:.3f}]'
              f'  [valid acc: {val_acc:.3f}]')
        
        if learning_rate_scheduler:
            lr_scheduler.step(epoch+1)
            
        if check_point:
            checkpoint(val_loss, model)
            
        if early_stop:
            assert check_point==False, 'Choose between Early Stopping and Check Point'
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print('\n##########################\n'
                      '##### Early Stopping #####\n'
                      '##########################')
                break
                
    if early_stop==False and check_point==False:
        torch.save(model.state_dict(), last_epoch_save_path)
        print('Saving model of last epoch.')
        
    end_training = time.time()
    print(f'\nTotal time for training is {end_training-start_training:.3f}s')
    
    return model, loss_list, acc_list, val_loss_list, val_acc_list


if __name__ == '__main__':
    batch_size = 32
    lr = 1e-3
    EPOCH = 500
    device = torch.device('cuda')

    # load data
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms
    from catdogdataset import CatDogDataset

    path = 'C:/Users/user/MY_DL/classification/dataset/'
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
    
    # load callbacks
    from torchsummary import summary
    
    summary(VisionTransformer(**model_config), (3, 224, 224), device='cpu')

    model, train_loss, train_acc, valid_loss, valid_acc = train_step(
        model,
        train_data=train_loader,
        validation_data=valid_loader,
        epochs=EPOCH,
        learning_rate_scheduler=False,
        check_point=False,
        early_stop=False,
    )