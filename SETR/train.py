import time
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from util.torchcallback import EarlyStopping, CheckPoint
from util.torchmetrics import Metrics
from util.torchlosses import OhemCELoss
from util.torchscheduler import PolynomialLRDecay
from model import SETR

lr = 0.01
EPOCH = 1000
device = torch.device('cuda')
num_classes = 19
model = SETR().to(device)

es_save_path = './model/es_checkpoint.pt'
cp_save_path = './model/cp_checkpoint.pt'
loss_func = OhemCELoss(thresh=0.7, ignore_lb=255).to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
lr_scheduler = PolynomialLRDecay(optimizer, max_decay_steps=EPOCH)
metric = Metrics(n_classes=num_classes, dim=1)
checkpoint = CheckPoint(verbose=True, path=cp_save_path)
early_stopping = EarlyStopping(patience=20, verbose=True, path=es_save_path)

def valid_step(model, validation_data):
    model.eval()
    with torch.no_grad():
        vbatch_loss, vbatch_miou = 0, 0
        for vbatch, (val_images, val_labels) in enumerate(validation_data):
            val_images, val_labels = val_images.to(device), val_labels.to(device)

            val_outputs, v_s2, v_s3, v_s4, v_s5 = model(val_images)

            val_miou = metric.mean_iou(val_outputs, val_labels)
            vbatch_miou += val_miou.item()

            val_loss = loss_func(val_outputs, val_labels.squeeze())
            vbatch_loss += val_loss.item()

            del val_images; del val_labels; del val_outputs
            torch.cuda.empty_cache()

    return vbatch_loss/(vbatch+1), vbatch_miou/(vbatch+1)

def train_on_batch(model, train_data):
    batch_loss, batch_miou = 0, 0
    for batch, (train_images, train_labels) in enumerate(train_data):
        model.train()

        train_images = train_images.to(device)
        train_labels = train_labels.to(device)

        optimizer.zero_grad()

        train_outputs, s2, s3, s4, s5 = model(train_images)

        miou = metric.mean_iou(train_outputs, train_labels)
        batch_miou += miou.item()

        loss = loss_func(train_outputs, train_labels.squeeze())
        batch_loss += loss.item()

        loss.backward()
        optimizer.step()

        del train_images; del train_labels; del train_outputs
        del s2; del s3; del s4; del s5
        torch.cuda.empty_cache()

    return batch_loss/(batch+1), batch_miou/(batch+1)

def train_step(model,
               train_data,
               validation_data,
               epochs,
               learning_rate_scheduler=False,
               check_point=False,
               early_stop=False,
               last_epoch_save_path='./model/last_checkpoint.pt'):
    
    loss_list, miou_list = [], []
    val_loss_list, val_miou_list = [], []
    
    print('Start Model Training...!')
    start_training = time.time()
    for epoch in tqdm(range(epochs)):
        init_time = time.time()
        
        train_loss, train_miou = train_on_batch(model, train_data)
        loss_list.append(train_loss)
        miou_list.append(train_miou)
            
        val_loss, val_miou = valid_step(model, validation_data)
        val_loss_list.append(val_loss)
        val_miou_list.append(val_miou)
        
        end_time = time.time()
        
        print(f'\n[Epoch {epoch+1}/{epochs}]'
              f'  [time: {end_time-init_time:.3f}s]'
              f'  [lr = {optimizer.param_groups[0]["lr"]}]')
        print(f'[train loss: {train_loss:.3f}]'
              f'  [train miou: {train_miou:.3f}]'
              f'  [valid loss: {val_loss:.3f}]'
              f'  [valid miou: {val_miou:.3f}]')
        
        if learning_rate_scheduler:
            lr_scheduler.step()
            
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
    
    return {
        'model': model, 
        'loss': loss_list, 
        'miou': miou_list, 
        'val_loss': val_loss_list, 
        'val_miou': val_miou_list
        }