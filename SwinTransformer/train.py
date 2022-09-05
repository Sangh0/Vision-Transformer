import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from util.callback import CheckPoint, EarlyStopping
from util.scheduler import CosineAnnealingWarmupRestarts

class TrainModel(object):
    """
    Args:
        model: model to train
        lr: learning rate
        epochs: max epochs
        weight_decay: weight decay of optimizer
        lr_scheduling: apply learning rate scheduler
        check_point: save weight of model when model achieved best score
        early_stop: stop training when overfitting occurs
    """

    def __init__(
        self,
        model,
        lr=1e-3,
        epochs=300,
        weight_decay=5e-2,
        lr_scheduling=False,
        check_point=False,
        early_stop=False,
        train_log_step=40,
        valid_log_step=20,
    ):  

        assert (check_point==True and early_stop==False) or (check_point==False and early_stop==True), \
            'Choose between Early stopping and Check Point'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)

        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        
        self.epochs = epochs
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.lr_scheduling = lr_scheduling
        self.lr_scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer, 
            first_cycle_steps=50, 
            cycle_mult=1,
            max_lr=lr,
            min_lr=lr*1e-2,
            warmup_steps=20,
            gamma=1.0,
        )

        self.check_point = check_point
        self.cp = CheckPoint(verbose=True)

        self.early_stop = early_stop
        self.es = EarlyStopping(patience=20, verbose=True, path='./weights/early_stop.pt')

        self.writer = SummaryWriter()

        self.train_log_step = train_log_step
        self.valid_log_step = valid_log_step

    def fit(self, train_data, validation_data):
        print('Start Model Training...!')
        start_training = time.time()
        pbar = tqdm(range(self.epochs), total=int(self.epochs))
        for epoch in pbar:
            init_time = time.time()

            train_loss, train_acc = self.train_on_batch(
                train_data, epoch, self.train_log_step,
            )

            valid_loss, valid_acc = self.validation_on_batch(
                validation_data, epoch, self.valid_log_step,
            )

            end_time = time.time()

            self.writer.add_scalar('lr', self.optimizer.param_groups[0]["lr"], epoch)

            print(f'\n{"="*45} Epoch {epoch+1}/{self.epochs} {"="*45}'
                  f'time: {end_time-init_time:2f}s'
                  f'   lr = {self.optimizer.param_groups[0]["lr"]}')
            print(f'\ntrain average loss: {train_loss:.3f}'
                  f'   accuracy: {train_acc:.3f}')
            print(f'\nvalid average loss: {valid_loss:.3f}'
                  f'   accuracy: {valid_acc:.3f}')
            print(f'\n{"="*100}')

            if self.lr_scheduling:
                self.lr_scheduler.step()

            if self.check_point:
                path = f'./weights/check_point_{epoch+1}.pt'
                self.cp(valid_loss, self.model, path)

            if self.early_stop:
                self.es(valid_loss, self.model)
                if self.es.early_stop:
                    print('\n##########################\n'
                          '##### Early Stopping #####\n'
                          '##########################')
                    break

        self.writer.close()
        end_training = time.time()
        print(f'\nTotal time for training is {end_training-start_training:.2f}s')

        return {
            'model': self.model,
        }

    @torch.no_grad()
    def validation_on_batch(self, validation_data, epoch, log_step):
        self.model.eval()
        batch_loss, batch_acc = 0, 0
        for batch, (images, labels) in enumerate(validation_data):
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_func(outputs, labels)
            output_index = torch.argmax(outputs, dim=1)
            acc = (output_index==labels).sum()/len(outputs)

            if batch == 0:
                print(f'\n{" "*20} Validate Step {" "*20}')

            if (batch+1) % log_step == 0:
                print(f'\n[Batch {batch+1}/{len(validation_data)}]'
                      f'  valid loss: {loss:.3f}   accuracy: {acc:.3f}')

            steps = epoch * len(validation_data) + batch
            self.writer.add_scalar('Valid/Loss', loss, steps)
            self.writer.add_scalar('Valid/Accuracy', acc, steps)

            batch_loss += loss.item()
            batch_acc += acc.item()

            del images; del labels; del outputs
            torch.cuda.empty_cache()

        return batch_loss/(batch+1), batch_acc/(batch+1)

    def train_on_batch(self, train_data, epoch, log_step):
        batch_loss, batch_acc = 0, 0
        for batch, (images, labels) in enumerate(train_data):
            self.model.train()

            self.optimizer.zero_grad()

            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.loss_func(outputs, labels)
            output_index = torch.argmax(outputs, dim=1)
            acc = (output_index==labels).sum()/len(outputs)

            loss.backward()
            self.optimizer.step()

            if batch == 0:
                print(f'\n{" "*20} Train Step {" "*20}')

            if (batch+1) % log_step == 0:
                print(f'\n[Batch {batch+1}/{len(train_data)}]'
                      f'  train loss: {loss:.3f}   accuracy: {acc:.3f}')

            steps = epoch * len(train_data) + batch
            self.writer.add_scalar('Train/Loss', loss, steps)
            self.writer.add_scalar('Train/Accuracy', acc, steps)

            batch_loss += loss.item()
            batch_acc += acc.item()
            
            del images; del labels; del outputs
            torch.cuda.empty_cache()

        return batch_loss/(batch+1), batch_acc/(batch+1)