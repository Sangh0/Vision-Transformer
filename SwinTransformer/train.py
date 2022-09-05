import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

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
        self.lr_scheduler = 