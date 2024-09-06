# GOAL

# Among imgs, find contrails, and marking
# how can i find contrails > data & data labels

import time

import torch
import torch.nn as nn
import torch.optim as optim

import seaborn as sns

from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader


from data import *
from model import *
from util import *

from datetime import datetime

from torch.utils.data import Subset

from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyTrainer:
    def __init__(self, model, optimizer, loss_fn, lr_scheduler):
        self.validation_losses = []
        self.batch_losses = []
        self.epoch_losses = []
        self.learning_rates = []

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self._check_optim_net_aligned()

    def fit(self, data_train: DataLoader, data_valid: DataLoader, epochs=11, eval_every: int = 1):

        for e in range(epochs):
            print("New learning rate: {}".format(self.lr_scheduler.get_last_lr()))
            self.learning_rates.append(self.lr_scheduler.get_last_lr()[0])

            batch_losses = []
            sub_batch_losses = []

            for i, data in enumerate(data_train):
                if i % 100 == 0:
                    print(
                        f'epotch: {e} batch: {i}/{len(data_valid)} loss: {torch.Tensor(sub_batch_losses).mean()}')
                    sub_batch_losses.clear()
                # Every data instance is an input + label pair
                images, mask = data

                if torch.cuda.is_available():
                    images = images.cuda()
                    mask = mask.cuda()

                print("image =======")

                self.optimizer.zero_grad()

                # check image here
                pred = self.model(images)
                print(pred.shape)

                print("mask =======")
                print(mask.shape)

                loss = self.loss_fn(pred, mask)

                self.loss_fn.backward()
                self.optimizer.step()

                self.batch_losses.append(loss.item())
                batch_losses.append(loss)
                sub_batch_losses.append(loss)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            mean_epoch_loss = torch.Tensor(batch_losses).mean()
            self.epoch_losses.append(mean_epoch_loss.item())
            print('Train Epoch: {} Average Loss: {:.6f}'.format(e, mean_epoch_loss))

            if (e + 1) % eval_every == 0:
                # torch.save(self.model.state_dict(), "model_checkpoint")
                with torch.no_grad():
                    self.model.eval()
                    losses = []
                    for i, data in enumerate(data_valid):
                        images, mask = data

                        if torch.cuda.is_available():
                            images = images.cuda()
                            mask = mask.cuda()

                        output = self.model(images)
                        loss = self.loss_fn(output, mask)
                        losses.append(loss.item())

                    avg_loss = torch.Tensor(losses).mean().item()
                    self.validation_losses.append(avg_loss)
                    print("Validation loss after", (e + 1), "epoch was", round(avg_loss, 4))




if __name__ == '__main__':
    print("main")

    print(f"cuda : {torch.cuda.is_available()}")

    time_start = datetime.now()
    print(f"start time : {time_start}")

    dice = Dice()

    dataset_train = ContrailsAshDataset("train")
    dataset_validation = ContrailsAshDataset("validation")

    dataset_train = Subset(dataset_train, range(3))
    dataset_validation = Subset(dataset_validation, range(3))

    loader_train = DataLoader(dataset_train, batch_size=16, shuffle=False, num_workers=2)
    loader_validation = DataLoader(dataset_validation, batch_size=16, shuffle=False, num_workers=2)

    train = True
    if train is True:
        print("if model")
        model = UNet()
        model.to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(100))
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.70)

        num_epochs = 2

        trainer = MyTrainer(model, optimizer, criterion, lr_scheduler)
        trainer.fit(loader_train, loader_validation, epochs=num_epochs)

    # find optimal threshold with Validation data

    # predict with validation data





