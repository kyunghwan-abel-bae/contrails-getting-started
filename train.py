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

    # Ensures that the given optimizer points to the given model
    def _check_optim_net_aligned(self):
        assert self.optimizer.param_groups[0]['params'] == list(self.model.parameters())

    # Trains the model
    def fit(self,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            epochs: int = 10,
            eval_every: int = 1,
            ):

        for e in range(epochs):
            print("New learning rate: {}".format(self.lr_scheduler.get_last_lr()))
            self.learning_rates.append(self.lr_scheduler.get_last_lr()[0])

            # Stores data about the batch
            batch_losses = []
            sub_batch_losses = []

            for i, data in enumerate(train_dataloader):
                self.model.train()

                if i % 100 == 0:
                    print(
                        f'epotch: {e} batch: {i}/{len(train_dataloader)} loss: {torch.Tensor(sub_batch_losses).mean()}')
                    sub_batch_losses.clear()
                # Every data instance is an input + label pair
                images, mask = data

                if torch.cuda.is_available():
                    images = images.cuda()
                    mask = mask.cuda()

                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                # Make predictions for this batch
                outputs = self.model(images)
                # Compute the loss and its gradients
                loss = self.loss_fn(outputs, mask)
                loss.backward()
                # Adjust learning weights
                self.optimizer.step()

                # Saves data
                self.batch_losses.append(loss.item())
                batch_losses.append(loss)
                sub_batch_losses.append(loss)

            # Adjusts learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Reports on the path
            mean_epoch_loss = torch.Tensor(batch_losses).mean()
            self.epoch_losses.append(mean_epoch_loss.item())
            print('Train Epoch: {} Average Loss: {:.6f}'.format(e, mean_epoch_loss))

            # Reports on the training progress
            if (e + 1) % eval_every == 0:
                torch.save(self.model.state_dict(), "model_checkpoint_e" + str(e) + ".pt")
                with torch.no_grad():
                    self.model.eval()
                    losses = []
                    for i, data in enumerate(test_dataloader):
                        # Every data instance is an input + label pair
                        images, mask = data

                        if torch.cuda.is_available():
                            images = images.cuda()
                            mask = mask.cuda()

                        output = self.model(images)
                        loss = self.loss_fn(output, mask)
                        losses.append(loss.item())

                    avg_loss = torch.Tensor(losses).mean().item()
                    self.validation_losses.append(avg_loss)
                    print("Validation loss after", (e + 1), "epochs was", round(avg_loss, 4))


if __name__ == '__main__':
    print(f"cuda : {torch.cuda.is_available()}")

    time_start = datetime.now()
    print(f"start time : {time_start}")


    dice = Dice()

    dataset_train = ContrailsAshDataset('train')
    dataset_validation = ContrailsAshDataset('validation')

    dataset_train = Subset(dataset_train, range(500))

    #data_loader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=2)
    #data_loader_validation = DataLoader(dataset_validation, batch_size=16, shuffle=True, num_workers=2)
    data_loader_train = DataLoader(dataset_train, batch_size=16, shuffle=False, num_workers=2)
    data_loader_validation = DataLoader(dataset_validation, batch_size=16, shuffle=False, num_workers=2)

    #dir(data_loader_validation)

    temp_dataset = data_loader_validation.dataset

    print(data_loader_validation)
    print(dataset_validation.df_idx.head(10))
    # print(dataset_train.parrent_folder)


    #2,5,6,7,10

    # ImageFolder로부터 폴더 이름 정보를 추출합니다.
    class_names = getattr(temp_dataset, 'classes', None)
    print(class_names)

    train = True

    if train:
        model = UNet()
        model.to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(100))
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.70)

        num_epochs = 2

        trainer = MyTrainer(model, optimizer, criterion, lr_scheduler)
        trainer.fit(data_loader_train, data_loader_validation, epochs=num_epochs)
    else:
        model = UNet()
        model.load_state_dict(torch.load('model_checkpoint_e10.pt'))
        model.eval()
        model.to(device)


    if train:
        df_data = pd.DataFrame({'Batch Losses': trainer.batch_losses})

        sns.lineplot(data=df_data)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Batch Loss')
        plt.show()

    time_end = datetime.now()
    print(f"end time : {time_end}")

    print(f"elapsed time : {time_end-time_start}")
