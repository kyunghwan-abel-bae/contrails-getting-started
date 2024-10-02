import time

import torch
import torch.nn as nn
import torch.optim as optim

import seaborn as sns

from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader


# from data import *
from test_data import *
# from model import *
from test_model import *
from util import *

from datetime import datetime

from torch.utils.data import Subset

from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyTrainer:
    def __init__(self):
        print("init")
        self.model = UNet()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        # BCEWithLogits ~~ : loss functions : BCE + sigmoid

    def fit(self, images, masks):
        # print("fit")
        outputs = self.model(images)
        # print(f"output shape : {outputs.shape}")
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(100))

        loss = loss_fn(outputs, masks)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        '''
        model(image)
        
        '''

    def show_test(self, img, mask):
        print("show_test")
        print(img.shape)
        print(mask.shape)

        with torch.no_grad():
            output = self.model(img)

            output = rearrange(output[0], 'c h w -> h w c')
            mask = rearrange(mask[0], 'c h w -> h w c')

            print("rearrange")
            print(output.shape)
            print(mask.shape)

            fig, ax = plt.subplots(1, 2)
            axes = ax.flatten()

            axes[0].imshow(output)
            axes[1].imshow(mask)

            plt.show()


if __name__ == "__main__":
    trainer = MyTrainer()

    dataset_train = ContrailsAshDatasetTest("train")
    dataset_validataion = ContrailsAshDatasetTest("validation")

    print(f"data_train : {dataset_train}")
    # loader_train = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=2)#.dataset(dataset_train, batch_size=32)
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=2)#.dataset(dataset_train, batch_size=32)
    loader_validation = DataLoader(dataset_validataion, batch_size=32, shuffle=False, num_workers=2)#.dataset(dataset_train, batch_size=32)
    # print(f"dataloader_train : {dataloader_train}")

    episode = 1
    for e in range(episode):
        print(f"episode {e}...")
        for img, mask in loader_train:
            trainer.fit(img, mask)
            break
        break

    loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=2)#.dataset(dataset_train, batch_size=32)
    # # for test
    for img, mask in loader_train:
        trainer.show_test(img, mask)
        break




    # Load dataset Constrail
    # define trainer

    # loop episdoe
    #   loop all data
    #       trainer.fit(img, mask)
    #

