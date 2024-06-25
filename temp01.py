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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    print("main")
    # data load as Subset
    #
    dataset_train = ContrailsAshDataset("train")
    dataset_validation = ContrailsAshDataset("validation")

    dataset_train = Subset(dataset_train, range(3))
    dataset_validation = Subset(dataset_validation, range(3))

    # list_idx = dataset_train
    # print(list_idx)
    # quit()

    # dataloader_train = DataLoader(dataset_train)
    loader_train = DataLoader(dataset_train, batch_size=20, shuffle=False, num_workers=2)
    loader_validation = DataLoader(dataset_validation, batch_size=20, shuffle=False, num_workers=2)

    quit()

    train = True
    if train is True:
        print("if model")
        # define model
        # model fit with loaded data

    # find optimal threshold with Validation data

    # predict with validation data





