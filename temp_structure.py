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

    train = True
    if train is True:
        print("if model")
        # define model
        # model fit with loaded data

    # find optimal threshold with Validation data

    # predict with validation data





