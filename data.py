import os

import numpy as np
import torch
import pandas as pd

from util import *

class ContrailAshDataset(torch.utils.data.Dataset):
    def __init__(self, str_folder):
        self.str_data_type = str_folder
        self.df_idx = pd.DataFrame({'idx':os.listdir(os.path.join("data", str_folder))})

    def __len__(self):
        return len(self.df_idx)

    def __getitem__(self, index):
        str_folder = str(self.df_idx.iloc[index]['idx'])

        img = torch.tensor(get_data(self.str_data_type, str_folder))
        img = rearrange(img, 'c s h w -> (c s) h w') # s means sequence

        mask = torch.tensor(get_band_mask(self.str_data_type, str_folder))
        mask = rearrange(mask, 'h w s -> s h w')

        return img.float(), mask.float()


# data_train = ContrailAshDataset("train")
# img, mask = data_train[0]

# print(f"img&mask.shape : {img.shape}, {mask.shape}")