import os

import numpy as np
import torch
import pandas as pd

# from util import *
from final_util import *

from util import *


class ContrailAshDataset_CMP1(torch.utils.data.Dataset):
    def __init__(self, str_folder):
        print("init")
        self.str_data_type = str_folder
        self.df_idx = pd.DataFrame({'idx':os.listdir(os.path.join("data", str_folder))})

    def __getitem__(self, index):
        img, mask = get_data(self.str_data_type, self.df_idx.iloc[index].item())
        print("get item")
        return img, mask


class ContrailAshDataset_CMP2(torch.utils.data.Dataset):
    def __init__(self, str_folder):
        print("init")
        self.str_data_type = str_folder
        self.df_idx = pd.DataFrame({'idx':os.listdir(os.path.join("data", str_folder))})

    def __getitem__(self, index):
        # get_ash_color_images()
        img = get_ash_color_images(self.df_idx.iloc[index].item(), self.str_data_type) #get_data(self.str_data_type, self.df_idx.iloc[index].item())
        mask = get_mask_image(self.df_idx.iloc[index].item(), self.str_data_type)
        print("get item")
        return img, mask


data_train = ContrailAshDataset_CMP1("train")
img, mask = data_train[0]
print(f"cmp1 img.shape : {img.shape}, mask.shape : {mask.shape}")

data_train02 = ContrailAshDataset_CMP2("train")
img2, mask2 = data_train02[0]
img2 = rearrange(img2, 'h w c s -> (c s) h w')
mask2 = rearrange(mask2, 'h w s -> s h w')
print(f"cmp2 img.shape : {img2.shape}, mask.shape : {mask2.shape}")

print(f"equal img : {np.array_equal(img, img2)}, mask : {np.array_equal(mask, mask2)}")
print(f"img : {img[0][30][30]}, img2 : {img2[0][30][30]}")