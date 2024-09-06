import os
import torch

import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt

from einops import *


_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)

def get_band_image(str_data_type, str_folder, num_band_index):
    print("get_band_iamge")
    # print(os.path.join("data", str_folder, str(num_band_index)))
    return np.load(os.path.join("data", str_data_type, str_folder, f'band_{num_band_index:02}.npy'))

def get_band_mask(str_data_type, str_folder):
    print("get band mask")
    return np.load(os.path.join("data", str_data_type, str_folder, "human_pixel_masks.npy"))

def normalize_data(data, bounds):

    data = np.clip(data, bounds[0], bounds[1])

    data = rearrange(data, 'h w s -> s h w')
    # print("normalize data")
    smooth = 1e-15
    for i, d in enumerate(data):
        d = (d - bounds[0]) / (bounds[1] - bounds[0])# + smooth)

        data[i] = d

    return data


def get_data(str_data_type, str_folder):
    print("get_data")
    print(f"str_folder : {str_folder}")
    band_11 = get_band_image(str_data_type, str_folder, 11)
    band_14 = get_band_image(str_data_type, str_folder, 14)
    band_15 = get_band_image(str_data_type, str_folder, 15)

    # print(band_14)
    # filtered_band_14 = normalize_data(band_11)
    ch1 = normalize_data(band_14, _T11_BOUNDS)#, _T11_BOUNDS[0], _T11_BOUNDS[1]))
    ch2 = normalize_data((band_14-band_11), _CLOUD_TOP_TDIFF_BOUNDS)
    ch3 = normalize_data((band_15-band_14), _TDIFF_BOUNDS)
    # print(ch1)

    # ch_rgb = np.concatenate([ch1, ch2, ch3], axis=0)
    ch_rgb = np.stack([ch1, ch2, ch3])

    ch_rgb = rearrange(ch_rgb, 'c s h w -> (c s) h w')

    print(ch_rgb.shape)

    mask = get_band_mask(str_data_type, str_folder)
    mask = rearrange(mask, 'h w s -> s h w')

    # print(mask)
    print(mask.shape)

    # Data Filtering
    # 14
    # 14-11
    # 15-14

    # normalize band 11


    # band_test = get_band_image(str_data_type, str_folder, 8)
    # print(ch1.shape)
    # print(band_test.shape)
    # print(band_11)

    return ch_rgb, mask