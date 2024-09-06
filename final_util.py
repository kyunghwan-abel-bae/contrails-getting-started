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
    return np.load(os.path.join("data", str_data_type, str_folder, f'band_{num_band_index:02}.npy'))


def get_band_mask(str_data_type, str_folder):
    return np.load(os.path.join("data", str_data_type, str_folder, "human_pixel_masks.npy"))


def process_data(data, bounds):
    data = np.clip(data, bounds[0], bounds[1])
    data = (data - bounds[0]) / (bounds[1] - bounds[0])
    return rearrange(data, 'h w s -> s h w')


def get_data(str_data_type, str_folder):
    band_11 = get_band_image(str_data_type, str_folder, 11)
    band_14 = get_band_image(str_data_type, str_folder, 14)
    band_15 = get_band_image(str_data_type, str_folder, 15)

    ch1 = process_data((band_15 - band_14), _TDIFF_BOUNDS)
    ch2 = process_data((band_14 - band_11), _CLOUD_TOP_TDIFF_BOUNDS)
    ch3 = process_data(band_14, _T11_BOUNDS)

    return np.stack([ch1, ch2, ch3])
