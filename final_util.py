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


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()


class DiceFinder:
    def __init__(self):
        print("init")

    def find_threshold(self, pred, mask):
        thresholds = torch.arange(0.5, 1, 0.001)
        max_dice_score = -1
        max_th = -1
        smooth = 1
        for th in thresholds:
            dice_score = 0
            for i in range(len(pred)):
                item_pred = torch.zeros_like(pred[i])
                item_mask = mask[i]

                item_pred[pred[i] > th] = 1

                intersection = (item_pred * item_mask).sum()
                # intersection = (item_pred == item_mask).float().sum().item()

                dice = (2.0 * intersection + smooth) / (item_pred.sum() + item_mask.sum() + smooth)

                dice_score += dice

            if dice_score > max_dice_score:
                max_dice_score = dice_score
                max_th = th

        return max_th