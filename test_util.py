import os
import torch

import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt


_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)



def get_band_images(idx: str, parrent_folder: str, band: str):
    return np.load(os.path.join("data", parrent_folder, idx, f"band_{int(band):02}.npy"))

def normalize_range(data, bounds):
    return (data - bounds[0]) / (bounds[1] - bounds[0])

def get_ash_color_images(idx: str, parrent_folder: str, get_mask_frame_only=False):
    band_image_11 = get_band_images(idx, parrent_folder, 11) # 14-11
    band_image_14 = get_band_images(idx, parrent_folder, 14) # T11
    band_image_15 = get_band_images(idx, parrent_folder, 15) # 15-14

    if get_mask_frame_only:
        band_image_11 = band_image_11[:, :, 4]
        band_image_14 = band_image_14[:, :, 4]
        band_image_15 = band_image_15[:, :, 4]

    normalized_img1 = normalize_range(band_image_14, _T11_BOUNDS)
    normalized_img2 = normalize_range((band_image_14-band_image_11), _CLOUD_TOP_TDIFF_BOUNDS)
    normalized_img3 = normalize_range((band_image_15-band_image_14), _TDIFF_BOUNDS)

    false_color = np.stack([normalized_img1, normalized_img2, normalized_img3])

    print("false_color")
    print(false_color.shape)
    false_color = np.clip(false_color, 0, 1)

    return false_color


def get_mask_image(idx: str, parrent_folder: str):
    return np.load(os.path.join("data", parrent_folder, idx, "human_pixel_masks.npy"))


class Dice(nn.Module):
    def __init__(self, use_sigmoid=True):
        super(Dice, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = use_sigmoid

    def forward(self, inputs, targets, smooth=1):
        if self.use_sigmoid:
            inputs = self.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)


class DiceThresholdTester:
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        self.cumulative_mask_pred = []
        self.cumulative_mask_true = []

    def precalculate_prediction(self):
        sigmoid = nn.Sigmoid()

        for images, mask_true in self.data_loader:
            if torch.cuda.is_available():
                images = images.cuda()

            mask_pred = sigmoid(self.model.forward(images))

            self.cumulative_mask_pred.append(mask_pred.cpu().detach().numpy())
            self.cumulative_mask_true.append(mask_true.cpu().detach().numpy())

        self.cumulative_mask_pred = np.concatenate(self.cumulative_mask_pred, axis=0)
        self.cumulative_mask_true = np.concatenate(self.cumulative_mask_true, axis=0)

        self.cumulative_mask_pred = torch.flatten(torch.from_numpy(self.cumulative_mask_pred))
        self.cumulative_mask_true = torch.flatten(torch.from_numpy(self.cumulative_mask_true))

    def test_threshold(self, threshold):
        _dice = Dice(use_sigmoid=False)
        after_threshold = np.zeros(self.cumulative_mask_pred.shape)
        after_threshold[self.cumulative_mask_pred[:] > threshold] = 1
        after_threshold[self.cumulative_mask_pred[:] < threshold] = 0
        after_threshold = torch.flatten(torch.from_numpy(after_threshold))
        return _dice(self.cumulative_mask_true, after_threshold).item()
