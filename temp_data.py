import os

import numpy as np
import torch
import pandas as pd

from util import *
from einops import rearrange

_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])

def get_band_images(idx, parent_folder, band):
    return np.load(os.path.join("data", parent_folder, idx, f'band_{band}.npy'))

def get_mask_image(idx, parent_folder):
    return np.load(os.path.join("data", parent_folder, idx, f'human_pixel_masks.npy'))

def get_ash_color_images(image_id, parent_folder, get_mask_frame_only=False):
    band11 = get_band_images(image_id, parent_folder, '11')
    band14 = get_band_images(image_id, parent_folder, '14')
    band15 = get_band_images(image_id, parent_folder, '15')

    if get_mask_frame_only:
        band11 = band11[:, :, 4]
        band14 = band14[:, :, 4]
        band15 = band15[:, :, 4]

    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)

    return false_color

    # print("false_color shape : ")
    # print(false_color.shape)
    #
    # false_color_rearrange = rearrange(false_color, 'h w c b -> b h w c')
    # print(false_color_rearrange.shape)
    # print(false_color_rearrange[0].shape)
    # plt.imshow(false_color_rearrange[0])
    # plt.show()

    # print("get_ash_color_images")
    # # load (local npy file) here
    # list_band_file_name = os.listdir(f"data/{parent_folder}/{image_id}")
    # print(list_band_file_name)
    # np_test = np.load(f"data/{parent_folder}/{image_id}/human_individual_masks.npy")
    # print("TEST")
    # print(np_test.shape)

    # np_mask = np.load(f"data/{parent_folder}/{image_id}/human_pixel_masks.npy")
    # print("MASK")
    # print(np_mask.shape)

    # np_image = np.load(f"'data/{parent_folder}/{image_id}/}'")




class ContrailsAshDataset(torch.utils.data.Dataset):
    def __init__(self, parent_folder: str):
        self.df_idx: pd.DataFrame = pd.DataFrame({'idx': os.listdir(f'data/{parent_folder}')})
        self.parent_folder = parent_folder

    def __len__(self):
        return len(self.df_idx)

    def __getitem__(self, idx):
        print("get_item")
        image_id = str(self.df_idx.iloc[idx]['idx'])
        # images = torch.tensor(np.reshape(get_ash_color_images(image_id, self.parent_folder, get_mask_frame_only=False), (256, 256, 24))).to(torch.float32).permute(2, 0, 1)
        ash_image = get_ash_color_images(image_id, self.parent_folder, get_mask_frame_only=False)
        print(f"ash images : {ash_image.shape}")
        images = torch.tensor(np.reshape(get_ash_color_images(image_id, self.parent_folder, get_mask_frame_only=False), (256, 256, 24))).to(torch.float32)
        # mask = torch.tesor(get_mask_image(image_id, self.parent_folder))
        mask = torch.tensor(get_mask_image(image_id, self.parent_folder))

        # return get_ash_color_images(image_id, self.parent_folder, get_mask_frame_only=False)
        # return get_mask_image(image_id, self.parent_folder)#get_ash_color_images(image_id, self.parent_folder, get_mask_frame_only=False)
        return images, mask
        # images = torch.tensor(np.reshape(get_ash_color_images(image_id, self.parent_folder, get_mask_frame_only=False), (256, 256, 24))).to(torch.float32).permute(2, 0, 1)



# dataset = ContrailsAshDataset("train")
# img, mask = dataset[0]
# print(img.shape)
# print(mask.shape)
# plt.imshow(img)
# plt.show()