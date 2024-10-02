import os.path

import torch
import pandas as pd

from util import *
# from test_util import *

from einops import rearrange


class ContrailsAshDatasetTest:
    def __init__(self, str_dir):
        print("init")
        self.list_idx = os.listdir(os.path.join("data", str_dir))
        # self.df_idx = pd.DataFrame(os.listdir((os.path.join("data", str_dir))))
        # self.df_idx = pd.DataFrame({'idx': os.listdir((os.path.join("data", str_dir)))})

        self.parent_dir = str_dir

    def __len__(self):
        return len(self.list_idx)

    def __getitem__(self, idx):
        # data load by idx
        data = get_ash_color_images(self.list_idx[idx], parrent_folder=self.parent_dir, get_mask_frame_only=False)
        mask = get_mask_image(self.list_idx[idx], parrent_folder=self.parent_dir)

        data = rearrange(data, 'h w c b -> (c b) h w')
        mask = rearrange(mask, 'h w c -> c h w')

        data = torch.tensor(data, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return data, mask

        '''
        mask = get_mask_image(self.list_idx[idx], parrent_folder=self.parent_dir)

        data = rearrange(data, 'h w c b -> (c b) h w')
        mask = rearrange(mask, 'h w c -> c h w')

        data = torch.tensor(data, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return data, mask
        '''


# temp_dataset = ContrailsAshDatasetTest("train")
# temp_data, temp_mask = temp_dataset[0]
#
# # print(temp_data.shape)
# temp_data = rearrange(temp_data, 'c h w b -> b h w c')
# print(temp_mask.shape)
# plt.imshow(temp_data[0])
# plt.show()