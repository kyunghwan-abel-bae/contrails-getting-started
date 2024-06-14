import matplotlib.pyplot as plt

from util import *

import pandas as pd


df_train_idx = pd.DataFrame({'idx': os.listdir('data/train')})
df_validation_idx = pd.DataFrame({'idx': os.listdir('data/validation')})
df_test_idx = pd.DataFrame({'idx': os.listdir('data/test')})

# ''' Added by KH for TESTING
df_train_idx = df_train_idx.head(1000)
df_validation_idx = df_validation_idx.head(1000)
df_test_idx = df_test_idx.head(1000)
# Added by KH for TESTING '''

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(40, 40))
axes = axes.flatten()

for i in range(len(axes)):
    images = get_ash_color_images(str(df_train_idx.iloc[730 + i]['idx']), 'train')
    axes[i].imshow(images[:,:,:,4])
    axes[i].axis('off')

plt.show()