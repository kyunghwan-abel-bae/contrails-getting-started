import torch.utils.data

from final_data import *
from final_model import *
from final_util import *

from tqdm import tqdm  # tqdm 라이브러리 임포트



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load dataset
data_train = ContrailAshDataset("train")
data_validation = ContrailAshDataset("validation")
data_test = ContrailAshDataset("test")

loader_train = torch.utils.data.DataLoader(data_train, batch_size=32, shuffle=False)
loader_validation = torch.utils.data.DataLoader(data_validation, batch_size=32, shuffle=False)
loader_test = torch.utils.data.DataLoader(data_test, batch_size=32, shuffle=False)

model = None

# model = UNet().to(device)
# model.load_state_dict(torch.load('model/unet.pt', map_location=device))

# TRAIN
if model is None:
    model = UNet().to(device)
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(100))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for e in range(1):
        for img, mask in tqdm(loader_train, desc=f"Epoch {e + 1}/{100}", unit="batch"):
            if torch.cuda.is_available():
                img = img.to(device)
                mask = mask.to(device)

            optimizer.zero_grad()

            pred = model(img)

            loss_value = loss(pred, mask)
            loss_value.backward()
            optimizer.step()

# find threshold
with torch.no_grad():
    flatten_pred = None
    flatten_mask = None
    for img, mask in loader_validation:
        pred = torch.sigmoid(model(img))

        temp_pred = rearrange(pred, 'b s h w -> b (s h w)')
        temp_mask = rearrange(mask, 'b s h w -> b (s h w)')

        if flatten_pred is None:
            flatten_pred = temp_pred
            flatten_mask = temp_mask
        else:
            torch.cat([flatten_pred, temp_pred])
            torch.cat([flatten_mask, temp_mask])

    print(f"flatten_pred : {flatten_pred.shape}, flatten_mask : {flatten_mask.shape}")
    # print(f"value : {flatten_pred[0]}")
    # quit()
    # dice(flatten_pred, flatten_mask)
    dice_finder = DiceFinder()
    threshold = dice_finder.find_threshold(flatten_pred, flatten_mask)
    print(f"threshold : {threshold}")


        # list_pred.append(flatten_pred)
        # list_mask.append(flatten_mask)

# predict & show
# 4 imgs : mask, frame, pred, masked

'''
pred = pred.detach().numpy()
mask = mask.detach().numpy()

fig, axes = plt.subplots(5, 2)

for i, m in enumerate(mask):
    m = rearrange(m, 'c h w -> h w c')

    p = pred[i]
    p = rearrange(p, 'c h w -> h w c')

    axes[i][0].imshow(m)
    axes[i][1].imshow(p)

plt.show()
'''



