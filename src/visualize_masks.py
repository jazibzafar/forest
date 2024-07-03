##
import numpy as np
from src.nnblocks import ClipSegStyleDecoder
import src.vits as vits
import torch


ckpt_path = "./test2/last-v4.ckpt"
loaded_ckpt = torch.load(ckpt_path)
patch_size = 16
backbone = vits.__dict__["vit_tiny"](patch_size=patch_size, num_classes=0)
model = ClipSegStyleDecoder(backbone, patch_size, reduce_dim=112, n_heads=4, complex_trans_conv=True)

from src.checkpoints import model_remove_prefix
ckpt_state_dict = loaded_ckpt['state_dict']
ckpt_state_dict = model_remove_prefix(ckpt_state_dict, "model.")
model.load_state_dict(ckpt_state_dict)
model.eval()
##
# load a test image
def to_rgb(img_in, mode='CHW'):
    if mode == 'CHW':
        return img_in[:3, :, :]
    elif mode == 'HWC':
        return img_in[:, :, :3]
    else:
        print("mode = CHW or HWC")


from tifffile import imread
img = imread("/home/jazib/projects/data/gartow_seg/single_class_sliced/8020/val/tiles/15.tif")
msk = imread("/home/jazib/projects/data/gartow_seg/single_class_sliced/8020/val/masks/15.tif")
img_rgb = to_rgb(img, 'HWC')


from torchvision.transforms import ToTensor
img_tr = ToTensor()(img).unsqueeze(0)
output = model(img_tr)
output = output.squeeze()
output = output.detach().numpy()

import matplotlib.pyplot as plt

fig = plt.figure()
rows = 1
cols = 3
fig.add_subplot(rows, cols, 1)
plt.imshow(img_rgb)
plt.axis('off')
plt.title('tile')
fig.add_subplot(rows, cols, 2)
plt.imshow(msk)
plt.axis('off')
plt.title('mask')
fig.add_subplot(rows, cols, 3)
plt.imshow(output)
plt.axis('off')
plt.title('pred')
plt.show()
