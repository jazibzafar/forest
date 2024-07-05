##
import numpy as np
from src.nnblocks import ClipSegStyleDecoder
import src.vits as vits
import torch

# v4 is with 0.891 miou, v6 is with 0.917
ckpt_path = "./test2/last-v6.ckpt"
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
img = imread("/home/jazib/projects/data/gartow_seg/single_class_sliced/8020/val/tiles/31.tif")
msk = imread("/home/jazib/projects/data/gartow_seg/single_class_sliced/8020/val/masks/31.tif")
img_rgb = to_rgb(img, 'HWC')


from torchvision.transforms import ToTensor
img_tr = ToTensor()(img).unsqueeze(0)
output = model(img_tr)
output = output.squeeze()
output = output.detach().numpy()

threshold = 0.5
thresh = np.where(output > threshold, 1, 0)


import matplotlib.pyplot as plt

fig = plt.figure()
rows = 2
cols = 2
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
fig.add_subplot(rows, cols, 4)
plt.imshow(thresh)
plt.axis('off')
plt.title('thresh')
plt.show()
fig.savefig("./figures/seg_test_results.png")
##

alphas_thr = np.ones((240, 240)) - thresh
alphas_msk = np.ones((250, 250)) - msk

fig2 = plt.figure()
fig2.add_subplot(131)
plt.imshow(img_rgb)
# plt.imshow(msk, cmap='gray', alpha=alphas_msk) #, cmap=plt.cm.gray)
plt.axis('off')
plt.title('img')
fig2.add_subplot(132)
plt.imshow(img_rgb)
plt.imshow(msk, cmap='gray', alpha=alphas_msk) #, cmap=plt.cm.gray)
plt.axis('off')
plt.title('img and input msk')
fig2.add_subplot(133)
plt.imshow(img_rgb)
plt.imshow(thresh, cmap='gray', alpha=alphas_thr) #, cmap=plt.cm.gray)
plt.axis('off')
plt.title('img and output msk')
plt.show()
fig2.savefig("./figures/seg_test_results_compare.png")

##
