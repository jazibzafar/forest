##
import torch
import numpy as np
from tifffile import imread


##
path = '/data_hdd/pauline/dataset/train/masks/mask_patch_HEG01_06.tif'
mask_npy = imread(path)
mask_tens = torch.Tensor(mask_npy)
mask_tens_int = mask_tens.round().to(torch.int64)

##
# characteristics of mask
print("shape of tens mask: ", mask_tens.shape)
unique_values = torch.unique(mask_tens)
print("values in tens mask: ", unique_values)
print("dtype of tens mask pixels: ", mask_tens.dtype)
print("dtype of npy mask pixels: ", mask_npy.dtype)
print("dtype of npy mask pixels: ", mask_tens_int.dtype)

##
one_hot = torch.nn.functional.one_hot(mask_tens_int, num_classes=4)
# one_hot = np.eye(4)[mask]

##
targets = torch.randint(0, 4, (32, 320, 320), dtype=torch.int64)
print(targets.size())

##
output = torch.randn(32, 4, 320, 320, dtype=torch.float32)
print(output.size())
