import albumentations as A
import os
from tifffile import imread
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import torch


# def img_loader(path, dim_ordering="CHW"):
#     if dim_ordering == "HWC":
#         return imread(path)
#     elif dim_ordering == "CHW":
#         return imread(path).transpose((2,0,1))
#     else:
#         raise RuntimeError(f"Received wrong dim ordering '{dim_ordering}', must be either CHW or HWC.")


def img_loader(path):
    return imread(path)


class ClassificationTransform:
    def __init__(self, input_size):
        self.input_size = input_size

        self.transforms = A.Compose([
            A.RandomCrop(height=self.input_size,
                         width=self.input_size,
                         always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=(0.2, 0.3),
                                       contrast_limit=(0.2, 0.3),
                                       p=0.2),
            A.RandomGamma(gamma_limit=(100, 140), p=0.2),
            A.RandomToneCurve(scale=0.1, p=0.2)
        ])

    def __call__(self, image):
        # Make sure image is a np array
        if type(image) == 'torch.Tensor':
            image = image.numpy()

        crop = ToTensor()(self.transforms(image=image)['image'])
        return crop


class CenterCrop:
    def __init__(self, input_size):
        self.transform = A.CenterCrop(height=input_size,
                                      width=input_size,
                                      always_apply=True)

    def __call__(self, image):
        # Make sure image is a np array
        if type(image) == 'torch.Tensor':
            image = image.numpy()

        crop = ToTensor()(self.transform(image=image)['image'])
        return crop


class SegDataset(Dataset):
    def __init__(self, data_path, crop_size, train=True):
        super().__init__()
        self.tile_path = os.path.join(data_path, 'tiles/')
        self.mask_path = os.path.join(data_path, 'masks/')
        self.tile_list = sorted(os.listdir(self.tile_path))  # , key=len)  # this is a list
        self.mask_list = sorted(os.listdir(self.mask_path))  # , key=len)
        for i in range(len(self.tile_list)):
            print(f"{self.tile_list[i]} -----> {self.mask_list[i]}")
        if train:
            self.default_augment = A.Compose([
                A.RandomCrop(height=crop_size,
                             width=crop_size,
                             always_apply=True),
                A.HorizontalFlip(p=0.3),  # 0.5
                A.RandomRotate90(p=0.1),  # 0.2
            ])  # these augments need to applied to both tile and mask
            # for any other augments, apply separate augments for tiles.
            # masks do not need additional augments.
        else:
            # for testing/validation
            self.default_augment = A.CenterCrop(height=crop_size,
                                                width=crop_size,
                                                always_apply=True)

    def __len__(self):
        return len(self.tile_list)

    def __getitem__(self, index):
        tile_name = self.tile_list[index]
        mask_name = self.mask_list[index]
        tile_path = os.path.join(self.tile_path, tile_name)
        mask_path = os.path.join(self.mask_path, mask_name)
        tile = imread(tile_path)
        mask = imread(mask_path)
        # stack tile and mask together for crops and flips
        stacked = np.dstack((tile, mask))
        # default augments
        # no need to verify if the img is np array or not.
        t_stacked = self.default_augment(image=stacked)['image']
        # split the stacked img back into tile and mask
        t_tile = t_stacked[:, :, 0:4]
        t_mask = t_stacked[:, :, 4]
        # # convert tile and mask to torch.Tensor
        # t_tile = torch.Tensor(t_tile)
        t_tile = ToTensor()(t_tile)
        t_mask = torch.Tensor(t_mask)
        return t_tile, t_mask

