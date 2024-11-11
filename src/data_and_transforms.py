import albumentations as A
import os
from tifffile import imread
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import get_worker_info
from itertools import islice
import torch
import random
# from osgeo import gdal
# import webdataset as wds

# def img_loader(path, dim_ordering="CHW"):
#     if dim_ordering == "HWC":
#         return imread(path)
#     elif dim_ordering == "CHW":
#         return imread(path).transpose((2,0,1))
#     else:
#         raise RuntimeError(f"Received wrong dim ordering '{dim_ordering}', must be either CHW or HWC.")


def img_loader(path):
    return imread(path)


def get_crop_indices(length, window, overlap):
    if length == window: return [0]
    stride = int((1 - overlap) * window)
    num_strides = length // stride
    indices = []
    for i in range(num_strides+1):
        idx = i*stride
        if not idx + window >= length:
            indices.append(idx)
        else:
            last_index = length - window
            #  to prevent repetition of last index
            if not last_index == indices[-1]: indices.append(length - window)
    return indices


def create_crops(img, crop_size, overlap):
    img_h, img_w = img.shape[0:2]
    x_coords = get_crop_indices(img_w, crop_size, overlap)
    y_coords = get_crop_indices(img_h, crop_size, overlap)
    crops = []
    points = []
    for i in x_coords:
        for j in y_coords:
            crops.append(img[i:i+crop_size, j:j+crop_size])
            points.append((i,j))
    return crops, points


class UsualTransform:
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


class DINOTransform:
    def __init__(self, global_crop_size, global_crops_scale, local_crop_size, local_crops_scale, local_crops_number):
        self.global_crop_size = global_crop_size
        self.global_crops_scale = global_crops_scale
        self.local_crop_size = local_crop_size
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number

        img_augmentations = A.Compose([
            A.HorizontalFlip(p=0.1),  # 0.5
            A.RandomRotate90(p=0.1),  # 0.2
            A.RandomBrightnessContrast(brightness_limit=(0.2, 0.3),
                                       contrast_limit=(0.2, 0.3),
                                       p=0.02),  # 0.1
            A.RandomGamma(gamma_limit=(100, 140), p=0.02),  # 0.1
            A.RandomToneCurve(scale=0.1, p=0.02)  # 0.1
        ])

        # Gaussian blur is not used atm.
        # Scale is used atm.
        self.global_transform_1 = A.Compose([
            A.RandomCrop(height=self.global_crop_size,
                         width=self.global_crop_size,
                         always_apply=True),
            img_augmentations,
        ])

        # Gaussian blur is not used atm.
        # Scale is used atm.
        # Solarization is not used atm.
        self.global_transform_2 = A.Compose([
            A.RandomCrop(height=self.global_crop_size,
                         width=self.global_crop_size,
                         always_apply=True),
                                #  interpolation=cv2.INTER_CUBIC),
            img_augmentations,
        ])

        self.local_transform = A.Compose([
            A.RandomCrop(height=self.local_crop_size,
                         width=self.local_crop_size,
                         always_apply=True),
                                #  interpolation=cv2.INTER_CUBIC),
            img_augmentations,
        ])

    def __call__(self, image):
        # Make sure image is a np array
        if type(image) == 'torch.Tensor':
            image = image.numpy()

        crops = [self.global_transform_1(image=image)['image'], self.global_transform_2(image=image)['image']]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image=image)['image'])
        crops = [ToTensor()(crop) for crop in crops]
        return crops


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
        # for i in range(len(self.tile_list)):
        #     print(f"{self.tile_list[i]} -----> {self.mask_list[i]}")
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

        # convert mask to single-class case
        mask[mask == 2] = 1
        mask[mask == 3] = 1

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


# TODO: in case mask_chans > 1, convert mask img into multi-channel img.
class SegDataMemBuffer(Dataset):
    def __init__(self, path, crop_size, crop_overlap=0., tile_chans=4, mask_chans=1, shuffle=True, tile_transform=None):
        super().__init__()
        self.tile_chans = tile_chans
        self.mask_chans = mask_chans
        self.data_buffer = self.create_data_buffer(path, crop_size, crop_overlap, shuffle)
        self.tile_transform = tile_transform
        self.common_transform = A.Compose([A.HorizontalFlip(p=0.3),  # 0.5
                                           A.RandomRotate90(p=0.1), ])   # 0.2

    @staticmethod
    def create_data_buffer(path, crop_size, crop_overlap, shuffle):
        data_buffer = []
        tile_path = os.path.join(path, "tiles")
        mask_path = os.path.join(path, "masks")
        tile_list = sorted(os.listdir(tile_path))
        mask_list = sorted(os.listdir(mask_path))
        for ti, ma in zip(tile_list, mask_list):
            tile = imread(os.path.join(tile_path, ti))
            mask = imread(os.path.join(mask_path, ma))
            stacked = np.dstack((tile, mask))
            slices, _ = create_crops(stacked, crop_size, crop_overlap)
            data_buffer.extend(slices)
        if shuffle:
            random.shuffle(data_buffer)
        return data_buffer

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self, index):
        img = self.data_buffer[index]
        img = self.common_transform(image=img)['image']
        assert img.shape[2] == self.mask_chans + self.tile_chans, \
            "make sure tile and mask channels add up to input channels"
        tile = img[:, :, 0:self.tile_chans]
        mask = img[:, :, self.tile_chans:self.tile_chans + self.mask_chans]
        if self.tile_transform:
            tile = self.tile_transform(tile)
        tile = ToTensor()(tile)
        mask = torch.Tensor(mask)  # to prevent normalizing
        return tile, mask


# class GeoWebDataset(IterableDataset):
#     def __init__(self,
#                  *,
#                  root,
#                  n_bands,
#                  augmentations,
#                  num_nodes=1,
#                  num_shards=100,
#                  imgs_per_shard=250):
#         self.root = root
#         self.n_bands = n_bands
#         self.augmentations = augmentations
#         self.num_nodes = num_nodes
#         self.num_shards = num_shards
#         self.imgs_per_shard = imgs_per_shard
#         self.cropsize = 224
#         #
#         self.num_patches = 1000000000000  # set it to sth really high for now, so that the generator doesnt get exhausted during trainng
#
#         self.dataset = wds.DataPipeline(wds.ResampledShards(self.root),
#                                         wds.split_by_node,
#                                         wds.split_by_worker,
#                                         self.split_by_dataloader_worker,
#                                         # self.printer,
#                                         wds.shuffle(8),
#                                         wds.tarfile_to_samples(),
#                                         wds.to_tuple("tif"),
#                                         wds.map(GeoWebDataset.preprocess),
#                                         self.slicer,
#                                         wds.shuffle(100),  # buffer of size 100
#                                         wds.map(self.augmentations),
#                                         ).with_length(self.num_patches)
#
#     @staticmethod
#     def read_geotif_from_bytestream(data: bytes) -> np.ndarray:
#         gdal.FileFromMemBuffer("/vsimem/tmp", data)
#         ds = gdal.Open("/vsimem/tmp")
#         bands = ds.RasterCount
#         ys = ds.RasterYSize
#         xs = ds.RasterXSize
#         # arr = np.empty((bands, ys, xs), dtype="float32")  # CHW
#         # for b in range(1, bands + 1):
#         #     band = ds.GetRasterBand(b)
#         #     arr[b - 1, :, :] = band.ReadAsArray()
#         # return torch.from_numpy(arr) / 255
#         arr = np.empty((ys, xs, bands), dtype="uint8")  # HWC
#         for b in range(1, bands + 1):
#             band = ds.GetRasterBand(b)
#             arr[:, :, b - 1] = band.ReadAsArray()
#         return arr
#
#     @staticmethod
#     def preprocess(sample):
#         return GeoWebDataset.read_geotif_from_bytestream(sample[0])
#
#     @staticmethod
#     def slice_image(samples, tilesize: int):
#         for img in samples:
#             for y in range(0, img.shape[1], tilesize):
#                 for x in range(0, img.shape[2], tilesize):
#                     yield img[:, y:y + tilesize, x:x + tilesize]  # CHW
#
#     @staticmethod
#     def split_by_dataloader_worker(iterable):
#         worker_info = get_worker_info()
#         print("Worker info: ", worker_info)
#         if worker_info is None:
#             return iterable
#         else:
#             worker_num = worker_info.num_workers
#             worker_id = worker_info.id
#             sliced_data = islice(iterable, worker_id, None, worker_num)
#             return sliced_data
#
#     @staticmethod
#     def printer(iterable):
#         for x in iterable:
#             print("From node X, worker Y, dataloader worker Z: ")
#             print(x)
#             yield x
#
#     def slicer(self, img):
#         return GeoWebDataset.slice_image(img, self.cropsize)
#
#     def __iter__(self):
#         return iter(self.dataset)
#
#     def __len__(self):
#         return self.imgs_per_shard * self.num_shards * 100  # each image has 100 crops.
