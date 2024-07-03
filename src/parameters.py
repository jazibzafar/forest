from dataclasses import dataclass


@dataclass
class ClassificationParameters:
    # model
    arch: str = "vit_small"
    checkpoint_path: str = "/path/to/checkpoint.pth"
    checkpoint_key: str = "teacher"
    patch_size: int = 16
    num_classes: int = 10
    linear_eval: bool = False
    # data
    data_path: str = "/path/to/data/"
    input_size: int = 96
    # lightning
    lr: float = 0.00001
    batch_size: int = 32
    num_workers: int = 8
    # training
    output_dir: str = "/path/to/output/dir/"
    max_epochs: int = 150
    resume: bool = False
    resume_ckpt: str = "/path/to/checkpoint/"


@dataclass
class SegmentationParameters:
    # model
    arch: str = "vit_small"
    checkpoint_path: str = "/path/to/checkpoint.pth"
    checkpoint_key: str = "teacher"
    patch_size: int = 16
    # data
    data_path: str = "/path/to/data/"
    crop_size: int = 224
    # segmentation specific
    reduce_dim: int = 112
    n_heads: int = 4
    # lightning
    lr: float = 0.00001
    batch_size: int = 32
    num_workers: int = 8
    # training
    output_dir: str = "/path/to/output/dir/"
    max_epochs: int = 150
    resume: bool = False
    resume_ckpt: str = "/path/to/checkpoint/"
