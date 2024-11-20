from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from src.data_and_transforms import SegDataset, SegDataMemBuffer
import torch.nn as nn
import time
import torch
import lightning as L
from src.checkpoints import load_dino_checkpoint, prepare_vit, prepare_arch
from torchmetrics import JaccardIndex
from src.nnblocks import ClipSegStyleDecoder
from src.predict_on_array import predict_on_array_cf
from src.utils import write_dict_to_yaml, event_to_yml
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.callbacks import Callback
import os
import argparse
from lightning.pytorch.callbacks import ModelSummary
import logging
from torch.autograd import Variable
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def get_args_parser_semseg():
    parser = argparse.ArgumentParser('Segmentation Parser', add_help=True)
    parser.add_argument('--arch', default='vit_small', type=str,
                        # choices=['vit_tiny', 'vit_small', 'vit_base']
                        help="""Name of architecture to train.""")
    parser.add_argument('--checkpoint_path', default='/path/to/checkpoint/', type=str,
                        help='Please specify path to the saved checkpoint.')
    parser.add_argument('--checkpoint_key', default='teacher', type=str,
                        # choices = ['teacher', 'student']
                        help="""Please specify whether to use teacher or student network.""")
    parser.add_argument('--patch_size', default=16, type=int,
                        help="""Size in pixels of input square patches - default 16 (for 16x16 patches). 
                            Using smaller values leads to better performance but requires more memory""")
    parser.add_argument('--data_path', default='/path/to/data/', type=str,
                        help="""Please specify path to the training data.""")
    parser.add_argument('--input_size', default=224, type=int,
                        help="""size of the input to the network. should be divisible by 16. """)
    parser.add_argument('--num_classes', default=1, type=int,
                        help="""size of the input to the network. should be divisible by 16. """)
    # TODO: when adding more decoders below used the arch style input. see above.
    parser.add_argument('--simple_decoder', action='store_true',
                        help="""by default complex clip seg decoder is used. pass this if simple
                        decoder should be used instead""")
    parser.add_argument('--freeze_backbone', action='store_true',
                        help="""Pass this to freeze the backbone and train just the decoder.""")
    parser.add_argument('--reduce_dim', default=112, type=int,
                        help="""decoder reduces the dim of the input to this size.""")
    parser.add_argument('--decoder_head_count', default=4, type=int,
                        help="""number of decoder heads.""")
    parser.add_argument('--lr', default=0.00001, type=float,
                        help="""learning rate""")
    parser.add_argument('--batch_size', default=32, type=int,
                        help="""batch size""")
    parser.add_argument('--num_workers', default=8, type=int,
                        help="""number of dataloader workers""")
    parser.add_argument('--output_dir', default='/path/to/data/', type=str,
                        help="""Please specify path to the training data.""")
    parser.add_argument('--max_epochs', default=150, type=int,
                        help="""max number of training epochs.""")
    parser.add_argument('--resume', action='store_true',
                        help="""pass this if resuming training. False by default.""")
    parser.add_argument('--resume_ckpt', default='/path/to/resume/checkpoint/', type=str,
                        help='in case of resuming training. specify the path to the checkpoint.')
    parser.add_argument('--exp_name', default="exp/", type=str,
                        help='name of the experiment directory.')
    parser.add_argument("--device", default="gpu", type=str)
    return parser


def compute_class_weights(loader, num_classes):
    counts = torch.zeros(num_classes)
    for _, mask in loader:
        for cls in range(num_classes):
            counts[cls] += (mask == cls).sum()

    # Compute inverse weights for all classes
    weights = 1.0 / counts  # Calculate inverse weights for all classes
    weights = weights / weights.sum()  # Normalize to sum to 1

    return weights


class LitSeg(L.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, args, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)

        self.lr = args.lr
        self.num_classes = args.num_classes
        # self.loss = nn.CrossEntropyLoss()

        # Apply class weights to CrossEntropyLoss if provided
        if class_weights is not None:
            print("\nclass weights not none!")
            self.loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            print("\nclass weights none!")
            # self.loss = nn.MSELoss()
            self.loss = nn.CrossEntropyLoss()

        if self.num_classes == 1:
            self.mIoU = JaccardIndex(task="binary")
        elif self.num_classes > 1:
            self.mIoU = JaccardIndex(task="multiclass", num_classes=self.num_classes)
        else:
            raise Exception("num_classes should be >0.")
        # other things
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    @staticmethod
    def multichannel_output_to_mask(img):
        print("\nin multi-channel to mask: shape and type of image: ", img.size(), img.dtype)
        softmax = torch.nn.Softmax(dim=1)
        img = softmax(img)
        # img = img
        #print("\nin multi-channel to mask: shape and type of image after softmax: ", img.size(), img.dtype)
        mask = torch.argmax(img, dim=1)
        print("\nin multi-channel to mask: shape and type of mask after argmax: ", mask.size(), mask.dtype)
        return mask
        #return img

    def configure_optimizers(self):
        regularized, not_regularized = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if n.endswith(".bias") or len(p.shape) == 1:
                not_regularized.append(p)
            else:
                regularized.append(p)
        param_groups = [{'params': regularized},
                        {'params': not_regularized, 'weight_decay': 0.}]  # weight decay is 0 because of the scheduler
        opt = torch.optim.AdamW(param_groups, self.lr)
        # opt_params = [x for x in self.model.parameters() if x.requires_grad]
        # opt = torch.optim.AdamW(params=opt_params,
        #                         lr=self.lr)
        return opt

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size,
                          pin_memory=True,
                          persistent_workers=True,
                          shuffle=True,
                          drop_last=True, )

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          sampler=self.val_sampler,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          drop_last=False,)

    def training_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1].squeeze()
        # targets = targets.permute((0, 3, 1, 2))
        output = self.model(samples).squeeze()
        # output = torch.squeeze(output, 1)
        # if self.num_classes > 1:
        #     print("training: multi-class convertion from output to mask applied")
        #     output = self.multichannel_output_to_mask(output)
        #     print("output shape and dytpe within loop: ", output.size(), output.dtype)
        print("\nLogits range:", output.min().item(), output.max().item())
        print("Target unique values:", torch.unique(targets))

        # try running with random data to see if loss function works (because it currently only produces 0)
        # output = torch.randn(32, 4, 320, 320, dtype=torch.float32)
        # targets = torch.randint(0, 4, (32, 320, 320), dtype=torch.int64)

        loss = self.loss(output, targets)
        loss = Variable(loss, requires_grad=True)
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True) # log inherited from LightningModule
        return loss

    def validation_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1].squeeze()
        # targets = targets.permute((0, 3, 1, 2))
        output = self.model(samples).squeeze()
        # output = torch.squeeze(output, 1)
        # if self.num_classes > 1:
        #     output = self.multichannel_output_to_mask(output)
        loss = self.loss(output, targets)
        iou = self.mIoU(output, targets)
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/mIoU', iou, prog_bar=True, on_step=False, on_epoch=True)
    #
    # def test_step(self, batch, batch_idx):
    #     sample = batch[0].squeeze(0).cpu().numpy()
    #     targets = batch[1].squeeze(0).cpu().numpy()
    #     pred = predict_on_array_cf(self.model.eval(), sample,
    #                                in_shape=(4, 320, 320),
    #                                out_bands=args.num_classes,
    #                                drop_border=0,
    #                                stride=26,
    #                                batchsize=1,
    #                                augmentation=True)
    #     output = pred["prediction"]
    #     output = torch.Tensor(output).squeeze(0).to(self.device)
    #     if self.num_classes > 1:
    #         output = self.multichannel_output_to_mask(output)
    #     targets = torch.Tensor(targets).to(self.device)
    #     loss = self.loss(output, targets)
    #     iou = self.mIoU(output, targets)
    #     self.log('test/loss', loss, True)
    #     self.log('test/mIoU', iou, True)

    def test_step(self, batch, batch_idx):
        sample = batch[0].squeeze(0).cpu().numpy()
        targets = batch[1].cpu().numpy()
        pred = predict_on_array_cf(self.model.eval(), sample,
                                   in_shape=(4, 256, 256),
                                   out_bands=self.args.num_classes,
                                   drop_border=0,
                                   stride=26,
                                   batchsize=1,
                                   augmentation=True,
                                   device=self.device,
                                   network_input_dtype=torch.bfloat16)
        output = pred["prediction"]
        output = torch.Tensor(output).to(self.device)
        output = output.unsqueeze(0)
        # if self.num_classes > 1:
        #     output = self.multichannel_output_to_mask(output)
        targets = torch.Tensor(targets).to(self.device)
        targets = targets.round().to(torch.int64)
        loss = self.loss(output, targets)
        iou = self.mIoU(output, targets)
        self.log('test/loss', loss, True)
        self.log('test/mIoU', iou, True)


def visualize_predictions(model, dataloader, device="cuda", num_samples=5):
    """
    Visualizes predictions for a few samples from the dataloader.

    Parameters:
    - model: The trained model for prediction.
    - dataloader: The dataloader containing the test samples.
    - device: Device to run the model on ('cpu' or 'cuda').
    - num_samples: Number of samples to visualize.
    """
    model.eval()
    model.to(device)

    # Fetch a few samples from the dataloader
    for idx, (images, targets) in enumerate(dataloader):
        if idx >= num_samples:
            break
        images = images.to(device)
        #print("image shape: ", images.size())
        targets = targets.to(device)
        #print("target shape: ", targets.size())

        with torch.no_grad():
            predictions = model(images).cpu()
            #print("prediction shape: ", predictions.size())

        # Visualize the results
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        #print("shape of images: ", images.size())
        #print("shape of images[0].numpy().transpose(1, 2, 0): ", images[0].cpu().numpy().transpose(1, 2, 0).shape)
        #print("shape of images[0].squeeze(): ", images[0].size())
        rgb_image = images[0].cpu().numpy().transpose(1, 2, 0)[:, :, :3]
        print("shape of rgb image: ", rgb_image.shape)
        rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

        ax[0].imshow(rgb_image)
        ax[0].set_title("Input Image")

        ax[1].imshow(targets[0].cpu().numpy().reshape(256, 256), cmap='gray')
        ax[1].set_title("Ground Truth")

        print("shape of predictions: ", predictions.size())
        print("shape of predictions[0].squeeze(): ", predictions[0].squeeze().size())
        ax[2].imshow(predictions[0].squeeze().cpu().numpy(), cmap='gray')
        ax[2].set_title("Prediction")

        plt.show()


def train_segmentation(args):
    # create the model
    checkpoint = load_dino_checkpoint(args.checkpoint_path, args.checkpoint_key)
    model_backbone = prepare_arch(args.arch, checkpoint, args.patch_size)
    model = ClipSegStyleDecoder(backbone=model_backbone,
                                patch_size=args.patch_size,
                                reduce_dim=args.reduce_dim,
                                n_heads=args.decoder_head_count,
                                simple_decoder=args.simple_decoder,
                                freeze_backbone=args.freeze_backbone,
                                num_classes=args.num_classes)

    # build the dataset
    train_path = os.path.join(args.data_path, 'train')
    # train_dataset = SegDataMemBuffer(train_path, args.input_size, crop_overlap=0.4)
    train_dataset = SegDataset(train_path, crop_size=args.input_size)
    val_path = os.path.join(args.data_path, 'val')
    # val_dataset = SegDataMemBuffer(val_path, args.input_size, crop_overlap=0.4)
    val_dataset = SegDataset(val_path, crop_size=args.input_size)

    # # Compute class weights
    # if args.num_classes > 1:
    #     class_weights = compute_class_weights(train_dataset, args.num_classes).to(args.device)

    # experiment directory
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    # lightning class
    # if args.num_classes > 1:
    #     light_seg = LitSeg(model, train_dataset, val_dataset, args, class_weights)
    # else:
    #     light_seg = LitSeg(model, train_dataset, val_dataset, args)

    light_seg = LitSeg(model, train_dataset, val_dataset, args)

    # logger and callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,
                                          every_n_epochs=int(args.max_epochs / 3),
                                          # every_n_train_steps=int(args.max_steps / 5),
                                          enable_version_counter=False,
                                          save_last=True)
    #earlystopping_callback = EarlyStopping(monitor='val/loss', mode='min', patience=3)
    logger = TensorBoardLogger(save_dir=exp_dir,
                               name="",
                               version="",
                               default_hp_metric=False)

    # trainer
    trainer = L.Trainer(accelerator=args.device,
                        max_epochs=args.max_epochs,
                        default_root_dir=args.output_dir,
                        enable_progress_bar=True,
                        logger=logger,
                        log_every_n_steps=10,
                        check_val_every_n_epoch=10,
                        num_sanity_val_steps=0,
                        precision="bf16-mixed",
                        gradient_clip_val=0.5,
                        # fast_dev_run=True, # run with just one epoch for fast development
                        callbacks=[checkpoint_callback,
                                   #earlystopping_callback,
                                   ])
    # trainer = L.Trainer(fast_dev_run=True)
    # begin training
    print("beginning the training.")
    start = time.time()
    if args.resume:
        trainer.fit(model=light_seg, ckpt_path=args.resume_ckpt)
    else:
        trainer.fit(model=light_seg) # invisible arguments here: train_dataloader and val_dataloader
    end = time.time()
    print(f"training completed. Elapsed time {end - start} seconds.")

    # begin testing
    test_path = os.path.join(args.data_path, 'test')
    test_dataset = SegDataset(test_path, 256, train=False)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(dataset=test_dataset,
                             sampler=test_sampler,
                             batch_size=1,
                             num_workers=args.num_workers,
                             persistent_workers=True,
                             pin_memory=True,
                             drop_last=False, )
    trainer.test(model=light_seg, dataloaders=test_loader)

    # Visualize predictions on some test tiles
    #visualize_predictions(light_seg.model, test_loader, device=args.device)

    # writing stats from tensorboard logs to yml
    event_to_yml(exp_dir)


ARCH='vit_small'
CKPT_PATH='/data_hdd/jazibmodels/dino_vit-s_32_500k_randonly/epoch=6-step=500000.ckpt'
CKPT_KEY='teacher'
DATA_PATH='/data_hdd/pauline/dataset/swf/256x256/'
MAX_EPOCHS=100
NUM_CLASSES=4
DEV="cuda"
INPUT_SIZE=256
OUTPUT_DIR="./test/"
EXP_NAME='v26_100_10_256x256_fixed_no_weights'
LR=0.01

if __name__ == '__main__':
    # args = get_args_parser_semseg().parse_args()
    args = get_args_parser_semseg().parse_args(f"--arch {ARCH} "
                                               f"--checkpoint_path {CKPT_PATH} "
                                               f"--checkpoint_key {CKPT_KEY} "
                                               f"--data_path {DATA_PATH} "
                                               f"--freeze_backbone "
                                               f"--input_size {INPUT_SIZE} "
                                               f"--num_classes {NUM_CLASSES} "
                                               f"--output_dir {OUTPUT_DIR} "
                                               f"--max_epochs {MAX_EPOCHS} "
                                               f"--device {DEV} "
                                               f"--lr {LR} "
                                               f"--exp_name {EXP_NAME}".split()
                                               )
    # write args to a yml file in output dir
    args_yml_fp = os.path.join(args.output_dir, "args.yaml")
    write_dict_to_yaml(args_yml_fp, args.__dict__)
    train_segmentation(args)
