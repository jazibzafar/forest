from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import random_split
from src.data_and_transforms import SegDataset
from timm.models.vision_transformer import PatchEmbed, Block
import torch.nn.functional as F
from functools import partial
import torch.nn as nn
import time
import torch
import lightning as L
from dataclasses import dataclass
from src.checkpoints import load_dino_checkpoint, prepare_arch
from torchmetrics import JaccardIndex
from src.nnblocks import ClipSegStyleDecoder
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import os


class LitSeg(L.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, args):
        super().__init__()
        self.save_hyperparameters(ignore=['models'])

        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)

        self.lr = args.lr
        self.loss = nn.MSELoss()
        self.mIoU = JaccardIndex(task="binary")

        # other things
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

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
                          drop_last=True, )

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          sampler=self.val_sampler,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=False,)

    def training_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1]
        # targets = targets.permute((0, 3, 1, 2))
        output = self.model(samples)
        output = torch.squeeze(output, 1)
        loss = self.loss(output, targets)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1]
        # targets = targets.permute((0, 3, 1, 2))
        output = self.model(samples)
        output = torch.squeeze(output, 1)
        loss = self.loss(output, targets)
        iou = self.mIoU(output, targets)
        self.log('val/loss', loss, True)
        self.log('val/mIoU', iou, True)


def train_segmentation(args):
    # create the model
    checkpoint = load_dino_checkpoint(args.checkpoint_path, args.checkpoint_key)
    model_backbone = prepare_arch(args.arch, checkpoint, args.patch_size)
    model = ClipSegStyleDecoder(backbone=model_backbone,
                                patch_size=args.patch_size,
                                reduce_dim=args.reduce_dim,
                                n_heads=args.n_heads)

    # build the dataset
    train_path = os.path.join(args.data_path, 'train/')
    train_dataset = SegDataset(train_path, args.crop_size)
    val_path = os.path.join(args.data_path, 'val/')
    val_dataset = SegDataset(val_path, args.crop_size)

    # lightning class
    light_seg = LitSeg(model, train_dataset, val_dataset, args)
    # logger and callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,
                                          every_n_epochs=int(args.max_epochs / 3),
                                          # every_n_train_steps=int(args.max_steps / 5),
                                          save_last=True)

    logger = TensorBoardLogger(save_dir=args.output_dir,
                               name="",
                               default_hp_metric=False)

    # trainer
    trainer = L.Trainer(accelerator='gpu',
                        max_epochs=args.max_epochs,
                        default_root_dir=args.output_dir,
                        enable_progress_bar=True,
                        logger=logger,
                        log_every_n_steps=10,
                        # check_val_every_n_epoch=10,
                        callbacks=[checkpoint_callback])

    # begin training
    print("beginning the training.")
    start = time.time()
    if args.resume:
        trainer.fit(model=light_seg, ckpt_path=args.resume_ckpt)
    else:
        trainer.fit(model=light_seg)
    end = time.time()
    print(f"training completed. Elapsed time {end - start} seconds.")
