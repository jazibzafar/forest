from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from src.data_and_transforms import SegDataset, SegDataMemBuffer
import torch.nn as nn
import time
import torch
import lightning as L
from src.checkpoints import load_dino_checkpoint, prepare_vit
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
    return parser


class MyPrintingCallback(Callback):
    def on_fit_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule"):
        print("fit starting")

    def on_sanity_check_start(self, trainer, pl_module):
        print("sanity checking")

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


class FitCallback(Callback):
    def setup(self, trainer, pl_module, stage: str) -> None:
        print(f"Setting up {stage}")

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        print("sanity checking")

    def on_fit_start(self, trainer, pl_module) -> None:
        print("fitting")

    def on_train_start(self, trainer, pl_module) -> None:
        print("train starting")


class LitSeg(L.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, args):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)

        self.lr = args.lr
        self.loss = nn.BCEWithLogitsLoss()
        self.mIoU = JaccardIndex(task="binary")
        self.num_classes = args.num_classes

        # other things
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    @staticmethod
    def multichannel_output_to_mask(img):
        softmax = torch.nn.Softmax(dim=1)
        img = softmax(img)
        mask = torch.argmax(img, dim=1)
        return mask

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
        if self.num_classes > 1:
            output = self.multichannel_output_to_mask(output)
        loss = self.loss(output, targets)
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1].squeeze()
        # targets = targets.permute((0, 3, 1, 2))
        output = self.model(samples).squeeze()
        # output = torch.squeeze(output, 1)
        if self.num_classes > 1:
            output = self.multichannel_output_to_mask(output)
        loss = self.loss(output, targets)
        iou = self.mIoU(output, targets)
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/mIoU', iou, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        sample = batch[0].squeeze(0).cpu().numpy()
        targets = batch[1].squeeze(0).cpu().numpy()
        pred = predict_on_array_cf(self.model.eval(), sample,
                                   in_shape=(4, 992, 992),
                                   out_bands=args.num_classes,
                                   drop_border=0,
                                   stride=26,
                                   batchsize=1,
                                   augmentation=True)
        output = pred["prediction"]
        output = torch.Tensor(output).squeeze(0).to(self.device)
        if self.num_classes > 1:
            output = self.multichannel_output_to_mask(output)
        targets = torch.Tensor(targets).to(self.device)
        loss = self.loss(output, targets)
        iou = self.mIoU(output, targets)
        self.log('test/loss', loss, True)
        self.log('test/mIoU', iou, True)


def train_segmentation(args):
    # create the model
    checkpoint = load_dino_checkpoint(args.checkpoint_path, args.checkpoint_key)
    model_backbone = prepare_vit(args.arch, checkpoint, args.patch_size)
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

    # lightning class
    light_seg = LitSeg(model, train_dataset, val_dataset, args)
    # logger and callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,
                                          every_n_epochs=int(args.max_epochs / 3),
                                          # every_n_train_steps=int(args.max_steps / 5),
                                          save_last=True)
    earlystopping_callback = EarlyStopping(monitor='val/loss', mode='min', patience=3)
    my_callback = MyPrintingCallback()
    mymonitor = DeviceStatsMonitor()
    model_summary = ModelSummary(max_depth=-1)
    fitcb = FitCallback()

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
                        check_val_every_n_epoch=10,
                        num_sanity_val_steps=0,
                        callbacks=[fitcb,
                                   checkpoint_callback,
                                   earlystopping_callback,])
    # trainer = L.Trainer(fast_dev_run=True)
    # begin training
    print("beginning the training.")
    start = time.time()
    if args.resume:
        trainer.fit(model=light_seg, ckpt_path=args.resume_ckpt)
    else:
        trainer.fit(model=light_seg)
    end = time.time()
    print(f"training completed. Elapsed time {end - start} seconds.")

    # begin testing
    test_path = os.path.join(args.data_path, 'test')
    test_dataset = SegDataset(test_path, 992, train=False)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(dataset=test_dataset,
                             sampler=test_sampler,
                             batch_size=1,
                             num_workers=args.num_workers,
                             persistent_workers=True,
                             pin_memory=True,
                             drop_last=False, )
    trainer.test(model=light_seg, dataloaders=test_loader)

    # writing stats from tensorboard logs to yml
    event_to_yml(os.path.join(args.output_dir, "version_0"))


if __name__ == '__main__':
    args = get_args_parser_semseg().parse_args()
    # write args to a yml file in output dir
    args_yml_fp = os.path.join(args.output_dir, "args.yaml")
    write_dict_to_yaml(args_yml_fp, args.__dict__)
    train_segmentation(args)
