import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from lightning.pytorch.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp
import os
from src.data_and_transforms import SegDataset, SegDataMemBuffer, Fortress, OAM_TCD
from torchmetrics import JaccardIndex
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.optim import AdamW
import lightning as L
import torch.nn as nn
import time
from src.checkpoints import load_dino_checkpoint, prepare_resnet, prepare_resnet2
import argparse

# from src.metrics import FocalLoss
from src.utils import write_dict_to_yaml
from src.predict_on_array import predict_on_array_cf


def get_args_parser_unet():
    parser = argparse.ArgumentParser('UNET parser')
    parser.add_argument('--arch', type=str)
    parser.add_argument('--checkpoint_path',default='/path/to/checkpoint/', type=str)
    parser.add_argument('--checkpoint_key', default='teacher', type=str)
    parser.add_argument('--data_path', default='/path/to/data/', type=str)
    parser.add_argument('--crop_size', default=224, type=int)
    parser.add_argument('--num_chans', default=4, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--output_dir', default='/path/to/data/', type=str)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--freeze_backbone', action='store_true')
    return parser


class LitUnet(L.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, args):
        super().__init__()
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)

        self.lr = args.lr
        self.num_classes = args.num_classes
        if self.num_classes == 1:
            self.loss = nn.BCEWithLogitsLoss()
            self.mIoU = JaccardIndex(task="binary")
        else:
            raise Exception("num_classes should be >0.")
        # other things
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

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

    def configure_optimizers(self):
        opt_params = [x for x in self.model.parameters() if x.requires_grad]
        opt = AdamW(params=opt_params,
                    lr=self.lr)
        return opt

    def training_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1]
        output = self.model(samples).squeeze()
        # output = output.squeeze(1)
        loss = self.loss(output, targets)
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1]
        output = self.model(samples).squeeze()
        # output = output.squeeze(1)
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
        targets = torch.Tensor(targets).to(self.device)
        loss = self.loss(output, targets)
        iou = self.mIoU(output, targets)
        self.log('test/loss', loss, True)
        self.log('test/mIoU', iou, True)


def train_unet(args):
    checkpoint = load_dino_checkpoint(args.checkpoint_path, args.checkpoint_key)
    backbone = prepare_resnet(args.arch, checkpoint, args.num_chans)
    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)
        print("backbone frozen.")
    else:
        for p in backbone.parameters():
            p.requires_grad_(True)
        print("backbone training.")

    model = smp.Unet(encoder_name=args.arch,
                     encoder_weights=None,
                     in_channels=args.num_chans,
                     classes=args.num_classes)
    model.encoder.load_state_dict(backbone.state_dict())
    # build the dataset
    train_path = os.path.join(args.data_path, 'train/')
    train_dataset = SegDataset(train_path, args.crop_size, train=True)
    val_path = os.path.join(args.data_path, 'val/')
    val_dataset = SegDataset(val_path, args.crop_size, train=False)

    unet_model = LitUnet(model, train_dataset, val_dataset, args)

    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,
                                          every_n_epochs=int(args.max_epochs / 3),
                                          # every_n_train_steps=int(args.max_steps / 10),
                                          save_last=True)
    earlystopping_callback = EarlyStopping(monitor='val/loss', mode='min', patience=3)
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
                        callbacks=[checkpoint_callback, earlystopping_callback])

    print("beginning the training.")
    start = time.time()
    trainer.fit(model=unet_model)
    # trainer.fit(model=dino)
    end = time.time()
    print(f"training completed. Elapsed time {end - start} seconds.")
    # begin testing
    # test_path = os.path.join(args.data_path, 'test/')
    # test_dataset = SegDataset(test_path, crop_size=992, train=False)
    # test_sampler = SequentialSampler(test_dataset)
    # test_loader = DataLoader(dataset=test_dataset,
    #                          sampler=test_sampler,
    #                          batch_size=1,
    #                          num_workers=args.num_workers,
    #                          persistent_workers=True,
    #                          pin_memory=True,
    #                          drop_last=False, )
    # trainer.test(model=unet_model, dataloaders=test_loader)


if __name__ == '__main__':
    args = get_args_parser_unet().parse_args()
    args_yml_fp = os.path.join(args.output_dir, "args.yaml")
    write_dict_to_yaml(args_yml_fp, args.__dict__)
    train_unet(args)
