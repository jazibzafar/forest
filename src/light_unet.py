from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from lightning.pytorch.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp
import os
from src.data_and_transforms import SegDataset
from torchmetrics import JaccardIndex
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.optim import AdamW
import lightning as L
import torch.nn as nn
import time


# @dataclass
# class ArgumentClass:
#     data_path: str = "/home/jazib/projects/SelfSupervisedLearning/gartow_seg/single_class/"
#     crop_size: int = 224
#     lr: float = 1e-3
#     batch_size: int = 64
#     num_workers: int = 4
#     train_ratio: float = 0.8
#     max_epochs: int = 100
#     output_dir: str = "./unet_bench_1/"


class LitUnet(L.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, args):
        super().__init__()
        # self.model = smp.Unet(encoder_name="resnet34",
        #                       encoder_weights="imagenet",
        #                       in_channels=4,
        #                       classes=1)
        #
        # self.dataset = SegDataBuffer(args.data_path,
        #                              args.crop_size)

        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)

        self.lr = args.lr
        self.loss = nn.BCEWithLogitsLoss()
        self.mIoU = JaccardIndex(task="binary")
        # self.train_dataset, self.val_dataset = random_split(self.dataset,
        #                                                     [args.train_ratio, 1 - args.train_ratio])
        # self.train_sampler = RandomSampler(self.train_dataset)
        # self.val_sampler = SequentialSampler(self.val_dataset)
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
        # targets = targets.squeeze(3)
        output = self.model(samples)
        output = output.squeeze(1)
        loss = self.loss(output, targets)
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1]
        output = self.model(samples)
        output = output.squeeze(1)
        loss = self.loss(output, targets)
        iou = self.mIoU(output, targets)
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/mIoU', iou, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1]
        # targets = targets.permute((0, 3, 1, 2))
        output = self.model(samples)
        output = output.squeeze(1)
        loss = self.loss(output, targets)
        iou = self.mIoU(output, targets)
        self.log('test/loss', loss, True)
        self.log('test/mIoU', iou, True)


def train_unet(args):
    # arguments = ArgumentClass()
    # unet_model = LitUnet(arguments)
    model = smp.Unet(encoder_name="resnet34",
                     encoder_weights="imagenet",
                     in_channels=4,
                     classes=1)
    # build the dataset
    train_path = os.path.join(args.data_path, 'train/')
    train_dataset = SegDataset(train_path, args.crop_size)
    val_path = os.path.join(args.data_path, 'val/')
    val_dataset = SegDataset(val_path, args.crop_size, train=False)

    unet_model = LitUnet(model, train_dataset, val_dataset, args)

    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,
                                          every_n_epochs=int(args.max_epochs / 3),
                                          # every_n_train_steps=int(args.max_steps / 5),
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
    test_path = os.path.join(args.data_path, 'test/')
    test_dataset = SegDataset(test_path, args.crop_size, train=False)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(dataset=test_dataset,
                             sampler=test_sampler,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             persistent_workers=True,
                             pin_memory=True,
                             drop_last=False, )
    trainer.test(model=unet_model, dataloaders=test_loader)
