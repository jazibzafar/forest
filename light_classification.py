import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import lightning as L
from src.metrics import accuracy
import os
import time
from src.nnblocks import LinearClassifier
from src.utils import write_dict_to_yaml
from src.data_and_transforms import UsualTransform, CenterCrop
from torchvision.datasets import ImageFolder
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.device_stats_monitor import DeviceStatsMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from src.checkpoints import load_dino_checkpoint, prepare_vit, prepare_resnet_model
from src.data_and_transforms import img_loader
import argparse
from src.utils import event_to_yml
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelSummary


torch.manual_seed(222)


def get_args_parser_class():
    parser = argparse.ArgumentParser('Classification Parser', add_help=True)
    parser.add_argument('--arch', default='', type=str,
                        # choices=['vit_tiny', 'vit_small', 'vit_base', 'resnet50']
                        help="""Name of architecture to train.""")
    parser.add_argument('--checkpoint_path', default='/path/to/checkpoint/', type=str,
                        help='Please specify path to the saved checkpoint.')
    parser.add_argument('--checkpoint_key', default='teacher', type=str,
                        # choices = ['teacher', 'student']
                        help="""Please specify whether to use teacher or student network.""")
    parser.add_argument('--patch_size', default=16, type=int,
                        help="""Size in pixels of input square patches - default 16 (for 16x16 patches). 
                        Using smaller values leads to better performance but requires more memory""")
    parser.add_argument('--num_classes', default=10, type=int,
                        help="""number of data classes.""")
    parser.add_argument('--linear_eval', action='store_true',
                        help="""Please specify whether to train the entire network (default) or
                        just the classification layer.""")
    parser.add_argument('--data_path', default='/path/to/data/', type=str,
                        help="""Please specify path to the training data.""")
    parser.add_argument('--input_size', default=96, type=int,
                        help="""size of images to be fed to the network. independent of actual
                        image size. should be divisible by 16.""")
    parser.add_argument('--num_chans', default=4, type=int,
                        help="""Number of input channels""")
    parser.add_argument('--lr', default=0.00001, type=float,
                        help="""learning rate""")
    parser.add_argument('--batch_size', default=32, type=int,
                        help="""batch size""")
    parser.add_argument('--num_workers', default=8, type=int,
                        help="""number of dataloader workers""")
    parser.add_argument('--output_dir', default='/path/to/data/', type=str,
                        help="""Please specify path to the training data.""")
    parser.add_argument('--max_epochs', default=100, type=int,
                        help="""max number of training epochs.""")
    parser.add_argument('--resume', action='store_true',
                        help="""pass this if resuming training. False by default.""")
    parser.add_argument('--resume_ckpt', default='/path/to/resume/checkpoint/', type=str,
                        help='in case of resuming training. specify the path to the checkpoint.')
    return parser


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


class LitClass(L.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, args):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)

        self.lr = self.args.lr
        self.loss = nn.CrossEntropyLoss()

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_dataset,
                                  sampler=self.train_sampler,
                                  batch_size=self.args.batch_size,
                                  num_workers=self.args.num_workers,
                                  pin_memory=True,
                                  persistent_workers=True,
                                  drop_last=True, )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(dataset=self.val_dataset,
                                sampler=self.val_sampler,
                                batch_size=self.args.batch_size,
                                num_workers=self.args.num_workers,
                                persistent_workers=True,
                                pin_memory=True,
                                drop_last=False, )
        return val_loader

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
        return opt

    def training_step(self, batch):
        samples = batch[0]
        targets = batch[1]
        output = self.model(samples)
        loss = self.loss(output, targets)
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1]
        output = self.model(samples)
        loss = self.loss(output, targets)
        acc1, _ = accuracy(output, targets, topk=(1, 5))
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/acc', acc1, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1]
        output = self.model(samples)
        loss = self.loss(output, targets)
        acc1, _ = accuracy(output, targets, topk=(1, 5))
        self.log('test/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test/acc', acc1, prog_bar=True, on_step=False, on_epoch=True)


def train_classification(args):
    # create the model
    checkpoint = load_dino_checkpoint(args.checkpoint_path, args.checkpoint_key)

    if args.arch.startswith("vit_"):
        model_backbone = prepare_vit(args.arch, checkpoint, args.patch_size, args.num_chans)
        linear_classifier = LinearClassifier(model_backbone.embed_dim, args.num_classes)
    elif args.arch.startswith("res"):
        model_backbone = prepare_resnet_model(args.arch, checkpoint, args.num_chans)
        linear_classifier = LinearClassifier(model_backbone.fc.out_features, args.num_classes)
    else:
        raise Exception("Please select a valid model.")


    model_backbone.eval() if args.linear_eval else model_backbone.train()
    model = nn.Sequential(model_backbone, linear_classifier)
    # build the dataset
    train_transform = UsualTransform(args.input_size)
    train_path = os.path.join(args.data_path, "train")
    train_dataset = ImageFolder(root=train_path, transform=train_transform, loader=img_loader)
    val_transform = CenterCrop(args.input_size)
    val_path = os.path.join(args.data_path, "val")
    val_dataset = ImageFolder(root=val_path, transform=val_transform, loader=img_loader)

    # lightning class
    light_class = LitClass(model, train_dataset, val_dataset, args)

    # logger and callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,
                                          every_n_epochs=int(args.max_epochs / 3),
                                          # every_n_train_steps=int(args.max_steps / 5),
                                          save_last=True)
    earlystopping_callback = EarlyStopping(monitor='val/loss', mode='min', patience=5)
    logger = TensorBoardLogger(save_dir=args.output_dir,
                               name="",
                               default_hp_metric=False)
    # mymonitor = DeviceStatsMonitor()
    # model_summary = ModelSummary(max_depth=-1)

    # trainer
    trainer = L.Trainer(accelerator='gpu',
                        max_epochs=args.max_epochs,
                        default_root_dir=args.output_dir,
                        enable_progress_bar=True,
                        logger=logger,
                        log_every_n_steps=10,
                        check_val_every_n_epoch=3,
                        profiler='simple',
                        callbacks=[checkpoint_callback, earlystopping_callback])

    # begin training
    print("beginning the training.")
    start = time.time()
    if args.resume:
        trainer.fit(model=light_class, ckpt_path=args.resume_ckpt)
    else:
        trainer.fit(model=light_class)

    end = time.time()
    print(f"training completed. Elapsed time {end - start} seconds.")

    # begin testing
    test_transform = CenterCrop(args.input_size)
    test_path = os.path.join(args.data_path, "test")
    test_dataset = ImageFolder(root=test_path, transform=test_transform, loader=img_loader)
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(dataset=test_dataset,
                             sampler=test_sampler,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             persistent_workers=True,
                             pin_memory=True,
                             drop_last=False, )
    trainer.test(model=light_class, dataloaders=test_loader)

    # writing stats from tensorboard logs to yml
    event_to_yml(os.path.join(args.output_dir, "version_0"))


if __name__ == '__main__':
    args = get_args_parser_class().parse_args()
    # write args to a yml file in output dir
    args_yml_fp = os.path.join(args.output_dir, "args.yaml")
    write_dict_to_yaml(args_yml_fp, args.__dict__)
    train_classification(args)
