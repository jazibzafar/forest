import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import lightning as L
from src.metrics import accuracy
from torchmetrics import JaccardIndex
import os
import time
from src.nnblocks import LinearClassifier
from src.data_and_transforms import UsualTransform, CenterCrop
from torchvision.datasets import ImageFolder
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from src.checkpoints import load_dino_checkpoint, prepare_arch
from src.data_and_transforms import img_loader


class LitDownstream(L.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, args):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_sampler = RandomSampler(self.train_dataset)
        self.val_sampler = SequentialSampler(self.val_dataset)
        self.test_sampler = SequentialSampler(self.test_dataset)
        self.lr = self.args.lr

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
                          shuffle=True,
                          drop_last=True, )

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          sampler=self.val_sampler,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=False,)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          sampler=self.test_sampler,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=True,
                          drop_last=False, )

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass


class LitClass(LitDownstream):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, args):
        super().__init__(model, train_dataset, val_dataset, test_dataset, args)
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
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


class LitSemSeg(LitDownstream):
    def __init__(self, model, train_dataset, val_dataset, test_dataset, args):
        super().__init__(model, train_dataset, val_dataset, test_dataset, args)
        self.loss = nn.BCEWithLogitsLoss()
        self.mIoU = JaccardIndex(task="binary")

    def training_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1]
        # targets = targets.permute((0, 3, 1, 2))
        output = self.model(samples)
        output = torch.squeeze(output, 1)
        loss = self.loss(output, targets)
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1]
        # targets = targets.permute((0, 3, 1, 2))
        output = self.model(samples)
        output = torch.squeeze(output, 1)
        loss = self.loss(output, targets)
        iou = self.mIoU(output, targets)
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/mIoU', iou, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        samples = batch[0]
        targets = batch[1]
        # targets = targets.permute((0, 3, 1, 2))
        output = self.model(samples)
        output = torch.squeeze(output, 1)
        loss = self.loss(output, targets)
        iou = self.mIoU(output, targets)
        self.log('test/loss', loss, True)
        self.log('test/mIoU', iou, True)


def train_downstream_task(args):
    # logger and callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,
                                          every_n_epochs=int(args.max_epochs / 3),
                                          # every_n_train_steps=int(args.max_steps / 5),
                                          save_last=True)
    earlystopping_callback = EarlyStopping(monitor='val/loss', mode='min', patience=3)
    logger = TensorBoardLogger(save_dir=args.output_dir,
                               name="",
                               default_hp_metric=False)
    # create the model
    checkpoint = load_dino_checkpoint(args.checkpoint_path, args.checkpoint_key)
    model_backbone = prepare_arch(args.arch, checkpoint, args.patch_size)
    # task specific model and data specification.
    match args.task:
        case "class":
            linear_classifier = LinearClassifier(model_backbone.embed_dim, args.num_classes)
            model_backbone.eval() if args.linear_eval else model_backbone.train()
            model = nn.Sequential(model_backbone, linear_classifier)

            train_transform = UsualTransform(args.input_size)
            train_path = os.path.join(args.data_path, "train")
            train_dataset = ImageFolder(root=train_path, transform=train_transform, loader=img_loader)
            val_transform = CenterCrop(args.input_size)
            val_path = os.path.join(args.data_path, "val")
            val_dataset = ImageFolder(root=val_path, transform=val_transform, loader=img_loader)
            test_transform = CenterCrop(args.input_size)
            test_path = os.path.join(args.data_path, "test")
            test_dataset = ImageFolder(root=test_path, transform=test_transform, loader=img_loader)

            lit_task = LitClass(model, train_dataset, val_dataset, test_dataset, args)
        case "semseg":
            model = ClipSegStyleDecoder(backbone=model_backbone,
                                        patch_size=args.patch_size,
                                        reduce_dim=args.reduce_dim,
                                        n_heads=args.n_heads,
                                        complex_trans_conv=args.complex_trans_conv,
                                        freeze_backbone=args.freeze_backbone)

            # build the dataset
            train_path = os.path.join(args.data_path, 'train/')
            train_dataset = SegDataset(train_path, args.crop_size)
            val_path = os.path.join(args.data_path, 'val/')
            val_dataset = SegDataset(val_path, args.crop_size, train=False)
            test_path = os.path.join(args.data_path, 'test/')
            test_dataset = SegDataset(test_path, args.crop_size, train=False)

            lit_task = LitSemSeg(model, train_dataset, val_dataset, test_dataset, args)
        case _:
            print("either unimplemented or invalid.")

    # trainer
    trainer = L.Trainer(accelerator='gpu',
                        max_epochs=args.max_epochs,
                        default_root_dir=args.output_dir,
                        enable_progress_bar=True,
                        logger=logger,
                        log_every_n_steps=10,
                        # check_val_every_n_epoch=10,
                        callbacks=[checkpoint_callback, earlystopping_callback])

    # begin training
    print("beginning the training.")
    start = time.time()
    if args.resume:
        trainer.fit(model=lit_task, ckpt_path=args.resume_ckpt)
    else:
        trainer.fit(model=lit_task)
    end = time.time()
    print(f"training completed. Elapsed time {end - start} seconds.")
    print("beginning the testing.")
    trainer.test(model=lit_task)

