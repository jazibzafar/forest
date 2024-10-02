import sys
import argparse
import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import time
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
import src.vits as vits
from src.vits import DINOHead
from src.utils import bool_flag
from src.nnblocks import MultiCropWrapper
from src.data_and_transforms import DINOTransform, GeoWebDataset
from torchvision.models import resnet50

# TODO: add support for torch compile
# DONE: add support for only training steps eliminate epochs altogether
# TODO: DDP support
# DONE: add resume support

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
                        # choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                        #         + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
                        help="""Name of architecture to train. For quick experiments with ViTs,
                        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--in_chans', default=4, type=int, help="""Number of color channels in the input image.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_steps', default=0, type=int,
                        help='Number of warmup steps for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay_init', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--max_steps', default=25000, type=int, help='Number of (gradient) steps of training.')
    parser.add_argument('--freeze_last_layer', default=1000, type=int, help="""Number of training steps
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.
        Also, for this code this has been modified to 1000 steps rather than 1 epoch""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_steps", default=5000, type=int,
                        help="Number of steps for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crop_size', type=int, default=224,
                        help="""Size of the global crop. In paper, its 224.""")
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crop_size', type=int, default=96,
                        help="""Size of the global crop. In paper, its 96.""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    # parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--resume', default='', type=str,
                        help='path to checkpoint to restore training. ')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=1, type=int, help="world size.")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def cosine_scheduler_step_based(base_value, final_value, warmup_iters, total_iters, start_warmup_value=0):
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(total_iters - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_iters
    return schedule


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_steps, num_steps, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_steps),
            np.ones(num_steps - warmup_teacher_temp_steps) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, step):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[step]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # TODO: When doing distributed don't forget to fix this thing.
        # dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        batch_center = batch_center / len(teacher_output)  # without DDP

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class LitDINO(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.automatic_optimization = False

        # ======= create student & teacher networks =======#
        if args.arch in vits.__dict__.keys():
            model = vits.__dict__[args.arch]
            student_backbone = model(patch_size=args.patch_size,
                                     drop_path_rate=args.drop_path_rate,
                                     in_chans=args.in_chans)
            teacher_backbone = model(patch_size=args.patch_size, in_chans=args.in_chans)
            student_in_dim_dinohead = student_backbone.embed_dim
            teacher_in_dim_dinohead = teacher_backbone.embed_dim
        elif args.arch == "resnet50":
            model = resnet50(progress=False)
            model.conv1 = nn.Conv2d(args.in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
            student_backbone = model
            teacher_backbone = model
            student_in_dim_dinohead = model.fc.in_features
            teacher_in_dim_dinohead = model.fc.in_features
        else:
            raise Exception(f"Unknown architecture: {args.arch}")

        student_head = DINOHead(in_dim=student_in_dim_dinohead,
                                out_dim=args.out_dim,
                                use_bn=args.use_bn_in_head,
                                norm_last_layer=args.norm_last_layer)

        teacher_head = DINOHead(in_dim=teacher_in_dim_dinohead,
                                out_dim=args.out_dim)

        self.student = MultiCropWrapper(student_backbone, student_head)
        self.teacher = MultiCropWrapper(teacher_backbone, teacher_head)

        # disable gradients for teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # ======= initialize the loss =======#
        self.loss = DINOLoss(args.out_dim,
                             args.local_crops_number + 2,
                             args.warmup_teacher_temp,
                             args.teacher_temp,
                             args.warmup_teacher_temp_steps,
                             args.max_steps)

        # ======= create the dataset =======#
        # alternatively, this could be in setup method
        dino_transform = DINOTransform(args.global_crop_size,
                                       args.global_crops_scale,
                                       args.local_crop_size,
                                       args.local_crops_scale,
                                       args.local_crops_number)
        self.dataset = GeoWebDataset(root=args.data_path,
                                     n_bands=args.in_chans,
                                     augmentations=dino_transform, )

    def configure_optimizers(self):
        regularized, not_regularized = [], []
        for n, p in self.student.named_parameters():
            if not p.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if n.endswith(".bias") or len(p.shape) == 1:
                not_regularized.append(p)
            else:
                regularized.append(p)
        param_groups = [{'params': regularized},
                        {'params': not_regularized, 'weight_decay': 0.}]  # weight decay is 0 because of the scheduler

        self.lr = args.lr * (args.batch_size_per_gpu * args.world_size / 256)
        if args.optimizer == "adamw":
            opt = torch.optim.AdamW(param_groups, self.lr)
        elif args.optimizer == "sgd":
            opt = torch.optim.SGD(param_groups, self.lr)
        else:
            raise Exception(f"Unknown optimizer: {args.optimizer}")
        return opt

    def train_dataloader(self):
        loader = DataLoader(self.dataset,
                                 num_workers=args.num_workers,
                                 batch_size=args.batch_size_per_gpu,
                                 pin_memory=True,
                                 drop_last=True, )

        self.lr_sch = cosine_scheduler_step_based(self.lr, 1e-6, args.warmup_steps, args.max_steps)
        self.wd_sch = cosine_scheduler_step_based(args.weight_decay_init, args.weight_decay_end, 0, args.max_steps)
        self.mm_sch = cosine_scheduler_step_based(args.momentum_teacher, 1.0, 0, args.max_steps)
        return loader

    def training_step(self, batch):
        opt = self.optimizers()

        # update learning rate, weight decay
        for i, param_group in enumerate(opt.param_groups):
            param_group['lr'] = self.lr_sch[self.global_step]
            if i == 0:  # only the first group is regularized
                param_group['weight_decay'] = self.wd_sch[self.global_step]

        # batch is a list: [g1, g2, l1, l2, ..., lm]
        # where each tensor has shape [batch, height, width, channel]
        teacher_output = self.teacher(batch[:2])
        student_output = self.student(batch)
        loss = self.loss(student_output, teacher_output, self.global_step)

        opt.zero_grad()
        self.manual_backward(loss)
        # clip gradient
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), args.clip_grad)
        # cancel gradient for the freeze_last_layer steps
        if self.global_step < args.freeze_last_layer:
            for n, p in self.student.named_parameters():
                if "last_layer" in n:
                    p.grad = None
        opt.step()

        # EMA update for the teacher
        m = self.mm_sch[self.global_step-1 if self.global_step != 0 else self.global_step]
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_((1 - m) * ps.data)

        self.log('rates/lr', opt.param_groups[0]['lr'])
        self.log('rates/weight_decay', opt.param_groups[0]['weight_decay'])
        self.log('rates/momentum', m)
        self.log('train/loss', loss, True)


def main(args):
    dino = LitDINO(args)

    checkpoint_callback = ModelCheckpoint(dirpath=args.output_dir,
                                          every_n_train_steps=int(args.max_steps/5),
                                          save_last=True)

    logger = TensorBoardLogger(save_dir=args.output_dir,
                               name="",
                               default_hp_metric=False)

    trainer = L.Trainer(accelerator='gpu',
                        max_steps=args.max_steps,
                        default_root_dir=args.output_dir,
                        enable_progress_bar=True,
                        logger=logger,
                        # precision="bf16-mixed",
                        callbacks=[checkpoint_callback])

    print("beginning the training.")
    start = time.time()
    if args.resume:
        trainer.fit(model=dino, ckpt_path=args.resume)
    else:
        trainer.fit(model=dino)
    # trainer.fit(model=dino)
    end = time.time()
    print(f"training completed. Elapsed time {end - start} seconds.")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # The below is already done in the bash script
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
