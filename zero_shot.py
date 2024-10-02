import argparse

import pandas as pd
from tifffile import imwrite
from src.nnblocks import ClipSegStyleDecoder
import src.vits as vits
import torch
import os
from src.utils import write_dict_to_yaml, write_dict_to_csv
from src.checkpoints import model_remove_prefix
from src.data_and_transforms import SegDataset
from src.predict_on_array import predict_on_array_cf
from torchmetrics import JaccardIndex
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('Segmentation Parser', add_help=True)
    parser.add_argument('--arch', default='vit_small', type=str,
                        # choices=['vit_tiny', 'vit_small', 'vit_base']
                        help="""Name of architecture to train.""")
    parser.add_argument('--checkpoint_path', default='/path/to/checkpoint/', type=str,
                        help='Please specify path to the saved checkpoint.')
    parser.add_argument('--patch_size', default=16, type=int,
                        help="""Size in pixels of input square patches - default 16 (for 16x16 patches). 
                            Using smaller values leads to better performance but requires more memory""")
    parser.add_argument('--data_path', default='/path/to/data/', type=str,
                        help="""Please specify path to the training data.""")
    parser.add_argument('--output_dir', default='/path/to/output/', type=str,
                        help="""Please specify path to the training data.""")
    parser.add_argument('--input_size', default=224, type=int,
                        help="""size of the input to the network. should be divisible by 16. """)
    parser.add_argument('--reduce_dim', default=112, type=int,
                        help="""decoder reduces the dim of the input to this size.""")
    parser.add_argument('--decoder_head_count', default=4, type=int,
                        help="""number of decoder heads.""")
    parser.add_argument('--in_chans', default=4, type=int,
                        help="""number of image channels""")
    parser.add_argument('--img_size', default=1024, type=int,
                        help="""size of the input image""")
    return parser


def do_zero_shot(dataset, model, args):
    '''make sure to call model.eval() before passing the model.'''
    jacc = JaccardIndex(task="binary")
    tile_ls = []
    iou_ls = []
    # stats = {'tile': [], 'iou': []}
    pred_ls = []
    for i in range(len(dataset)):
        img, msk = dataset[i]
        output = predict_on_array_cf(model=model,
                                     arr=img,
                                     in_shape=(4, 224, 224),
                                     out_bands=1,
                                     stride=26,  # 52/26
                                     batchsize=1,
                                     augmentation=True)
        pred = output["prediction"]
        pred_ls.append(pred)
        iou = jacc(torch.Tensor(pred), msk.unsqueeze(0))
        tile_ls.append(i+1)
        iou_ls.append(iou.item())
    #
    best_pred_idx = np.argmax(iou_ls)
    worst_pred_idx = np.argmin(iou_ls)
    best = pred_ls[best_pred_idx]
    worst = pred_ls[worst_pred_idx]
    stats = {'tile': tile_ls,
             'iou': iou_ls}
    # stats['average'] = np.mean(iou_ls)
    return stats, best, worst


def main(args):
    # load the checkpoint
    backbone = vits.__dict__[args.arch](patch_size=args.patch_size,
                                        drop_path_rate=0.1,
                                        in_chans=args.in_chans)
    model = ClipSegStyleDecoder(backbone,
                                patch_size=args.patch_size,
                                reduce_dim=args.reduce_dim,
                                n_heads=args.decoder_head_count)
    ckpt_state_dict = torch.load(args.checkpoint_path)['state_dict']
    ckpt_state_dict = model_remove_prefix(ckpt_state_dict, "model.")
    model.load_state_dict(ckpt_state_dict)
    model.eval()
    model.to('cuda')
    # prepare the data
    data = SegDataset(data_path=args.data_path,
                      crop_size=args.img_size,
                      train=False)
    stats, best_pred, worst_pred = do_zero_shot(data, model, args)
    print(f"Mean IoU for this training run: {np.mean(stats['iou'])}.")
    yml_file = os.path.join(args.output_dir, "stats.yml")
    write_dict_to_yaml(yml_file, stats)
    # write_dict_to_csv(csv_file, stats)
    imwrite(os.path.join(args.output_dir, "best.tif"), best_pred)
    imwrite(os.path.join(args.output_dir, "worst.tif"), worst_pred)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args_yml_fp = os.path.join(args.output_dir, "args.yaml")
    write_dict_to_yaml(args_yml_fp, args.__dict__)
    main(args)









