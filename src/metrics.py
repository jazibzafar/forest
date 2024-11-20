import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def multiclass_iou(predictions, targets, num_classes, non_zero_average=True):
    """
    Calculate Multiclass IoU between predictions and targets.

    Args:
        predictions (torch.Tensor): Predicted class labels (shape: [B, H, W])
        targets (torch.Tensor): Ground truth class labels (shape: [B, H, W])
        num_classes (int): Number of classes

    Returns:
        iou (torch.Tensor): IoU for each class (shape: [num_classes])
    """
    # Initialize intersection and union for each class
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)

    # Loop through each class
    for cls in range(num_classes):
        # Create boolean masks for intersection and union
        preds_cls = (predictions == cls)
        targets_cls = (targets == cls)

        intersection[cls] = (preds_cls & targets_cls).sum().float()
        union[cls] = preds_cls.sum() + targets_cls.sum() - intersection[cls]

    # Calculate IoU for each class
    iou = intersection / (union + 1e-6)  # Add a small constant to avoid division by zero
    # iou = iou.numpy()
    #
    # if non_zero_average:
    #     non_zeros = iou[np.nonzero(iou)]
    #     num = np.sum(non_zeros)
    #     den = len(non_zeros)
    #     iou = num/den

    return iou


def mIOU(label, pred, num_classes=19):
    # pred = F.softmax(pred, dim=1)
    # pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)
