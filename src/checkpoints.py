import copy

import torch
from collections import OrderedDict
import src.vits as vits
from torchvision.models import resnet50


def load_dino_checkpoint(checkpoint_path, checkpoint_key):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    try:
        pretrained_model = checkpoint[checkpoint_key]
    except KeyError:
        pretrained_model = checkpoint
    return pretrained_model


def extract_backbone(checkpoint, checkpoint_key):
    backbone = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith(checkpoint_key):
            backbone[k] = v

    backbone = {k.replace(f"{checkpoint_key}.", ""): v for k, v in backbone.items()}
    backbone = {k.replace("backbone.", ""): v for k, v in backbone.items()}
    return backbone


def convert_checkpoint_to_backbone(checkpoint):
    checkpoint = checkpoint['state_dict']
    backbone_teacher = extract_backbone(checkpoint, 'teacher')
    backbone_student = extract_backbone(checkpoint, 'student')
    return {'student': backbone_student, 'teacher': backbone_teacher}


def load_finetuned_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = checkpoint['state_dict']
    # pretrained_model = OrderedDict()
    # for k, v in checkpoint.items():
    #     if k.startswith(checkpoint_key):
    #         pretrained_model[k] = v

    # pretrained_model = {k.replace(f"{checkpoint_key}.", ""): v for k, v in pretrained_model.items()}
    finetuned_model = model_remove_prefix(checkpoint, "model.0.")
    return finetuned_model


def prepare_vit(arch, pretrained_model, patch_size, num_chans=4):
    if pretrained_model['patch_embed.proj.weight'].size(1) != num_chans:
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0, in_chans=3)
        msg = model.load_state_dict(pretrained_model, strict=False)
        print(msg)
        # Adapt is to four channels.
        weight = model.patch_embed.proj.weight.clone()
        model.patch_embed.proj = torch.nn.Conv2d(num_chans, model.embed_dim,
                                                 kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        with torch.no_grad():
            model.patch_embed.proj.weight[:, :num_chans-1] = weight
            # this line below assigns red weights to nir channel.
            model.patch_embed.proj.weight[:, num_chans-1] = model.patch_embed.proj.weight[:, 0]
    else:
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        msg = model.load_state_dict(pretrained_model, strict=False)
        print(msg)
    return model


def prepare_vit2(arch, pretrained_model, patch_size, num_chans=4):
    if pretrained_model['patch_embed.proj.weight'].size(1) != num_chans:
        # reduce num chans from 4 (backbone) to 3 (data)
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0, in_chans=4)
        msg = model.load_state_dict(pretrained_model, strict=False)
        print(msg)
        weight = model.patch_embed.proj.weight.clone()
        model.patch_embed.proj = torch.nn.Conv2d(num_chans, model.embed_dim,
                                                 kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        with torch.no_grad():
            model.patch_embed.proj.weight[:, :2] = weight[:, :2]
    else:
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0, in_chans=num_chans)
        msg = model.load_state_dict(pretrained_model, strict=False)
        print(msg)
    return model


def prepare_resnet_model(pretrained_model, num_data_chans):
    pt_in_chans = pretrained_model['conv1.weight'].shape[1]
    model = resnet50()
    # step 1: replace the conv1 to have the same number of channels as the pt model, so we can load the state dict.
    model.conv1 = torch.nn.Conv2d(pt_in_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    msg = model.load_state_dict(pretrained_model, strict=False)
    print(msg)
    # step 2: edit the conv1 weights, shape to adapt to the data channels.
    if not pt_in_chans == num_data_chans:
        if pt_in_chans > num_data_chans:  # 43
            weight = model.conv1.weight.clone()
            model.conv1 = torch.nn.Conv2d(num_data_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            with torch.no_grad():
                model.conv1.weight[:, :2] = weight[:, :2]
        else:  # pt_in_chans < num_data_chans # 34
            weight = model.conv1.weight.clone()
            model.conv1 = torch.nn.Conv2d(num_data_chans, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            with torch.no_grad():
                model.conv1.weight[:, :num_data_chans - 1] = weight
                model.conv1.weight[:, num_data_chans - 1] = model.conv1.weight[:, 0]
    return model


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def replace_prefix(text, prefix_add, prefix_rem):
    if text.startswith(prefix_rem):
        return prefix_add + text[len(prefix_rem):]
    return text


def model_remove_prefix(in_state_dict, prefix_to_remove):
    pairings = [
        (src_key, remove_prefix(src_key, prefix_to_remove))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return
    out_state_dict = {}
    for src_key, dest_key in pairings:
        # print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]
    return OrderedDict(out_state_dict)


def model_replace_prefix(in_state_dict, prefix_add, prefix_rem):
    pairings = [
        (src_key, replace_prefix(src_key, prefix_add, prefix_rem))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return
    out_state_dict = {}
    for src_key, dest_key in pairings:
        # print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]
    return OrderedDict(out_state_dict)
