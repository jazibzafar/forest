import torch
from collections import OrderedDict
import src.vits as vits


def load_dino_checkpoint(checkpoint_path, checkpoint_key, num_chans=4):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    pretrained_model = checkpoint[checkpoint_key]
    pretrained_model = model_remove_prefix(pretrained_model, 'backbone.')
    # checkpoint = checkpoint['state_dict']
    # pretrained_model = OrderedDict()
    # for k, v in checkpoint.items():
    #     if k.startswith(checkpoint_key):
    #         pretrained_model[k] = v
    #
    # pretrained_model = {k.replace(f"{checkpoint_key}.", ""): v for k, v in pretrained_model.items()}
    # pretrained_model = {k.replace("backbone.", ""): v for k, v in pretrained_model.items()}
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


def prepare_arch(arch, pretrained_model, patch_size, num_chans=4):
    if pretrained_model['patch_embed.proj.weight'].size(1) != num_chans:
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0, in_chans=3)
        msg = model.load_state_dict(pretrained_model, strict=False)
        print(msg)
        # Adapt is to four channels.
        weight = model.patch_embed.proj.weight.clone()
        model.patch_embed.proj = torch.nn.Conv2d(4, 768, kernel_size=(16, 16), stride=(16, 16))
        with torch.no_grad():
            model.patch_embed.proj.weight[:, :3] = weight
            # this line below assigns red weights to nir channel.
            model.patch_embed.proj.weight[:, 3] = model.patch_embed.proj.weight[:, 0]
    else:
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        msg = model.load_state_dict(pretrained_model, strict=False)
        print(msg)
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
        print(f"{src_key}  ==>  {dest_key}")
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
        print(f"{src_key}  ==>  {dest_key}")
        out_state_dict[dest_key] = in_state_dict[src_key]
    return OrderedDict(out_state_dict)
