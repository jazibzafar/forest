import torch
from collections import OrderedDict
import src.vits as vits


def load_dino_checkpoint(checkpoint_path, checkpoint_key):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # checkpoint = checkpoint['state_dict']
    pretrained_model = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith(checkpoint_key):
            pretrained_model[k] = v

    pretrained_model = {k.replace(f"{checkpoint_key}.", ""): v for k, v in pretrained_model.items()}
    pretrained_model = {k.replace("backbone.", ""): v for k, v in pretrained_model.items()}
    return pretrained_model


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


def prepare_arch(arch, pretrained_model, patch_size):
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
