import torch.nn as nn
import math


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


class ClipSegStyleDecoder(nn.Module):
    """ClipSeg style decoder for segmentation with plain ViTs."""
    def __init__(self, backbone, patch_size, reduce_dim, n_heads, complex_trans_conv=False, freeze_backbone=True, num_classes=1, extract_layers=(3, 6, 9)):
        super().__init__()
        self.backbone = backbone
        self.extract_layers = extract_layers
        # decoder modules
        depth = len(self.extract_layers)
        trans_conv_ks = (patch_size, patch_size)
        self.reduces = nn.ModuleList([nn.Linear(self.backbone.embed_dim, reduce_dim) for _ in range(depth)])
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(len(extract_layers))])
        if complex_trans_conv:
            tp_kernels = (trans_conv_ks[0] // 4, trans_conv_ks[0] // 4)
            self.trans_conv = nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(reduce_dim, reduce_dim // 2, kernel_size=tp_kernels[0], stride=tp_kernels[0]),
                    nn.ReLU(),
                    nn.ConvTranspose2d(reduce_dim // 2, 1, kernel_size=tp_kernels[1], stride=tp_kernels[1]),
                    )
        else:
            self.trans_conv = nn.ConvTranspose2d(reduce_dim, num_classes, trans_conv_ks, stride=trans_conv_ks)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, input_image):
        x_inp = input_image
        bs = input_image.shape[0]
        input_image_size = input_image[2:]
        vit_out, activations = self.backbone.forward_with_layer_extraction(x_inp,
                                                                           extract_layers=[0]+list(self.extract_layers))
        activation0 = activations[0]  # there just in case i need to extract features in the future
        activations = activations[1:]

        a = None
        for i, (activation, block, reduce) in enumerate(zip(activations[::-1], self.blocks, self.reduces)):
            if a is not None:
                a = reduce(activation) + a
            else:
                a = reduce(activation)

            a = block(a)
        a = a[1:].permute(1, 2, 0)  # remove class token and permute -> BS, feats, tokens
        size = int(math.sqrt(a.shape[2]))
        a = a.view(bs, a.shape[1], size, size)
        a = self.trans_conv(a)
        return a







