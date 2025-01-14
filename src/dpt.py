import torch
import torch.nn as nn


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn is True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn is True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn is True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand is True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x


class DPT(nn.Module):
    def __init__(self,
                 backbone,
                 num_classes=1,
                 input_size=224,
                 freeze_backbone=False,
                 extract_layers=(3, 6, 9)):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        #
        self.bb_embed_dim = backbone.embed_dim
        self.extract_layers = extract_layers
        self.input_size = input_size
        use_bn = True
        # reassemble blocks
        self.output_size = input_size
        features = [96, 192, 384, 768]
        self.reassemble = self._make_reassemble(self.input_size, features, self.bb_embed_dim)
        self.scratch = self._make_scratch(features, self.output_size)
        # fusion blocks
        self.fusion_1 = self._make_fusion_block(self.output_size, use_bn)
        self.fusion_2 = self._make_fusion_block(self.output_size, use_bn)
        self.fusion_3 = self._make_fusion_block(self.output_size, use_bn)
        self.fusion_4 = self._make_fusion_block(self.output_size, use_bn)
        self.head = self._make_head(self.output_size, num_classes)

    @staticmethod
    def _make_reassemble(img_size, features, vit_features, start_index=1):
        size = [img_size, img_size]
        readout_oper = [ProjectReadout(vit_features, start_index) for out_feat in features]
        reassemble = nn.Module()
        reassemble.postprocess1 = nn.Sequential(readout_oper[0],
                                                Transpose(1, 2),
                                                nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
                                                nn.Conv2d(in_channels=vit_features, out_channels=features[0],
                                                          kernel_size=1, stride=1, padding=0),
                                                nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0],
                                                                   kernel_size=4, stride=4, padding=0, bias=True,
                                                                   dilation=1, groups=1))
        reassemble.postprocess2 = nn.Sequential(readout_oper[1],
                                                Transpose(1, 2),
                                                nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
                                                nn.Conv2d(in_channels=vit_features, out_channels=features[1],
                                                          kernel_size=1, stride=1, padding=0),
                                                nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1],
                                                                   kernel_size=2, stride=2, padding=0, bias=True,
                                                                   dilation=1, groups=1))
        reassemble.postprocess3 = nn.Sequential(readout_oper[2],
                                                Transpose(1, 2),
                                                nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
                                                nn.Conv2d(in_channels=vit_features, out_channels=features[2],
                                                          kernel_size=1, stride=1, padding=0))
        reassemble.postprocess4 = nn.Sequential(readout_oper[3],
                                                Transpose(1, 2),
                                                nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
                                                nn.Conv2d(in_channels=vit_features, out_channels=features[3],
                                                          kernel_size=1, stride=1, padding=0),
                                                nn.Conv2d(in_channels=features[3], out_channels=features[3],
                                                          kernel_size=3, stride=2, padding=1),)
        return reassemble

    @staticmethod
    def _make_fusion_block(features, use_bn):
        return FeatureFusionBlock_custom(
            features,
            nn.ReLU(False),
            deconv=False,
            bn=use_bn,
            expand=False,
            align_corners=True,
        )

    @staticmethod
    def _make_scratch(in_shape, out_shape,):
        scratch = nn.Module()
        scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape,
                                      kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape,
                                      kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape,
                                      kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape,
                                      kernel_size=3, stride=1, padding=1, bias=False, groups=1)
        return scratch

    @staticmethod
    def _make_head(features, num_classes):
        return nn.Sequential(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                             nn.BatchNorm2d(features),
                             nn.ReLU(True),
                             nn.Dropout(0.1, False),
                             nn.Conv2d(features, num_classes, kernel_size=1),
                             Interpolate(scale_factor=2, mode="bilinear", align_corners=True),)

    def forward(self, inp):
        vit_out, vit_activations = self.backbone.forward_with_layer_extraction(inp,
                                                                               extract_layers=list(self.extract_layers),
                                                                               permute=False)

        act1 = self.reassemble.postprocess1(vit_activations[0])
        act2 = self.reassemble.postprocess2(vit_activations[1])
        act3 = self.reassemble.postprocess3(vit_activations[2])
        act4 = self.reassemble.postprocess4(vit_out)

        layer_1_rn = self.scratch.layer1_rn(act1)
        layer_2_rn = self.scratch.layer2_rn(act2)
        layer_3_rn = self.scratch.layer3_rn(act3)
        layer_4_rn = self.scratch.layer4_rn(act4)
        #
        path_4 = self.fusion_4(layer_4_rn)
        path_3 = self.fusion_3(path_4, layer_3_rn)
        path_2 = self.fusion_2(path_3, layer_2_rn)
        path_1 = self.fusion_1(path_2, layer_1_rn)

        out = self.head(path_1)
        return out
