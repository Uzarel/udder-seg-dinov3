import torch.nn as nn
import torch.nn.functional as F

from .adapters import InputAdapter1to3, ImageNetNormalizer
from .fpn_decoder import FPNDecoder

def get_enc_ch_from_convnext(convnext_backbone):
    # Read ConvNeXt stage widths from downsample layers
    ch = []
    # Layer 0 conv is at index 0; layers 1,2,3 convs are at index 1
    ch.append(convnext_backbone.downsample_layers[0][0].out_channels)
    for i in (1, 2, 3):
        ch.append(convnext_backbone.downsample_layers[i][1].out_channels)
    return ch

class DinoConvNeXtFeatures(nn.Module):
    """Run the hub ConvNeXt in a staged way and return C2..C5 features."""
    def __init__(self, backbone, use_input_adapter=True):
        super().__init__()
        self.backbone = backbone
        self.input_adapter = InputAdapter1to3() if use_input_adapter else None
        self.imagenet_norm = ImageNetNormalizer()

    def forward(self, x):
        if x.shape[1] == 1:
            x = self.input_adapter(x)
        x = self.imagenet_norm(x)

        # Grab feature maps per stage
        x = self.backbone.downsample_layers[0](x)   # stride 4
        x = self.backbone.stages[0](x)
        c2 = x                                      # 1/4
        x = self.backbone.downsample_layers[1](x)   # stride 8
        x = self.backbone.stages[1](x)
        c3 = x                                      # 1/8
        x = self.backbone.downsample_layers[2](x)   # stride 16
        x = self.backbone.stages[2](x)
        c4 = x                                      # 1/16
        x = self.backbone.downsample_layers[3](x)   # stride 32
        x = self.backbone.stages[3](x)
        c5 = x                                      # 1/32
        return [c2, c3, c4, c5]

class DinoConvNeXtFPN(nn.Module):
    def __init__(
        self,
        convnext_backbone,          # torch.hub ConvNeXt
        use_input_adapter=True,
        pyramid_channels=256,
        segmentation_channels=128,
        merge_policy="cat",         # "add" or "cat"
        dropout=0.1
    ):
        super().__init__()
        self.encoder = DinoConvNeXtFeatures(convnext_backbone, use_input_adapter)
        enc_ch = get_enc_ch_from_convnext(convnext_backbone)

        self.decoder = FPNDecoder(
            encoder_channels=enc_ch,
            pyramid_channels=pyramid_channels,
            segmentation_channels=segmentation_channels,
            dropout=dropout,
            merge_policy=merge_policy,
            interpolation_mode="nearest",
        )
        self.seg_head = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)

    def forward(self, x):
        feats = self.encoder(x)                                                         # [c2, c3, c4, c5]
        x = self.decoder(feats)                                                         # FPN features at 1/4 resolution
        x = self.seg_head(x)                                                            # Masks as (B, 1, H/4, W/4) tensor
        x = F.interpolate(x, scale_factor=4.0, mode="bilinear", align_corners=False)    # Rescaling to input size
        return x
