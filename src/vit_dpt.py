import torch
import torch.nn as nn
from typing import Sequence, List, Optional

from .adapters import InputAdapter1to3, ImageNetNormalizer
from .dpt_decoder import DPTDecoder, DPTSegmentationHead

class DINOv3ViTEncoderNoCLS(nn.Module):
    """
    Uses DINOv3 ViT's get_intermediate_layers(..., reshape=True) to produce
    4 feature maps [B,C,gh,gw]; no prefix tokens are returned.
    """
    def __init__(self, dino_vit, out_indices: Sequence[int] = (-1, -3, -6, -9), use_norm: bool = True, use_input_adapter: bool = True):
        super().__init__()
        self.input_adapter = InputAdapter1to3() if use_input_adapter else None
        self.imagenet_norm = ImageNetNormalizer()
        self.backbone = dino_vit
        self.use_norm = use_norm

        # Infer patch size & embed dim
        proj = self.backbone.patch_embed.proj
        self.patch_size = int(getattr(proj, "stride", getattr(proj, "kernel_size"))[0])
        self.embed_dim  = int(proj.out_channels)

        # Resolve indices: convert negatives to absolute 0..L-1 and validate
        self.total_blocks = len(self.backbone.blocks)  # e.g. 12, 24, 32, 40
        self.block_indices = self._resolve_indices(out_indices)

    def _resolve_indices(self, indices: Sequence[int]) -> List[int]:
        resolved = []
        for i in indices:
            j = i if i >= 0 else self.total_blocks + i  # -1 -> L-1, etc.
            if not (0 <= j < self.total_blocks):
                raise ValueError(
                    f"Layer index {i} (→ {j}) out of range [0, {self.total_blocks-1}]"
                )
            resolved.append(int(j))
        # Keep them in ascending depth (shallower → deeper)
        resolved = sorted(dict.fromkeys(resolved))  # dedupe, keep order
        return resolved

    def forward(self, x: torch.Tensor):
        if x.shape[1] == 1:
            x = self.input_adapter(x)
        x = self.imagenet_norm(x)
        # Get chosen blocks as feature maps [B,C,gh,gw]
        feats: List[torch.Tensor] = self.backbone.get_intermediate_layers(
            x,
            n=self.block_indices,       # <-- now 0-based non-negative
            reshape=True,
            norm=self.use_norm,
        )
        # Optional sanity check to catch mismatches early
        if len(feats) != len(self.block_indices):
            raise RuntimeError(
                f"get_intermediate_layers returned {len(feats)} maps, "
                f"expected {len(self.block_indices)} for indices {self.block_indices}."
            )
        prefixes: List[Optional[torch.Tensor]] = [None] * len(feats)  # no CLS in this ViT
        return feats, prefixes

class DinoViTDPT(nn.Module):
    """
    DINOv3 ViT encoder (no CLS) → DPT decoder → segmentation head.
    Output logits at full input resolution.
    """
    def __init__(
        self,
        dino_vit,
        use_input_adapter: bool = True,
        out_indices: Sequence[int] = (-1, -3, -6, -9),
        num_classes: int = 1,
        readout: str = "ignore",   # no prefix tokens available
        fusion_channels: int = 256,
    ):
        super().__init__()
        self.encoder = DINOv3ViTEncoderNoCLS(dino_vit, out_indices=out_indices, use_input_adapter=use_input_adapter)

        enc_out_ch   = (self.encoder.embed_dim,) * 4
        enc_strides  = (self.encoder.patch_size,) * 4

        self.decoder = DPTDecoder(
            encoder_out_channels=enc_out_ch,
            encoder_output_strides=enc_strides,
            encoder_has_prefix_tokens=False,  # <-- key
            readout=readout,
            fusion_channels=fusion_channels,
        )
        self.seg_head = DPTSegmentationHead(
            in_channels=fusion_channels,
            out_channels=num_classes,
            activation=None,
            upsampling=2.0,  # decoder gives H/2 → upsample to H
        )

    @property
    def patch_size(self): return self.encoder.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats, prefixes = self.encoder(x)
        fused = self.decoder(feats, prefixes)  # [B, C, H/2, W/2]
        return self.seg_head(fused)                # [B, K, H, W]
