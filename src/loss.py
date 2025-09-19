import segmentation_models_pytorch as smp
import torch.nn as nn


class WeightedSumLoss(nn.Module):
    def __init__(self, items):
        """
        Weighted sum of multiple loss modules.
        """
        super().__init__()
        self.items = nn.ModuleList([lm for _, lm, _ in items])
        self.names = [n for n, _, _ in items]
        self.weights = [w for _, _, w in items]

    def forward(self, y_pred, y_true):
        parts = {}
        total = 0.0
        for name, loss_mod, w in zip(self.names, self.items, self.weights):
            if w == 0:
                continue
            val = loss_mod(y_pred, y_true)
            parts[name] = val
            total = total + w * val
        return total, parts

def get_segmentation_loss(bce, dice, tversky, lovasz, focal, ignore_index):
    segmentation_loss = WeightedSumLoss([
        ("bce",   smp.losses.SoftBCEWithLogitsLoss(ignore_index=ignore_index), bce),
        ("dice",  smp.losses.DiceLoss(mode="binary", ignore_index=ignore_index), dice),
        ("tversky", smp.losses.TverskyLoss(mode="binary", ignore_index=ignore_index), tversky),
        ("lovasz",  smp.losses.LovaszLoss(mode="binary", ignore_index=ignore_index), lovasz),
        ("focal",   smp.losses.FocalLoss(mode="binary", ignore_index=ignore_index), focal),
    ])
    return segmentation_loss

