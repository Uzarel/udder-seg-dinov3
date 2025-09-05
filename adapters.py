import torch
import torch.nn as nn

# 1 to 3 channels adapter keeps pretrained weights intact
class InputAdapter1to3(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(1, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.proj.weight[:] = 1/3
    def forward(self, x1):
        return self.proj(x1)
    
# DINOv3 weights require ImageNet normalization
class ImageNetNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std",  std,  persistent=False)
    def forward(self, x3):
        return (x3 - self.mean) / self.std
