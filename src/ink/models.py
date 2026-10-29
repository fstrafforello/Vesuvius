"""Ink-detection models. All take (B, D, H, W) surface-volume stacks and return (B, 1, H, W) logits.

  2d  - layers are plain input channels of a 2D U-Net (the baseline).
  3d  - a small 3D CNN reads the stack as a volume, then per-pixel depth attention decides which
        depths matter at each location (ink sits at varying depth in the sheet), collapsing to a
        2D feature map that feeds the same 2D U-Net.
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block3d(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1, 1)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False), nn.BatchNorm3d(cout), nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout),
        )
        self.skip = nn.Sequential(nn.Conv3d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm3d(cout))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class DepthToPlane(nn.Module):
    """3D feature extractor + depth attention: (B, D, H, W) -> (B, 2*width, H, W).

    Runs at half depth and half spatial resolution (full-res 3D activations don't fit in memory),
    then upsamples the collapsed 2D features back to H x W.
    """

    def __init__(self, width=24):
        super().__init__()
        self.stem = Block3d(1, width // 2, stride=(2, 2, 2))
        self.body = Block3d(width // 2, width)
        self.attn = nn.Conv3d(width, 1, 1)

    def forward(self, x):
        f = self.body(self.stem(x[:, None]))                 # (B, C, D/2, H/2, W/2)
        w = torch.softmax(self.attn(f), dim=2)                # where along depth to look, per pixel
        p = torch.cat([(f * w).sum(2), f.amax(2)], 1)
        return F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)


class Ink3d(nn.Module):
    """Depth-aware features plus the raw stack (for full-res detail) into a 2D U-Net."""

    def __init__(self, n_layers, encoder="resnet34", width=24):
        super().__init__()
        self.lift = DepthToPlane(width)
        self.unet = smp.Unet(encoder, encoder_weights="imagenet", in_channels=2 * width + n_layers, classes=1)

    def forward(self, x):
        return self.unet(torch.cat([self.lift(x), x], 1))


class DepthPool(nn.Module):
    """Collapse depth with learned per-pixel attention: (B, C, D, H, W) -> (B, C, H, W)."""

    def __init__(self, c):
        super().__init__()
        self.attn = nn.Conv3d(c, 1, 1)

    def forward(self, f):
        return (f * torch.softmax(self.attn(f), dim=2)).sum(2)


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], 1))


class InkFull3d(nn.Module):
    """Fully 3D encoder (video-pretrained 3D ResNet-18) with depth attention at every scale and a
    2D U-Net decoder. The depth axis stays 3D through the whole encoder.

    Spatial scales: stem/layer1 H/2, layer2 H/4, layer3 H/8, layer4 H/16; depth 32 -> 32,32,16,8,4.
    """

    def __init__(self, pretrained=True, checkpoint=True, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        from torchvision.models.video import R3D_18_Weights, r3d_18
        net = r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None)
        w = net.stem[0].weight.data.sum(1, keepdim=True)            # RGB -> 1 channel
        net.stem[0] = nn.Conv3d(1, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        net.stem[0].weight.data.copy_(w)
        self.stages = nn.ModuleList([nn.Sequential(net.stem, net.layer1), net.layer2, net.layer3, net.layer4])
        self.pools = nn.ModuleList([DepthPool(c) for c in (64, 128, 256, 512)])
        self.up3 = Up(512, 256, 256)
        self.up2 = Up(256, 128, 128)
        self.up1 = Up(128, 64, 64)
        self.head = nn.Conv2d(64, 1, 1)
        self.checkpoint = checkpoint

    def forward(self, x):
        h, w = x.shape[-2:]
        f, skips = x[:, None], []
        for i, (stage, pool) in enumerate(zip(self.stages, self.pools)):
            if self.checkpoint and self.training and i < 2:   # the two biggest stages: recompute, don't store
                f = torch.utils.checkpoint.checkpoint(stage, f, use_reentrant=False)
            else:
                f = stage(f)
            skips.append(pool(f))
        y = self.up3(skips[3], skips[2])
        y = self.up2(y, skips[1])
        y = self.up1(y, skips[0])
        return F.interpolate(self.head(self.drop(y)), size=(h, w), mode="bilinear", align_corners=False)


def build(kind, n_layers, encoder="resnet34", dropout=0.0):
    if kind == "2d":
        return smp.Unet(encoder, encoder_weights="imagenet", in_channels=n_layers, classes=1)
    if kind == "3d":
        return Ink3d(n_layers, encoder)
    if kind == "full3d":
        return InkFull3d(dropout=dropout)
    raise ValueError(kind)
