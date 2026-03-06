"""
PatchGAN 判别器：输入 (条件图, 生成图/真实图)，输出 patch 真/假。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN：输入 6 通道 (input + target 拼接)，输出 NxN 的判别图。"""

    def __init__(self, in_channels=6, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 8, 1, 4, stride=2, padding=1),
        )

    def forward(self, x, y):
        """
        x: 条件图 (带笔迹), [B, 3, H, W]
        y: 待判别图 (生成或真实), [B, 3, H, W]
        """
        if y.shape[2:] != x.shape[2:]:
            y = F.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.net(torch.cat([x, y], dim=1))
