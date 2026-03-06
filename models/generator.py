"""
Pix2Pix U-Net 生成器：输入带笔迹图像，输出擦除笔迹后的图像。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch, out_ch, norm=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
    ]
    if norm:
        layers.insert(1, nn.BatchNorm2d(out_ch))
    return nn.Sequential(*layers)


def _upconv_block(in_ch, out_ch, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class UnetGenerator(nn.Module):
    """U-Net 结构生成器，带 skip connections。"""

    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.enc2 = _conv_block(ngf, ngf * 2)
        self.enc3 = _conv_block(ngf * 2, ngf * 4)
        self.enc4 = _conv_block(ngf * 4, ngf * 8)
        self.enc5 = _conv_block(ngf * 8, ngf * 8)
        self.enc6 = _conv_block(ngf * 8, ngf * 8)
        self.enc7 = _conv_block(ngf * 8, ngf * 8)

        # Bottleneck（输出 1x1，不用 BatchNorm，否则 batch=1 时会报错）
        self.bottleneck = _conv_block(ngf * 8, ngf * 8, norm=False)

        # Decoder (with skip from encoder)
        self.dec7 = _upconv_block(ngf * 8 * 2, ngf * 8, dropout=True)
        self.dec6 = _upconv_block(ngf * 8 * 2, ngf * 8, dropout=True)
        self.dec5 = _upconv_block(ngf * 8 * 2, ngf * 8, dropout=True)
        self.dec4 = _upconv_block(ngf * 8 * 2, ngf * 4)
        self.dec3 = _upconv_block(ngf * 4 * 2, ngf * 2)
        self.dec2 = _upconv_block(ngf * 2 * 2, ngf)
        self.dec1 = _upconv_block(ngf * 2, ngf)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def _crop_to_match(skip, target):
        """将 skip 的空间尺寸裁剪为与 target 一致，避免 cat 时尺寸不匹配。"""
        _, _, h, w = target.shape
        return skip[:, :, :h, :w]

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        b = self.bottleneck(e7)

        d7 = self.dec7(torch.cat([b, self._crop_to_match(e7, b)], dim=1))
        d6 = self.dec6(torch.cat([d7, self._crop_to_match(e6, d7)], dim=1))
        d5 = self.dec5(torch.cat([d6, self._crop_to_match(e5, d6)], dim=1))
        d4 = self.dec4(torch.cat([d5, self._crop_to_match(e4, d5)], dim=1))
        d3 = self.dec3(torch.cat([d4, self._crop_to_match(e3, d4)], dim=1))
        d2 = self.dec2(torch.cat([d3, self._crop_to_match(e2, d3)], dim=1))
        d1 = self.dec1(torch.cat([d2, self._crop_to_match(e1, d2)], dim=1))
        d0 = self.dec0(torch.cat([d1, self._crop_to_match(e1, d1)], dim=1))
        return d0
