import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
      return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetResNet50(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):
        super().__init__()
        if pretrained:
            bb = resnet50(weights=ResNet50_Weights.DEFAULT)
        bb = resnet50()

        self.enc0 = nn.Sequential(bb.conv1, bb.bn1, bb.relu)  # [B, 64,  128, 128]
        self.pool = bb.maxpool
        self.enc1 = bb.layer1   # [B, 256,  64,  64]
        self.enc2 = bb.layer2   # [B, 512,  32,  32]
        self.enc3 = bb.layer3   # [B, 1024, 16,  16]
        self.enc4 = bb.layer4   # [B, 2048,  8,   8]

        self.bottleneck = nn.Sequential(
            ConvBnRelu(2048, 1024),
            ConvBnRelu(1024, 1024),
        )

        self.dec4 = DecoderBlock(1024, 1024, 512)
        self.dec3 = DecoderBlock(512,  512,  256)
        self.dec2 = DecoderBlock(256,  256,  128)
        self.dec1 = DecoderBlock(128,  64,   64)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            ConvBnRelu(32, 32),
        )

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b  = self.bottleneck(e4)

        d4 = self.dec4(b,  e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        d0 = self.dec0(d1)

        return self.head(d0)  # [B, num_classes, 256, 256]