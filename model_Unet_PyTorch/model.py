# model.py
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """
    Simple 3D U-Net for binary segmentation.
    Input:  (N, 1, D, H, W)
    Output: (N, 1, D, H, W)  (logits; apply sigmoid for probs)
    """
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc   = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv(128, 256))

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output
        self.outc = nn.Conv3d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)          # (N,32,D,H,W)
        x2 = self.down1(x1)       # (N,64,D/2,H/2,W/2)
        x3 = self.down2(x2)       # (N,128, ...)
        x4 = self.down3(x3)       # (N,256, ...)

        # Bottleneck
        x5 = self.bottleneck(x4)  # (N,512,...)

        # Decoder
        x = self.up3(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec1(x)

        logits = self.outc(x)
        return logits