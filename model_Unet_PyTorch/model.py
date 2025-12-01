# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Conv3D -> INorm -> ReLU -> Conv3D -> INorm -> ReLU
    (with padding=1 so spatial size is preserved)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, base_c=32):
        super().__init__()

        # Encoder
        self.inc   = DoubleConv(n_channels, base_c)           # 32
        self.down1 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(base_c, base_c * 2)                    # 64
        )
        self.down2 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(base_c * 2, base_c * 4)                # 128
        )
        self.down3 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(base_c * 4, base_c * 8)                # 256
        )

        # Bottleneck
        self.down4 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(base_c * 8, base_c * 16)               # 512
        )

        # Decoder
        self.up1 = nn.ConvTranspose3d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(base_c * 16, base_c * 8)

        self.up2 = nn.ConvTranspose3d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(base_c * 8, base_c * 4)

        self.up3 = nn.ConvTranspose3d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(base_c * 4, base_c * 2)

        self.up4 = nn.ConvTranspose3d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(base_c * 2, base_c)

        self.outc = nn.Conv3d(base_c, n_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)      # (N, 32, 64, 64, 64)
        x2 = self.down1(x1)   # (N, 64, 32, 32, 32)
        x3 = self.down2(x2)   # (N, 128,16, 16, 16)
        x4 = self.down3(x3)   # (N, 256, 8,  8,  8)
        x5 = self.down4(x4)   # (N, 512, 4,  4,  4)

        # decoder
        x = self.up1(x5)      # (N, 256, 8, 8, 8)
        x = torch.cat([x4, x], dim=1)  # (N, 512, 8, 8, 8)
        x = self.conv1(x)

        x = self.up2(x)       # (N, 128, 16,16,16)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)       # (N, 64, 32,32,32)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)       # (N, 32, 64,64,64)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        x = self.outc(x)
        # final sigmoid for binary segmentation
        return torch.sigmoid(x)