import torch
import torch.nn as nn
import torch.nn.functional as F

# --- U-Net Helper Blocks ---

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First 3D Convolution
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            # Second 3D Convolution
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling with ConvTranspose3d and concatenation then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Use ConvTranspose3d for upsampling
        # The input channels are reduced by half here, but the skip connection will be added back.
        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        # The subsequent DoubleConv block receives the combined channels
        self.conv = DoubleConv(in_channels, out_channels) 

    def forward(self, x1, x2):
        # x1 is the output from the previous decoder stage (to be upsampled)
        # x2 is the skip connection from the encoder
        
        # 1. Upsampling (Transpose Conv)
        x1 = self.up(x1)
        
        # Note: Padding/cropping steps are omitted here for simplicity, assuming 64^3 input is clean.
        # Concatenate the upsampled feature map and the skip connection
        x = torch.cat([x2, x1], dim=1) # Concatenate along the channel dimension (dim=1)
        
        # 2. Double Convolution
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1x1 convolution to map to the number of classes"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- The 3D U-Net Model ---

class UNet3D(nn.Module):
    """
    3D U-Net model based on the requested architecture (3 encoder blocks).
    Input: (N, 1, 64, 64, 64)
    Output: (N, 1, 64, 64, 64)
    """
    def __init__(self, n_channels=1, n_classes=1, base_filters=32):
        super(UNet3D, self).__init__()
        f = base_filters # 32

        # 1. Encoder (Contracting Path)
        self.inc = DoubleConv(n_channels, f)    # 1 -> 32
        self.down1 = Down(f, f * 2)             # 32 -> 64
        self.down2 = Down(f * 2, f * 4)         # 64 -> 128
        self.down3 = Down(f * 4, f * 8)         # 128 -> 256
        
        # 2. Bottleneck (Maintains 8x8x8 volume)
        self.bottleneck = DoubleConv(f * 8, f * 8) # 256 -> 256

        # 3. Decoder (Expanding Path)
        # Up block input channels = (up_output + skip_output)
        self.up1 = Up(f * 8 + f * 4, f * 4)     # (256+128=384) -> 128
        self.up2 = Up(f * 4 + f * 2, f * 2)     # (128+64=192) -> 64
        self.up3 = Up(f * 2 + f, f)             # (64+32=96) -> 32
        
        # 4. Output
        self.outc = OutConv(f, n_classes)       # 32 -> 1

    def forward(self, x):
        # Encoder & Skip Connections
        x1 = self.inc(x)            # E1: (64, 64, 64, 32C)
        x2 = self.down1(x1)         # E2: (32, 32, 32, 64C)
        x3 = self.down2(x2)         # E3: (16, 16, 16, 128C)
        x4 = self.down3(x3)         # E4: (8, 8, 8, 256C)

        # Bottleneck
        x5 = self.bottleneck(x4)    # Bottom: (8, 8, 8, 256C)

        # Decoder (x5 is upsampled and concatenated with x4)
        x = self.up1(x5, x4)        # D1: (16, 16, 16, 128C)
        x = self.up2(x, x3)         # D2: (32, 32, 32, 64C)
        x = self.up3(x, x2)         # D3: (64, 64, 64, 32C)

        # Output
        logits = self.outc(x)
        
        # Final layer for binary segmentation (n_classes=1)
        return torch.sigmoid(logits)