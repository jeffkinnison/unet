import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):

    def __init__(self, in_channels, min_filters):
        super(UNet3D, self).__init__()

        prev_filters = in_channels
        curr_filters = min_filters
        self.input = ConvBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters * 2)
        self.down1 = DownBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters * 2)
        self.down2 = DownBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters * 2)
        self.down3 = DownBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters / 2)
        self.up1 = UpBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters / 2)
        self.up2 = UpBlock(prev_filters, curr_filters, 3)

        prev_filters = curr_filters
        curr_filters = int(curr_filters / 2)
        self.up3 = UpBlock(prev_filters, curr_filters, 3)

        self.out = OutBlock(curr_filters)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)

        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.out(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shape):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, shape, padding=1)
        self.conv1 = self.conv1.half()
        self.conv2 = nn.Conv3d(out_channels, out_channels, shape, padding=1)
        self.conv2 = self.conv2.half()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shape):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, shape)

    def forward(self, x):
        x = F.max_pool3d(x, (1, 2, 2))
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shape):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.upsample.half()
        self.conv = ConvBlock(in_channels + out_channels, out_channels, shape)

    def forward(self, x, x2):
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv(x)
        return x


class OutBlock(nn.Module):
    def __init__(self, in_channels):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, 1, 1)
        self.conv.half()

    def forward(self, x):
        x = self.conv(x)
        return x
