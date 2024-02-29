from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor

# Almost all codes in this file is taken from torchvision

def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self,
        in_channel: int,
        out_channel: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channel, out_channel, stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = None
        if stride != 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                conv1x1(in_channel, out_channel, stride),
                nn.BatchNorm2d(out_channel),
            )
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        return out


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)
