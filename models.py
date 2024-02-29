import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from model_utils import BasicBlock, FCNHead

logger = logging.getLogger(__name__)


class FCNBase(nn.Module):
    # init the fully convolutional network with a VGG16 backbone
    # do not use the torchvision
    def __init__(self):
        super(FCNBase, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.conv2 = two_conv_block(64, 128)
        self.conv3 = three_conv_block(128, 256)
        self.conv4 = three_conv_block(256, 512)
        self.conv5 = three_conv_block(512, 512)

        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')
                torch.nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        raise NotImplementedError

    def compute_loss(self, image, mask, criterion):
        raise NotImplementedError


class FCN32s(FCNBase):
    def __init__(self, num_classes=21):
        super(FCN32s, self).__init__()
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.upscore = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=64, stride=32, bias=False
        )
        self._initialize_weights()

        logger.info(
            "number of parameters: %.2f M", 
            sum(p.numel() for p in self.parameters()) / 1e6
        )

    def forward(self, x):
        """
        note: return is the 32x of last feature map, not the original size
        """
        input_shape = x.shape[-2:]

        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h) # 64, h/2+99, w/2+99

        h = self.conv2(h) # 128, (h+2)/4+49, (w+2)/4+49
        h = self.conv3(h) # 256, (h+6)/8+24, (w+6)/4+24
        h = self.conv4(h) # 512, (h+6)/16+12, (w+6)/16+12
        h = self.conv5(h) # 512, (h+6)/32+6, (w+6)/32+6
        
        h = self.relu6(self.fc6(h))
        h = self.drop6(h) # 4096, (h+6)/32, (w+6)/32
        h = self.relu7(self.fc7(h))
        h = self.drop7(h) # 4096, (h+6)/32, (w+6)/32

        h = self.score_fr(h) # num_classes, (h+6)/32, (w+6)/32
        h = self.upscore(h) # num_classes, h+38, w+38

        h = nn.functional.interpolate(
            h, size=input_shape, mode='bilinear', align_corners=False
        )

        return h
    
    def compute_loss(self, image, mask, criterion):
        pred = self(image)

        # compute loss
        loss = criterion(pred, mask)
        return loss


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self.make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(BasicBlock, 512, 2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, out_channel, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, out_channel, stride))
            self.in_channel = out_channel
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)  # 64, h/4, w/4
        x = self.layer1(x)  # 64, h/4, w/4
        layer2 = self.layer2(x)  # 128, h/8, w/8
        layer3 = self.layer3(layer2)  # 256, h/16, w/16
        layer4 = self.layer4(layer3)  # 512, h/32, w/32
        return {
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4
        }


class ResNet18FCN32s(nn.Module):
    def __init__(self, num_classes=21):
        super(ResNet18FCN32s, self).__init__()
        self.backbone = ResNet18()
        self.classifier = FCNHead(512, num_classes)
        logger.info(
            "number of parameters: %.2f M", 
            sum(p.numel() for p in self.parameters()) / 1e6
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = features["layer4"]
        x = self.classifier(x)  # num_classes, h/32, w/32
        x = F.interpolate(
            x, size=input_shape, mode='bilinear', align_corners=False
        )

        return x

    def compute_loss(self, image, mask, criterion):
        pred = self(image)

        # compute loss
        loss = criterion(pred, mask)
        return loss
    
class ResNet18PretrainedFCN32s(nn.Module):
    def __init__(self, num_classes=21):
        super(ResNet18PretrainedFCN32s, self).__init__()
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.backbone = IntermediateLayerGetter(
            model, return_layers={"layer4": "layer4"}
        )
        self.classifier = FCNHead(512, num_classes)
        logger.info(
            "number of parameters: %.2f M", 
            sum(p.numel() for p in self.parameters()) / 1e6
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = features["layer4"]
        x = self.classifier(x)  # num_classes, h/32, w/32
        x = F.interpolate(
            x, size=input_shape, mode='bilinear', align_corners=False
        )
        return x

    def compute_loss(self, image, mask, criterion):
        pred = self(image)

        # compute loss
        loss = criterion(pred, mask)
        return loss


def two_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )

def three_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    )
