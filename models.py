import numpy as np
import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from model_utils import BasicBlock, FCNHead

logger = logging.getLogger(__name__)


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


class ResNet50PretrainedFCN32s(nn.Module):
    def __init__(self, num_classes=21):
        super(ResNet50PretrainedFCN32s, self).__init__()
        model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        self.backbone = IntermediateLayerGetter(
            model, return_layers={"layer4": "layer4"}
        )
        self.classifier = FCNHead(2048, num_classes)
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


class ResNet50PretrainedFCN16s(nn.Module):
    def __init__(self, num_classes=21):
        super(ResNet50PretrainedFCN16s, self).__init__()
        model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        self.backbone = IntermediateLayerGetter(
            model, return_layers={"layer3": "layer3", "layer4": "layer4"}
        )
        self.layer4_classifier = FCNHead(2048, num_classes)
        self.layer3_classifier = FCNHead(1024, num_classes)
        logger.info(
            "number of parameters: %.2f M", 
            sum(p.numel() for p in self.parameters()) / 1e6
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = features["layer4"]
        x = self.layer4_classifier(x)  # nc, h/32, w/32
        x = F.interpolate(
            x, size=features["layer3"].shape[-2:], mode='bilinear', align_corners=False
        )  # nc, h/16, w/16
        x = x + self.layer3_classifier(features["layer3"])  # nc, h/16, w/16
        x = F.interpolate(
            x, size=input_shape, mode='bilinear', align_corners=False
        )  # nc, h, w

        return x

    def compute_loss(self, image, mask, criterion):
        pred = self(image)

        # compute loss
        loss = criterion(pred, mask)
        return loss
    

class ResNet50PretrainedFCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(ResNet50PretrainedFCN8s, self).__init__()
        model = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        self.backbone = IntermediateLayerGetter(
            model, 
            return_layers={
                "layer2": "layer2",
                "layer3": "layer3",
                "layer4": "layer4"
            }
        )
        self.layer4_classifier = FCNHead(2048, num_classes)
        self.layer3_classifier = FCNHead(1024, num_classes)
        self.layer2_classifier = FCNHead(512, num_classes)
        logger.info(
            "number of parameters: %.2f M", 
            sum(p.numel() for p in self.parameters()) / 1e6
        )
    
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = features["layer4"]
        feat3 = features["layer3"]
        feat2 = features["layer2"]
        x = self.layer4_classifier(x) # nc, h/32, w/32
        x = F.interpolate(x, 
            size=feat3.shape[-2:], mode='bilinear', align_corners=False
        ) # nc, h/16, w/16

        x = x + self.layer3_classifier(feat3) # nc, h/16, w/16
        x = F.interpolate(x, 
            size=feat2.shape[-2:], mode='bilinear', align_corners=False
        ) # nc, h/8, w/8

        x = x + self.layer2_classifier(feat2) # nc, h/8, w/8
        x = F.interpolate(x, 
            size=input_shape, mode='bilinear', align_corners=False
        ) # nc, h, w

        return x


    def compute_loss(self, image, mask, criterion):
        pred = self(image)
        loss = criterion(pred, mask)
        return loss
