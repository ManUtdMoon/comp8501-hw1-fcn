import numpy as np
import torch
import torch.nn as nn


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


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
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

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

    def forward(self, x):
        """
        note: return is the 32x of last feature map, not the original size
        """
        height = x.shape[2]
        width = x.shape[3]

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

        h = h[:, :, 19:19+height, 19:19+width].contiguous()

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
