import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels,
                 expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze   = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, 1),
            nn.ReLU(inplace=True),
        )
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand1x1_channels, 1),
            nn.ReLU(inplace=True),
        )
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channels, expand3x3_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], dim=1)


class SqueezeNet(nn.Module):
    def __init__(self, version='1.0', num_classes=1000):
        super().__init__()
        if version == '1.0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, 7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                Fire(96,  16,  64,  64),
                Fire(128, 16,  64,  64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                Fire(64,  16,  64,  64),
                Fire(128, 16,  64,  64),
                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, num_classes, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def squeezenet1_0(num_classes=1000):
    return SqueezeNet('1.0', num_classes)


def squeezenet1_1(num_classes=1000):
    return SqueezeNet('1.1', num_classes)
