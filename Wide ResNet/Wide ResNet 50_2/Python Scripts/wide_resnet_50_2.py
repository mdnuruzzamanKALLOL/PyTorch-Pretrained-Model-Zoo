import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, planes, stride=1, downsample=None,
                 width_per_group=128):
        super().__init__()
        width          = int(planes * (width_per_group / 64.0))
        self.conv1      = nn.Conv2d(in_ch, width, 1, bias=False)
        self.bn1        = nn.BatchNorm2d(width)
        self.conv2      = nn.Conv2d(width, width, 3, stride=stride,
                                   padding=1, bias=False)
        self.bn2        = nn.BatchNorm2d(width)
        self.conv3      = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes * self.expansion)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class WideResNet(nn.Module):
    def __init__(self, layers, width_per_group=128, num_classes=1000):
        super().__init__()
        self.in_ch           = 64
        self.width_per_group = width_per_group

        self.conv1   = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1  = self._make_layer(64,  layers[0])
        self.layer2  = self._make_layer(128, layers[1], stride=2)
        self.layer3  = self._make_layer(256, layers[2], stride=2)
        self.layer4  = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_ch != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, planes * Bottleneck.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.in_ch, planes, stride, downsample,
                             self.width_per_group)]
        self.in_ch = planes * Bottleneck.expansion
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(self.in_ch, planes,
                                     width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def wide_resnet50_2(num_classes=1000):
    return WideResNet([3, 4, 6, 3], width_per_group=128, num_classes=num_classes)
