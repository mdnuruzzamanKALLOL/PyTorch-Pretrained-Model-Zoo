import torch
import torch.nn as nn


class SE(nn.Module):
    def __init__(self, width, se_ratio=0.25):
        super().__init__()
        width_se = max(1, int(round(width * se_ratio)))
        self.excitation = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, width_se, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(width_se, width, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.excitation(x)


class XBlock(nn.Module):
    def __init__(self, width_in, width_out, stride, group_width,
                 bottleneck_multiplier=1.0, se_ratio=0.0):
        super().__init__()
        width_b = int(round(width_out * bottleneck_multiplier))
        groups  = max(1, width_b // group_width)
        layers  = [
            nn.Conv2d(width_in, width_b, 1, bias=False),
            nn.BatchNorm2d(width_b),
            nn.ReLU(inplace=True),
            nn.Conv2d(width_b, width_b, 3, stride=stride, padding=1,
                      groups=groups, bias=False),
            nn.BatchNorm2d(width_b),
            nn.ReLU(inplace=True),
        ]
        if se_ratio > 0.0:
            layers.append(SE(width_b, se_ratio))
        layers += [
            nn.Conv2d(width_b, width_out, 1, bias=False),
            nn.BatchNorm2d(width_out),
        ]
        self.f    = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Conv2d(width_in, width_out, 1, stride=stride, bias=False),
            nn.BatchNorm2d(width_out),
        ) if (width_in != width_out or stride != 1) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.f(x) + self.proj(x))


class RegNet(nn.Module):
    def __init__(self, stem_width, depths, widths, group_widths,
                 bottleneck_multiplier=1.0, se_ratio=0.0, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
        )
        prev_w  = stem_width
        stages  = []
        for depth, width, gw in zip(depths, widths, group_widths):
            blocks = [XBlock(prev_w, width, stride=2, group_width=gw,
                             bottleneck_multiplier=bottleneck_multiplier,
                             se_ratio=se_ratio)]
            for _ in range(depth - 1):
                blocks.append(XBlock(width, width, stride=1, group_width=gw,
                                     bottleneck_multiplier=bottleneck_multiplier,
                                     se_ratio=se_ratio))
            stages.append(nn.Sequential(*blocks))
            prev_w = width
        self.trunk_output = nn.Sequential(*stages)
        self.avgpool      = nn.AdaptiveAvgPool2d(1)
        self.fc           = nn.Linear(prev_w, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk_output(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ── RegNet X (no SE) ──
def regnet_x_400mf(num_classes=1000):
    return RegNet(32, [1,2,7,12], [32,64,160,384],     [16,16,16,16],   se_ratio=0.0, num_classes=num_classes)

def regnet_x_800mf(num_classes=1000):
    return RegNet(32, [1,3,7,5],  [64,128,288,672],    [16,16,16,16],   se_ratio=0.0, num_classes=num_classes)

def regnet_x_1_6gf(num_classes=1000):
    return RegNet(32, [2,4,10,2], [72,168,408,912],    [24,24,24,24],   se_ratio=0.0, num_classes=num_classes)

def regnet_x_3_2gf(num_classes=1000):
    return RegNet(32, [2,6,15,2], [96,192,432,1008],   [48,48,48,48],   se_ratio=0.0, num_classes=num_classes)

def regnet_x_8gf(num_classes=1000):
    return RegNet(32, [2,5,15,1], [80,240,720,1920],   [120,120,120,120], se_ratio=0.0, num_classes=num_classes)

def regnet_x_16gf(num_classes=1000):
    return RegNet(32, [2,6,13,1], [256,512,896,2048],  [128,128,128,128], se_ratio=0.0, num_classes=num_classes)

def regnet_x_32gf(num_classes=1000):
    return RegNet(32, [2,7,13,1], [336,672,1344,2520], [168,168,168,168], se_ratio=0.0, num_classes=num_classes)


# ── RegNet Y (with SE, se_ratio=0.25) ──
def regnet_y_400mf(num_classes=1000):
    return RegNet(32, [1,3,6,6],  [48,104,208,440],    [8,8,8,8],       se_ratio=0.25, num_classes=num_classes)

def regnet_y_800mf(num_classes=1000):
    return RegNet(32, [1,3,8,2],  [64,128,320,768],    [16,16,16,16],   se_ratio=0.25, num_classes=num_classes)

def regnet_y_1_6gf(num_classes=1000):
    return RegNet(32, [2,6,17,2], [48,120,336,888],    [24,24,24,24],   se_ratio=0.25, num_classes=num_classes)

def regnet_y_3_2gf(num_classes=1000):
    return RegNet(32, [2,5,13,1], [72,216,576,1512],   [24,24,24,24],   se_ratio=0.25, num_classes=num_classes)

def regnet_y_8gf(num_classes=1000):
    return RegNet(32, [2,17,5,1], [168,448,896,2016],  [56,56,56,56],   se_ratio=0.25, num_classes=num_classes)

def regnet_y_16gf(num_classes=1000):
    return RegNet(32, [2,4,11,1], [224,448,1232,3024], [112,112,112,112], se_ratio=0.25, num_classes=num_classes)

def regnet_y_32gf(num_classes=1000):
    return RegNet(32, [2,5,12,1], [232,696,1392,3712], [232,232,232,232], se_ratio=0.25, num_classes=num_classes)

def regnet_y_128gf(num_classes=1000):
    return RegNet(32, [2,7,13,1], [528,1056,2904,7392],[264,264,264,264], se_ratio=0.25, num_classes=num_classes)
