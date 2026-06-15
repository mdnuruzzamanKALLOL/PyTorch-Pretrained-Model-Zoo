import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# (expand_ratio, out_channels, num_blocks, stride)
STAGE_CONFIG = [
    (1,  16, 1, 1),
    (6,  24, 2, 2),
    (6,  32, 3, 2),
    (6,  64, 4, 2),
    (6,  96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden            = int(round(in_ch * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, dropout=0.2):
        super(MobileNetV2, self).__init__()

        def c(ch): return _make_divisible(int(ch * width_mult))

        layers = [
            nn.Conv2d(3, c(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU6(inplace=True),
        ]

        in_ch = c(32)
        for expand, out, blocks, stride in STAGE_CONFIG:
            out_ch = c(out)
            for i in range(blocks):
                layers.append(InvertedResidual(in_ch, out_ch, stride if i == 0 else 1, expand))
                in_ch = out_ch

        last_ch = c(1280) if width_mult > 1.0 else 1280
        layers += [
            nn.Conv2d(in_ch, last_ch, 1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU6(inplace=True),
        ]

        self.features   = nn.Sequential(*layers)
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_ch, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mobilenet_v2(num_classes=1000):
    return MobileNetV2(num_classes=num_classes)
