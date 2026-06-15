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


class HardSwish(nn.Module):
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class HardSigmoid(nn.Module):
    def forward(self, x):
        return F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        sq = _make_divisible(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, sq, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(sq, channels, 1),
            HardSigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, exp_ch, out_ch, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        act = HardSwish if use_hs else nn.ReLU

        layers = []
        if in_ch != exp_ch:
            layers += [
                nn.Conv2d(in_ch, exp_ch, 1, bias=False),
                nn.BatchNorm2d(exp_ch),
                act(),
            ]
        layers += [
            nn.Conv2d(exp_ch, exp_ch, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=exp_ch, bias=False),
            nn.BatchNorm2d(exp_ch),
            act(),
        ]
        if use_se:
            layers.append(SqueezeExcitation(exp_ch))
        layers += [
            nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


# (kernel, expand_ch, out_ch, use_se, use_hs, stride)
LARGE_CONFIG = [
    (3,   16,   16, False, False, 1),
    (3,   64,   24, False, False, 2),
    (3,   72,   24, False, False, 1),
    (5,   72,   40, True , False, 2),
    (5,  120,   40, True , False, 1),
    (5,  120,   40, True , False, 1),
    (3,  240,   80, False, True , 2),
    (3,  200,   80, False, True , 1),
    (3,  184,   80, False, True , 1),
    (3,  184,   80, False, True , 1),
    (3,  480,  112, True , True , 1),
    (3,  672,  112, True , True , 1),
    (5,  672,  160, True , True , 2),
    (5,  960,  160, True , True , 1),
    (5,  960,  160, True , True , 1),
]


class MobileNetV3(nn.Module):
    def __init__(self, bneck_config, last_conv_ch=960, last_channel=1280,
                 num_classes=1000, dropout=0.2):
        super(MobileNetV3, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish(),
        )

        layers, in_ch = [], 16
        for k, exp_ch, out_ch, se, hs, stride in bneck_config:
            layers.append(InvertedResidual(in_ch, exp_ch, out_ch, k, stride, se, hs))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_ch, last_conv_ch, 1, bias=False),
            nn.BatchNorm2d(last_conv_ch),
            HardSwish(),
        )

        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_ch, last_channel),
            HardSwish(),
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mobilenet_v3_large(num_classes=1000):
    return MobileNetV3(LARGE_CONFIG, last_conv_ch=960,
                       last_channel=1280, num_classes=num_classes)
