import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=8):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# (kernel_size, out_channels, repeats, stride, expand_ratio)
BASE_CONFIG = [
    (3,  16, 1, 1, 1),
    (3,  24, 3, 2, 3),
    (5,  40, 3, 2, 3),
    (3,  80, 3, 2, 6),
    (3,  96, 2, 1, 6),
    (5, 192, 4, 2, 6),
    (3, 320, 1, 1, 6),
]


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden            = _make_divisible(in_ch * expand_ratio)

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride,
                          padding=kernel_size // 2, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, kernel_size, stride=stride,
                          padding=kernel_size // 2, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MNASNet(nn.Module):
    def __init__(self, alpha=1.0, num_classes=1000, dropout=0.2):
        super(MNASNet, self).__init__()

        def c(ch): return _make_divisible(int(ch * alpha))

        self.stem = nn.Sequential(
            nn.Conv2d(3, c(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU(inplace=True),
        )

        layers, in_ch = [], c(32)
        for k, out, repeat, stride, expand in BASE_CONFIG:
            out_ch = c(out)
            for i in range(repeat):
                layers.append(InvertedResidual(in_ch, out_ch, k,
                                               stride if i == 0 else 1, expand))
                in_ch = out_ch
        self.layers = nn.Sequential(*layers)

        self.head_conv = nn.Sequential(
            nn.Conv2d(in_ch, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        x = self.head_conv(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mnasnet(num_classes=1000):
    return MNASNet(alpha=1.0, num_classes=num_classes)
