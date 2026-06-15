import math
import torch
import torch.nn as nn


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def round_filters(filters, width_coefficient):
    return make_divisible(int(filters * width_coefficient))


def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(repeats * depth_coefficient))


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.fc1     = nn.Conv2d(in_channels, reduced_dim, kernel_size=1)
        self.act     = nn.SiLU()
        self.fc2     = nn.Conv2d(reduced_dim, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.pool(x)
        scale = self.act(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return x * scale


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, se_ratio=0.25, drop_path_rate=0.0):
        super().__init__()
        self.use_residual   = (in_channels == out_channels and stride == 1)
        self.drop_path_rate = drop_path_rate

        hidden_dim  = in_channels * expand_ratio
        reduced_dim = max(1, int(in_channels * se_ratio))

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
            ]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            if self.drop_path_rate > 0 and self.training:
                keep = 1 - self.drop_path_rate
                mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep
                return x + out * mask.float() / keep
            return x + out
        return out


# B0 baseline configuration:
# (expand_ratio, out_channels, num_repeats, stride, kernel_size)
BASE_CONFIG = [
    (1,  16, 1, 1, 3),
    (6,  24, 2, 2, 3),
    (6,  40, 2, 2, 5),
    (6,  80, 3, 2, 3),
    (6, 112, 3, 1, 5),
    (6, 192, 4, 2, 5),
    (6, 320, 1, 1, 3),
]


class EfficientNet(nn.Module):
    def __init__(self, width_coefficient=1.0, depth_coefficient=1.0,
                 dropout_rate=0.2, num_classes=1000):
        super().__init__()

        in_ch = round_filters(32, width_coefficient)
        self.stem = nn.Sequential(
            nn.Conv2d(3, in_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(),
        )

        total_blocks = sum(
            round_repeats(rep, depth_coefficient)
            for _, _, rep, _, _ in BASE_CONFIG
        )
        block_idx = 0
        blocks    = []
        for expand, out_ch, reps, stride, ks in BASE_CONFIG:
            out_c    = round_filters(out_ch, width_coefficient)
            num_reps = round_repeats(reps,   depth_coefficient)
            for i in range(num_reps):
                dr = 0.2 * block_idx / total_blocks
                blocks.append(MBConv(
                    in_channels    = in_ch,
                    out_channels   = out_c,
                    kernel_size    = ks,
                    stride         = stride if i == 0 else 1,
                    expand_ratio   = expand,
                    drop_path_rate = dr,
                ))
                in_ch      = out_c
                block_idx += 1
        self.blocks = nn.Sequential(*blocks)

        head_ch = round_filters(1280, width_coefficient)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_ch),
            nn.SiLU(),
        )
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(head_ch, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def efficientnet_b1(num_classes=1000):
    return EfficientNet(width_coefficient=1.0, depth_coefficient=1.1,
                        dropout_rate=0.2, num_classes=num_classes)
