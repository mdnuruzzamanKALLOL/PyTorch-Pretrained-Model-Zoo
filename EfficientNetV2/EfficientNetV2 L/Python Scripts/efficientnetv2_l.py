import torch
import torch.nn as nn


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


class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expand_ratio, drop_path_rate=0.0):
        super().__init__()
        self.use_residual   = (in_channels == out_channels and stride == 1)
        self.drop_path_rate = drop_path_rate

        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size,
                          stride=stride, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            if self.drop_path_rate > 0 and self.training:
                keep = 1 - self.drop_path_rate
                mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < keep
                return x + out * mask.float() / keep
            return x + out
        return out


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


# (block_type, kernel_size, stride, expand_ratio, out_channels, num_layers, se_ratio)
# 'f' = FusedMBConv  |  'm' = MBConv
BASE_CONFIG = [
    ('f', 3, 1, 1,  32,  4, 0.00),
    ('f', 3, 2, 4,  64,  7, 0.00),
    ('f', 3, 2, 4,  96,  7, 0.00),
    ('m', 3, 2, 4, 192, 10, 0.25),
    ('m', 3, 1, 6, 224, 19, 0.25),
    ('m', 3, 2, 6, 384, 25, 0.25),
    ('m', 3, 1, 6, 640,  7, 0.25),
]


class EfficientNetV2(nn.Module):
    def __init__(self, stage_config, stem_channels=24, head_channels=1280,
                 dropout_rate=0.2, num_classes=1000):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(),
        )

        total_blocks = sum(n for _, _, _, _, _, n, _ in stage_config)
        block_idx    = 0
        in_ch        = stem_channels
        blocks       = []
        for btype, ks, stride, expand, out_ch, n_layers, se_ratio in stage_config:
            for i in range(n_layers):
                dr       = 0.2 * block_idx / total_blocks
                stride_i = stride if i == 0 else 1
                if btype == 'f':
                    blocks.append(FusedMBConv(
                        in_channels    = in_ch,
                        out_channels   = out_ch,
                        kernel_size    = ks,
                        stride         = stride_i,
                        expand_ratio   = expand,
                        drop_path_rate = dr,
                    ))
                else:
                    blocks.append(MBConv(
                        in_channels    = in_ch,
                        out_channels   = out_ch,
                        kernel_size    = ks,
                        stride         = stride_i,
                        expand_ratio   = expand,
                        se_ratio       = se_ratio,
                        drop_path_rate = dr,
                    ))
                in_ch      = out_ch
                block_idx += 1
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(),
        )
        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(head_channels, num_classes),
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


def efficientnetv2_l(num_classes=1000):
    return EfficientNetV2(
        stage_config  = BASE_CONFIG,
        stem_channels = 32,
        dropout_rate  = 0.4,
        num_classes   = num_classes,
    )
