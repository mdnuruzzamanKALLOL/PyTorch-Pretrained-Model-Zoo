import torch
import torch.nn as nn


def channel_shuffle(x, groups=2):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super().__init__()
        self.stride = stride
        branch_features = oup // 2

        if stride == 1:
            self.branch1 = nn.Sequential()
            self.branch2 = nn.Sequential(
                nn.Conv2d(branch_features, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_features, branch_features, 3, stride=1, padding=1,
                          groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(inp, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_features, branch_features, 3, stride=stride, padding=1,
                          groups=branch_features, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, 1, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        else:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return channel_shuffle(out, groups=2)


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000):
        super().__init__()
        in_ch = stages_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, in_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.stage2, in_ch = self._make_stage(in_ch, stages_out_channels[1], stages_repeats[0])
        self.stage3, in_ch = self._make_stage(in_ch, stages_out_channels[2], stages_repeats[1])
        self.stage4, in_ch = self._make_stage(in_ch, stages_out_channels[3], stages_repeats[2])

        out_ch = stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(out_ch, num_classes)

    def _make_stage(self, in_ch, out_ch, num_blocks):
        layers = [InvertedResidual(in_ch, out_ch, stride=2)]
        for _ in range(num_blocks - 1):
            layers.append(InvertedResidual(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers), out_ch

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])
        return self.fc(x)


def shufflenet_v2_x0_5(num_classes=1000):
    return ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], num_classes)


def shufflenet_v2_x1_0(num_classes=1000):
    return ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], num_classes)


def shufflenet_v2_x1_5(num_classes=1000):
    return ShuffleNetV2([4, 8, 4], [24, 176, 352, 704, 1024], num_classes)


def shufflenet_v2_x2_0(num_classes=1000):
    return ShuffleNetV2([4, 8, 4], [24, 244, 488, 976, 2048], num_classes)
