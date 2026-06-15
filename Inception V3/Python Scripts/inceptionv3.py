import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBnRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_proj):
        super(InceptionA, self).__init__()

        self.branch1 = ConvBnRelu(in_channels, 64, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_channels, 48, kernel_size=1),
            ConvBnRelu(48, 64, kernel_size=5, padding=2),
        )
        self.branch3 = nn.Sequential(
            ConvBnRelu(in_channels, 64, kernel_size=1),
            ConvBnRelu(64, 96, kernel_size=3, padding=1),
            ConvBnRelu(96, 96, kernel_size=3, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()

        self.branch1 = ConvBnRelu(in_channels, 384, kernel_size=3, stride=2)
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_channels, 64, kernel_size=1),
            ConvBnRelu(64, 96, kernel_size=3, padding=1),
            ConvBnRelu(96, 96, kernel_size=3, stride=2),
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        c7 = channels_7x7

        self.branch1 = ConvBnRelu(in_channels, 192, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_channels, c7, kernel_size=1),
            ConvBnRelu(c7, c7,  kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(c7, 192, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.branch3 = nn.Sequential(
            ConvBnRelu(in_channels, c7, kernel_size=1),
            ConvBnRelu(c7, c7,  kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(c7, c7,  kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(c7, c7,  kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(c7, 192, kernel_size=(1, 7), padding=(0, 3)),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_channels, 192, kernel_size=1),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class InceptionD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBnRelu(in_channels, 192, kernel_size=1),
            ConvBnRelu(192, 320, kernel_size=3, stride=2),
        )
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_channels, 192, kernel_size=1),
            ConvBnRelu(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(192, 192, kernel_size=3, stride=2),
        )
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class InceptionE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionE, self).__init__()

        self.branch1      = ConvBnRelu(in_channels, 320, kernel_size=1)
        self.branch2_conv = ConvBnRelu(in_channels, 384, kernel_size=1)
        self.branch2a     = ConvBnRelu(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch2b     = ConvBnRelu(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3_conv = nn.Sequential(
            ConvBnRelu(in_channels, 448, kernel_size=1),
            ConvBnRelu(448, 384, kernel_size=3, padding=1),
        )
        self.branch3a     = ConvBnRelu(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3b     = ConvBnRelu(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch4      = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_channels, 192, kernel_size=1),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2_conv(x)
        b2 = torch.cat([self.branch2a(b2), self.branch2b(b2)], dim=1)
        b3 = self.branch3_conv(x)
        b3 = torch.cat([self.branch3a(b3), self.branch3b(b3)], dim=1)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()

        self.pool    = nn.AdaptiveAvgPool2d((5, 5))
        self.conv    = ConvBnRelu(in_channels, 128, kernel_size=1)
        self.fc1     = nn.Linear(128 * 5 * 5, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2     = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, use_aux=True):
        super(InceptionV3, self).__init__()
        self.use_aux = use_aux

        self.stem = nn.Sequential(
            ConvBnRelu(3, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 32, kernel_size=3),
            ConvBnRelu(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvBnRelu(64, 80, kernel_size=1),
            ConvBnRelu(80, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.inceptionA1 = InceptionA(192, pool_proj=32)
        self.inceptionA2 = InceptionA(256, pool_proj=64)
        self.inceptionA3 = InceptionA(288, pool_proj=64)
        self.inceptionB  = InceptionB(288)

        self.inceptionC1 = InceptionC(768, channels_7x7=128)
        self.inceptionC2 = InceptionC(768, channels_7x7=160)
        self.inceptionC3 = InceptionC(768, channels_7x7=160)
        self.inceptionC4 = InceptionC(768, channels_7x7=192)

        if self.use_aux:
            self.aux = AuxClassifier(768, num_classes)

        self.inceptionD  = InceptionD(768)
        self.inceptionE1 = InceptionE(1280)
        self.inceptionE2 = InceptionE(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc      = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.inceptionA1(x)
        x = self.inceptionA2(x)
        x = self.inceptionA3(x)
        x = self.inceptionB(x)

        x = self.inceptionC1(x)
        x = self.inceptionC2(x)
        x = self.inceptionC3(x)
        x = self.inceptionC4(x)
        aux_out = self.aux(x) if (self.use_aux and self.training) else None

        x = self.inceptionD(x)
        x = self.inceptionE1(x)
        x = self.inceptionE2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.use_aux and self.training:
            return x, aux_out
        return x


def inception_v3(num_classes=1000):
    return InceptionV3(num_classes=num_classes, use_aux=True)
