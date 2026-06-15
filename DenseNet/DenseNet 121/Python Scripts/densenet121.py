import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.norm1     = nn.BatchNorm2d(num_input_features)
        self.conv1     = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, bias=False)
        self.norm2     = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2     = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.norm(x), inplace=True)))


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6,12,24,16),
                 num_init_features=64, bn_size=4, drop_rate=0.0, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        num_features = num_init_features
        self.dense_blocks = nn.ModuleList()
        self.transitions  = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            self.dense_blocks.append(DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate))
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                self.transitions.append(Transition(num_features, num_features // 2))
                num_features //= 2
        self.norm_final = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):     nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):   nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = F.relu(self.norm_final(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return self.classifier(torch.flatten(x, 1))


def densenet121(num_classes=1000):
    return DenseNet(growth_rate=32, block_config=(6,12,24,16), num_init_features=64, num_classes=num_classes)


def count_parameters(model, model_name="DenseNet-121"):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'='*45}")
    print(f"  Model           : {model_name}")
    print(f"  Total params    : {total:,}")
    print(f"  Trainable       : {trainable:,}")
    print(f"  Non-trainable   : {total - trainable:,}")
    print(f"{'='*45}")
    print(f"\n{'Layer':<55} {'Shape':<25} {'Params':>10}")
    print("-" * 92)
    for name, param in model.named_parameters():
        print(f"{name:<55} {str(list(param.shape)):<25} {param.numel():>10,}")
    print("-" * 92)
    print(f"{'TOTAL':<81} {total:>10,}")


if __name__ == "__main__":
    model = densenet121(num_classes=1000)
    print(model)
    dummy  = torch.randn(2, 3, 224, 224)
    output = model(dummy)
    print(f"\nInput  shape : {dummy.shape}")
    print(f"Output shape : {output.shape}\n")
    count_parameters(model, "DenseNet-121")
