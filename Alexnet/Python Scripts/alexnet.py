import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv Block 2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv Block 3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Conv Block 4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # Conv Block 5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def count_parameters(model):
    total         = sum(p.numel() for p in model.parameters())
    trainable     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    print(f"{'='*45}")
    print(f"  Total parameters      : {total:,}")
    print(f"  Trainable parameters  : {trainable:,}")
    print(f"  Non-trainable params  : {non_trainable:,}")
    print(f"{'='*45}")

    print(f"\n{'Layer':<40} {'Shape':<25} {'Params':>10}")
    print("-" * 77)
    for name, param in model.named_parameters():
        print(f"{name:<40} {str(list(param.shape)):<25} {param.numel():>10,}")
    print("-" * 77)
    print(f"{'TOTAL':<66} {total:>10,}")


if __name__ == "__main__":
    model = AlexNet(num_classes=1000)
    print(model)
    print()

    dummy = torch.randn(2, 3, 227, 227)
    output = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {output.shape}\n")

    count_parameters(model)
