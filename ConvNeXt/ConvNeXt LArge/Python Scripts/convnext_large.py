import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var  = (x - mean).pow(2).mean(1, keepdim=True)
        x    = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv  = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm    = LayerNorm(dim, eps=1e-6, data_format='channels_last')
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma   = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return residual + x


class ConvNeXt(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000,
                 depths=(3,3,27,3), dims=(192,384,768,1536),
                 layer_scale_init=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format='channels_first')
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            ds = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format='channels_first'),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(ds)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i], layer_scale_init) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.head(x)
        return x


def convnext_large(num_classes=1000):
    return ConvNeXt(depths=(3,3,27,3), dims=(192,384,768,1536), num_classes=num_classes)


def count_parameters(model, model_name="ConvNeXt-Large"):
    total        = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_train    = total - trainable

    print(f"{'='*45}")
    print(f"  Model           : {model_name}")
    print(f"  Total params    : {total:,}")
    print(f"  Trainable       : {trainable:,}")
    print(f"  Non-trainable   : {non_train:,}")
    print(f"{'='*45}")

    print(f"\n{'Layer':<50} {'Shape':<25} {'Params':>10}")
    print("-" * 87)
    for name, param in model.named_parameters():
        print(f"{name:<50} {str(list(param.shape)):<25} {param.numel():>10,}")
    print("-" * 87)
    print(f"{'TOTAL':<76} {total:>10,}")


if __name__ == "__main__":
    model = convnext_large(num_classes=1000)
    print(model)

    dummy  = torch.randn(2, 3, 224, 224)
    output = model(dummy)
    print(f"\nInput  shape : {dummy.shape}")
    print(f"Output shape : {output.shape}\n")

    count_parameters(model, "ConvNeXt-Large")
