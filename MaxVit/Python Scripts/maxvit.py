import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        hidden = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, se_reduction=4):
        super(MBConv, self).__init__()
        hidden = in_channels * expansion
        self.pre_norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GELU(),
            SqueezeExcitation(hidden, se_reduction),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        return self.conv(self.pre_norm(x)) + self.shortcut(x)


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super(RelativePositionBias, self).__init__()
        self.window_size = window_size
        self.num_heads   = num_heads
        seq = 2 * window_size - 1
        self.bias_table  = nn.Parameter(torch.zeros(seq * seq, num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'
        ))
        coords_flat = coords.flatten(1)
        relative     = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative[0] += window_size - 1
        relative[1] += window_size - 1
        relative[0] *= 2 * window_size - 1
        idx = relative.sum(0)
        self.register_buffer('relative_index', idx)

    def forward(self):
        bias = self.bias_table[self.relative_index.view(-1)]
        bias = bias.view(self.window_size**2, self.window_size**2, self.num_heads)
        return bias.permute(2, 0, 1).unsqueeze(0)


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super(WindowAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3)
        self.proj      = nn.Linear(dim, dim)
        self.rel_bias  = RelativePositionBias(window_size, num_heads)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.rel_bias()
        attn = attn.softmax(dim=-1)
        x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class GridAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(GridAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MaxViTBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4):
        super(MaxViTBlock, self).__init__()
        self.window_size = window_size
        hidden           = int(dim * mlp_ratio)

        self.mbconv   = MBConv(dim, dim)
        self.norm1    = nn.LayerNorm(dim)
        self.win_attn = WindowAttention(dim, num_heads, window_size)
        self.norm2    = nn.LayerNorm(dim)
        self.mlp1     = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm3    = nn.LayerNorm(dim)
        self.grid_attn = GridAttention(dim, num_heads)
        self.norm4    = nn.LayerNorm(dim)
        self.mlp2     = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _partition_windows(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        x  = x.reshape(B, C, H // ws, ws, W // ws, ws)
        x  = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, C)
        return x, B, H, W

    def _unpartition_windows(self, x, B, H, W):
        ws = self.window_size
        C  = x.shape[-1]
        x  = x.reshape(B, H // ws, W // ws, ws, ws, C)
        x  = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return x

    def _partition_grid(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        x  = x.reshape(B, C, ws, H // ws, ws, W // ws)
        x  = x.permute(0, 3, 5, 2, 4, 1).reshape(-1, ws * ws, C)
        return x, B, H, W

    def _unpartition_grid(self, x, B, H, W):
        ws = self.window_size
        C  = x.shape[-1]
        x  = x.reshape(B, H // ws, W // ws, ws, ws, C)
        x  = x.permute(0, 5, 3, 1, 4, 2).reshape(B, C, H, W)
        return x

    def forward(self, x):
        x = self.mbconv(x)

        tokens, B, H, W = self._partition_windows(x)
        tokens = tokens + self.win_attn(self.norm1(tokens))
        tokens = tokens + self.mlp1(self.norm2(tokens))
        x      = self._unpartition_windows(tokens, B, H, W)

        tokens, B, H, W = self._partition_grid(x)
        tokens = tokens + self.grid_attn(self.norm3(tokens))
        tokens = tokens + self.mlp2(self.norm4(tokens))
        x      = self._unpartition_grid(tokens, B, H, W)

        return x


class MaxViTStage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, num_heads, window_size=8):
        super(MaxViTStage, self).__init__()
        self.downsample = MBConv(in_channels, out_channels)
        self.blocks     = nn.Sequential(*[
            MaxViTBlock(out_channels, num_heads, window_size)
            for _ in range(depth)
        ])

    def forward(self, x):
        return self.blocks(self.downsample(x))


class MaxViT(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000,
                 depths=(2, 2, 5, 2), channels=(64, 128, 256, 512),
                 num_heads=(2, 4, 8, 16), window_size=8):
        super(MaxViT, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )

        stage_in   = [channels[0]] + list(channels[:-1])
        self.stages = nn.Sequential(*[
            MaxViTStage(stage_in[i], channels[i], depths[i], num_heads[i], window_size)
            for i in range(len(depths))
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], channels[-1]),
            nn.Tanh(),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


def maxvit_tiny(num_classes=1000):
    return MaxViT(
        num_classes = num_classes,
        depths      = (2, 2, 5, 2),
        channels    = (64, 128, 256, 512),
        num_heads   = (2, 4, 8, 16),
        window_size = 8,
    )
