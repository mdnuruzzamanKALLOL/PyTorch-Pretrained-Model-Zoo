import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x  = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x  = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x)), H // 2, W // 2


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads   = num_heads
        self.window_size = window_size
        self.scale       = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size-1) * (2*window_size-1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_f = torch.flatten(coords, 1)
        rel      = coords_f[:, :, None] - coords_f[:, None, :]
        rel      = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('relative_position_index', rel.sum(-1))

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        rpb  = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size**2, self.window_size**2, -1)
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., dropout=0., attn_drop=0.):
        super().__init__()
        self.shift_size  = shift_size
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, window_size, num_heads,
                                     attn_drop=attn_drop, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        B, H, W, C = x.shape
        shortcut    = x
        x           = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_win = window_partition(x, self.window_size).view(-1, self.window_size**2, C)
        x_win = self.attn(x_win, mask=attn_mask)
        x     = window_reverse(
            x_win.view(-1, self.window_size, self.window_size, C), self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., dropout=0., attn_drop=0., downsample=None):
        super().__init__()
        self.window_size = window_size
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, dropout=dropout, attn_drop=attn_drop,
            ) for i in range(depth)
        ])
        self.downsample = downsample(dim) if downsample else None

    def forward(self, x, H, W):
        shift_size = self.window_size // 2 if min(H, W) > self.window_size else 0
        if shift_size > 0:
            img_mask = torch.zeros((1, H, W, 1), device=x.device)
            for h_s, w_s, cnt in [
                (slice(0, -self.window_size), slice(0, -self.window_size), 0),
                (slice(0, -self.window_size), slice(-self.window_size, -shift_size), 1),
                (slice(0, -self.window_size), slice(-shift_size, None), 2),
                (slice(-self.window_size, -shift_size), slice(0, -self.window_size), 3),
                (slice(-self.window_size, -shift_size), slice(-self.window_size, -shift_size), 4),
                (slice(-self.window_size, -shift_size), slice(-shift_size, None), 5),
                (slice(-shift_size, None), slice(0, -self.window_size), 6),
                (slice(-shift_size, None), slice(-self.window_size, -shift_size), 7),
                (slice(-shift_size, None), slice(-shift_size, None), 8),
            ]:
                img_mask[:, h_s, w_s, :] = cnt
            mw = window_partition(img_mask, self.window_size).view(-1, self.window_size**2)
            attn_mask = (mw.unsqueeze(1) - mw.unsqueeze(2))
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        B, _, C = x.shape
        x = x.view(B, H, W, C)
        for blk in self.blocks:
            x = blk(x, attn_mask if blk.shift_size > 0 else None)
        x = x.view(B, H * W, C)
        if self.downsample:
            x, H, W = self.downsample(x, H, W)
        return x, H, W


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, num_classes=1000,
                 embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24),
                 window_size=7, mlp_ratio=4., dropout=0., attn_drop=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_drop    = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            SwinStage(
                dim=embed_dim * (2**i), depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, mlp_ratio=mlp_ratio,
                dropout=dropout, attn_drop=attn_drop,
                downsample=PatchMerging if i < len(depths) - 1 else None,
            ) for i in range(len(depths))
        ])
        num_features = embed_dim * (2 ** (len(depths) - 1))
        self.norm = nn.LayerNorm(num_features)
        self.head = nn.Linear(num_features, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x, H, W = layer(x, H, W)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


def swin_b(num_classes=1000):
    return SwinTransformer(
        img_size=224, embed_dim=128,
        depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32),
        window_size=7, num_classes=num_classes,
    )
