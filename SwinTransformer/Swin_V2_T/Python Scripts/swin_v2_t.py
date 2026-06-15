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


class WindowAttentionV2(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads   = num_heads
        self.window_size = window_size

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads, 1, 1)))

        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        rch = torch.arange(-(window_size-1), window_size, dtype=torch.float32)
        rcw = torch.arange(-(window_size-1), window_size, dtype=torch.float32)
        rct = torch.stack(torch.meshgrid([rch, rcw], indexing='ij'), dim=-1).unsqueeze(0)
        rct[:, :, :, 0] /= (window_size - 1)
        rct[:, :, :, 1] /= (window_size - 1)
        rct = rct * 8
        rct = torch.sign(rct) * torch.log2(torch.abs(rct) + 1.0) / math.log2(8)
        self.register_buffer('relative_coords_table', rct)

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

        scale = torch.clamp(self.logit_scale, max=math.log(100)).exp()
        attn  = scale * F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)

        cpb = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        rpb = cpb[self.relative_position_index.view(-1)]
        rpb = rpb.view(self.window_size**2, self.window_size**2, -1)
        rpb = 16 * torch.sigmoid(rpb)
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlockV2(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., dropout=0., attn_drop=0.):
        super().__init__()
        self.shift_size  = shift_size
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttentionV2(dim, window_size, num_heads,
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

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_win = window_partition(x, self.window_size).view(-1, self.window_size**2, C)
        x_win = self.attn(x_win, mask=attn_mask)
        x     = window_reverse(
            x_win.view(-1, self.window_size, self.window_size, C), self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + self.norm1(x)
        x = x + self.norm2(self.mlp(x))
        return x


class SwinStageV2(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., dropout=0., attn_drop=0., downsample=None):
        super().__init__()
        self.window_size = window_size
        self.blocks = nn.ModuleList([
            SwinTransformerBlockV2(
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


class SwinTransformerV2(nn.Module):
    def __init__(self, img_size=256, patch_size=4, in_channels=3, num_classes=1000,
                 embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24),
                 window_size=8, mlp_ratio=4., dropout=0., attn_drop=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.pos_drop    = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            SwinStageV2(
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


def swin_v2_t(num_classes=1000):
    return SwinTransformerV2(
        img_size=256, embed_dim=96,
        depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
        window_size=8, num_classes=num_classes,
    )
