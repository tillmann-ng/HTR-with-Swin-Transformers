import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    B = int(windows.shape[0] / (num_windows_h * num_windows_w))
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = torch.softmax(attn, dim=-1)
        else:
            attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_patch, num_heads, window_size=7, shift_size=0, num_mlp=1024, qkv_bias=True,
                 dropout_rate=0.0):
        super().__init__()
        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_mlp = num_mlp
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = WindowAttention(dim, (self.window_size, self.window_size), num_heads, qkv_bias, dropout_rate)
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(dim, num_mlp),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_mlp, dim),
            nn.Dropout(dropout_rate),
        )
        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)
        if self.shift_size > 0:
            height, width = self.num_patch
            pad_h = (self.window_size - height % self.window_size) % self.window_size
            pad_w = (self.window_size - width % self.window_size) % self.window_size
            img_mask = torch.zeros((1, height, width, 1))
            if pad_h > 0 or pad_w > 0:
                img_mask = F.pad(img_mask.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        height, width = self.num_patch
        B, L, C = x.shape
        x_skip = x
        x = self.norm1(x)
        x = x.view(B, height, width, C)
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h)).permute(0, 2, 3, 1)
        padded_height, padded_width = x.shape[1], x.shape[2]
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, padded_height, padded_width)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_h > 0 or pad_w > 0:
            x = x[:, :height, :width, :].contiguous()
        x = x.view(B, height * width, C)
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, num_patch, depth, num_heads, window_size, num_mlp, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_patch=num_patch, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2, num_mlp=num_mlp,
            ) for i in range(depth)
        ])
        if downsample:
            self.downsample = PatchMerging(dim=dim)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = H // 2, W // 2
        return x, H, W


class SwinBackbone(nn.Module):
    def __init__(self, img_size=(128, 1024), patch_size=4, in_chans=1, embed_dim=64,
                 depths=[2, 2, 4], num_heads=[2, 4, 8], window_size=7, num_mlp=256):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                num_patch=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                num_mlp=int(num_mlp * 2 ** i_layer),
                downsample=(i_layer < len(depths) - 1)
            )
            self.layers.append(layer)
        self.output_channels = int(embed_dim * 2 ** (len(depths) - 1))
        self.norm = nn.LayerNorm(self.output_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        current_H = H // self.patch_embed.patch_size[0]
        current_W = W // self.patch_embed.patch_size[1]
        for layer in self.layers:
            x, current_H, current_W = layer(x, current_H, current_W)
        x = self.norm(x)
        x = x.view(B, current_H, current_W, -1).permute(0, 3, 1, 2).contiguous()
        return x

    def get_output_channels(self):
        return self.output_channels


class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding='same')
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()  # Swish
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # -> (B, D, T)

        x = self.pointwise_conv1(x)  # -> (B, 2D, T)
        x = self.glu(x)  # -> (B, D, T)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        return x.transpose(1, 2)  # -> (B, T, D)


class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_expansion_factor=4, conv_kernel_size=15, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion_factor, d_model),
            nn.Dropout(dropout)
        )
        self.self_attn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True),
            nn.Dropout(dropout)
        )
        self.conv_module = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * ff_expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion_factor, d_model),
            nn.Dropout(dropout)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)

        # Multi-headed Self-Attention
        attn_out, _ = self.self_attn[1](self.self_attn[0](x), self.self_attn[0](x), self.self_attn[0](x))
        x = x + self.self_attn[2](attn_out)

        x = x + self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


class ConformerHead(nn.Module):
    def __init__(self, input_size, nclasses, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model=input_size, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(input_size, nclasses)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.size()

        # Gộp chiều cao, giữ lại chiều rộng làm sequence
        y = F.adaptive_avg_pool2d(x, (1, W))  # -> (B, C, 1, W)
        y = y.squeeze(2)  # -> (B, C, W)
        y = y.permute(0, 2, 1).contiguous()  # -> (B, W, C)

        for layer in self.layers:
            y = layer(y)

        y = self.fc(y)
        return y.permute(1, 0, 2)  # -> (W, B, NClasses)

class HTRNet(nn.Module):
    def __init__(self, arch_cfg, nclasses):
        super(HTRNet, self).__init__()

        self.features = SwinBackbone(
            img_size=tuple(arch_cfg.img_size),
            patch_size=arch_cfg.patch_size,
            in_chans=arch_cfg.in_chans,
            embed_dim=arch_cfg.embed_dim,
            depths=arch_cfg.depths,
            num_heads=arch_cfg.num_heads,
            window_size=arch_cfg.window_size,
            num_mlp=arch_cfg.num_mlp
        )

        hidden = self.features.get_output_channels()

        if arch_cfg.head_type == 'conformer':
            self.top = ConformerHead(
                input_size=hidden,
                nclasses=nclasses,
                num_heads=arch_cfg.head_num_heads,
                num_layers=arch_cfg.head_num_layers,
                dropout=arch_cfg.head_dropout
            )
        else:
            raise ValueError(f"Head type '{arch_cfg.head_type}' is not supported. Please use 'conformer'.")

    def forward(self, x):
        features = self.features(x)
        output_logits = self.top(features)

        batch_size = output_logits.size(1)
        seq_len = output_logits.size(0)

        act_lens = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.long, device=x.device)

        return output_logits, act_lens