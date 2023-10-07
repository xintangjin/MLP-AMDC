import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
from timm.models.layers import DropPath, to_2tuple
import time

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class FeedForward_spectral(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class SpectralMLP(nn.Module):
    def __init__(
            self,
            dim,
            channel_dim,
    ):
        super().__init__()
        self.LN = nn.LayerNorm(channel_dim)
        self.ML = FeedForward_spectral(dim=channel_dim, hidden_dim=channel_dim)
        self.mapping = nn.Linear(dim, dim, bias=False)

        self.emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        v = self.mapping(x_in)
        x = rearrange(x_in, 'b h w c -> b (h w) c', h=h, w=w)
        x = self.LN(x)
        x = self.ML(x)
        x = rearrange(x, 'b (h w) c -> b h w c', h=h, w=w)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.emb(v.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class SpectralAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                SpectralMLP(dim=dim, channel_dim=dim),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None

        output:(num_windows*B, N, C)
                """
        B_, N, C = x.shape
        # print(x.shape)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            # H,  W= self.input_resolution, self.input_resolution
            H = self.input_resolution[0]
            W = self.input_resolution[1]
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            #print(mask_windows.shape)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        # x: [b,h,w,c]
        B, H, W, C = x.shape
        x = self.norm1(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class SpatialMLP(nn.Module):
    def __init__(
            self,
            dim,
            channel_dim,
    ):
        super().__init__()
        self.LN = nn.LayerNorm(channel_dim)
        self.mapping = nn.Linear(dim, dim, bias=False)

        self.emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )

        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        """
        x_in: [nW*B, window_size*window_size, C]
        return out: [nW*B, window_size*window_size, C]
        """

        x = rearrange(x, 'b hw c -> b c hw')
        x = self.LN(x)
        x = rearrange(x, 'b c hw -> b hw c')
        out_c = self.proj(x)

        return out_c

class SwinSpatialBlock(nn.Module):
    def __init__(self, dim, input_resolution, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.SpatialMLP= SpatialMLP(dim=dim, channel_dim=window_size*window_size)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            # H,  W= self.input_resolution, self.input_resolution
            H = self.input_resolution[0]
            W = self.input_resolution[1]
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            #print(mask_windows.shape)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        # x: [b,h,w,c]
        B, H, W, C = x.shape
        x = self.norm(x)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # MLP

        MLP_windows = self.SpatialMLP(x_windows)

        # merge windows
        MLP_windows = MLP_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(MLP_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        return x


class SpatialAB(nn.Module):
    def __init__(
            self,
            stage,
            dim,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                # SwinTransformerBlock(dim=dim, input_resolution=(256 // (2 ** stage), 256 // (2 ** stage)),
                #                      num_heads=2 ** stage, window_size=8,
                #                      shift_size=0),
                # SwinTransformerBlock(dim=dim, input_resolution=(256 // (2 ** stage), 256 // (2 ** stage)),
                #                      num_heads=2 ** stage, window_size=8,
                #                      shift_size=4),
                SwinSpatialBlock(dim=dim, input_resolution=(256 // (2 ** stage), 256 // (2 ** stage)), window_size=8,
                                 shift_size=0),
                SwinSpatialBlock(dim=dim, input_resolution=(256 // (2 ** stage), 256 // (2 ** stage)), window_size=8,
                                 shift_size=4),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn1, attn2, ff) in self.blocks:
            x = attn1(x) + x
            x = attn2(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class Spatial_block(nn.Module):
    def __init__(self):
        super(Spatial_block, self).__init__()
        # Input projection
        self.embedding = nn.Conv2d(28, 28, 3, 1, 1, bias=False)
        """spatial"""
        # Encoder_spatial
        self.down_0_0 = SpatialAB(stage=0, dim=28, num_blocks=1)
        self.downTranspose_0_0 = nn.Conv2d(28, 56, 4, 2, 1, bias=False)


        self.down_1_0 = SpatialAB(stage=1, dim=56, num_blocks=1)
        self.downTranspose_1_0 = nn.Conv2d(56, 112, 4, 2, 1, bias=False)

        # Bottleneck_spatial
        self.bottleneck_2_0 = SpatialAB(stage=2, dim=112, num_blocks=1)

        # Decoder_spatial
        self.upTranspose_1_1 = nn.ConvTranspose2d(112, 56, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upconv_1_1 = nn.Conv2d(112, 56, 1, 1, bias=False)
        self.up_1_1 = SpatialAB(stage=1, dim=56, num_blocks=1)

        self.upTranspose_0_1 = nn.ConvTranspose2d(56, 28, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upconv_0_1 = nn.Conv2d(56, 28, 1, 1, bias=False)
        self.up_0_1 = SpatialAB(stage=1, dim=28, num_blocks=1)

        self.upTranspose_0_2 = nn.ConvTranspose2d(56, 28, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upconv_0_2 = nn.Conv2d(56, 28, 1, 1, bias=False)
        self.up_0_2 = SpatialAB(stage=1, dim=28, num_blocks=1)
        self.conv_0_2 = nn.Conv2d(56, 28, 1, 1, bias=False)

        # Output projection
        self.mapping = nn.Conv2d(28, 28, 3, 1, 1, bias=False)


        # inter conv
        self.inter_conv1 = nn.Conv2d(112 +28, 112, 1, 1, bias=False)
        self.bottle1 = nn.Conv2d(28 * 7, 28, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, inter_fea=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        fea = self.embedding(x)
        """spatial"""
        # down
        x_0_0 = self.down_0_0(fea)
        fea = self.downTranspose_0_0(x_0_0)

        x_1_0 = self.down_1_0(fea)
        fea = self.downTranspose_1_0(x_1_0)

        #bottleneck
        x_2_0 = self.bottleneck_2_0(fea)
        if inter_fea is not None:
            x_2_0 = self.inter_conv1(torch.cat([x_2_0, inter_fea], dim=1))

        # up
        x_1_1 = self.upTranspose_1_1(x_2_0)
        x_1_1 = self.upconv_1_1(torch.cat([x_1_0, x_1_1], dim=1))
        x_1_1 = self.up_1_1(x_1_1)
        x_0_1 = self.upTranspose_0_2(x_1_1)
        x_0_1 = self.conv_0_2(torch.cat([x_0_1, x_0_0], dim=1))

        # inter fea
        res21 = F.interpolate(x_1_0, scale_factor=0.5, mode='bicubic')
        res32 = F.interpolate(x_0_0, scale_factor=0.5, mode='bicubic')
        res31 = F.interpolate(res32, scale_factor=0.5, mode='bicubic')
        inter_fea = self.bottle1(torch.cat([x_2_0, res21, res31], dim=1))
        # Mapping
        out = self.mapping(x_0_1) + x

        return out[:, :, :h_inp, :w_inp], inter_fea

class spectral_block(nn.Module):
    def __init__(self, dim=28):
        super(spectral_block, self).__init__()
        # Input projection
        self.embedding = nn.Conv2d(28, 28, 3, 1, 1, bias=False)

        """spectral"""
        # Encoder_spectral
        self.down_0_1 = SpectralAB(dim=28, num_blocks=1, dim_head=dim, heads=28 // dim)
        self.downTranspose_0_1 = nn.Conv2d(28, 56, 4, 2, 1, bias=False)

        self.down_1_1 = SpectralAB(dim=56, num_blocks=1, dim_head=dim, heads=56 // dim)
        self.downTranspose_1_1 = nn.Conv2d(56, 112, 4, 2, 1, bias=False)

        # Bottleneck_spectral
        self.bottleneck = SpectralAB(dim=112, dim_head=dim, heads=112 // dim, num_blocks=1)

        # Decoder_spectral
        self.upTranspose_1_2 = nn.ConvTranspose2d(112, 56, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upconv_1_2 = nn.Conv2d(112, 56, 1, 1, bias=False)
        self.up_1_2 = SpectralAB(dim=56, num_blocks=1, dim_head=dim, heads=56 // dim)

        self.upTranspose_0_2 = nn.ConvTranspose2d(56, 28, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upconv_0_2 = nn.Conv2d(56, 28, 1, 1, bias=False)
        self.up_0_2 = SpectralAB(dim=28, num_blocks=1, dim_head=dim, heads=28 // dim)

        # Output projection
        self.mapping = nn.Conv2d(28, 28, 3, 1, 1, bias=False)


        # inter conv
        self.inter_conv1 = nn.Conv2d(56, 28, 1, 1, bias=False)
        self.bottle1 = nn.Conv2d(28 * 7, 28, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, inter_fea=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        fea = self.embedding(x)

        """spectral"""
        # down
        x_0_1 = self.down_0_1(fea)
        fea = self.downTranspose_0_1(x_0_1)

        x_1_1 = self.down_1_1(fea)
        fea = self.downTranspose_1_1(x_1_1)

        #bottleneck
        x_2_1 = self.bottleneck(fea)

        # up
        x_1_2 = self.upTranspose_1_2(x_2_1)
        x_1_2 = self.upconv_1_2(torch.cat([x_1_1, x_1_2], dim=1))
        x_1_2 = self.up_1_2(x_1_2)

        x_0_2 = self.upTranspose_0_2(x_1_2)
        x_0_2 = self.upconv_0_2(torch.cat([x_0_1, x_0_2], dim=1))

        if inter_fea is not None:
            x_0_2 = self.inter_conv1(torch.cat([x_0_2, inter_fea], dim=1))

        # inter fea
        res21 = F.interpolate(x_1_1, scale_factor=0.5, mode='bilinear')
        res32 = F.interpolate(x_0_1, scale_factor=0.5, mode='bilinear')
        res31 = F.interpolate(res32, scale_factor=0.5, mode='bilinear')
        inter_fea = self.bottle1(torch.cat([x_2_1, res21, res31], dim=1))

        # Mapping
        out = self.mapping(x_0_2) + x

        return out[:, :, :h_inp, :w_inp], inter_fea

class CASSI_block(nn.Module):
    def __init__(self, dim=28):
        super(CASSI_block, self).__init__()
        self.cassi = spectral_block(dim = dim)

    def forward(self, x, inter_fea=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out, inter_fea = self.cassi(x, inter_fea=inter_fea)
        return out

class RGB_block(nn.Module):
    def __init__(self, dim=28):
        super(RGB_block, self).__init__()
        self.rgb = spectral_block(dim = dim)

    def forward(self, x, inter_fea=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        out, inter_fea = self.rgb(x, inter_fea=inter_fea)
        return out, inter_fea

def shift(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    inputs = F.pad(inputs,pad=(0,(nC - 1) * step,0,0),mode='constant')
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back(inputs,step=2):
    bs, row, col = inputs.shape
    nC = 28
    inputs = inputs.unsqueeze(1).repeat(1,nC,1,1)
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs[:, :, :, :col - (nC - 1) * step]

def gen_meas_torch(data_batch, mask3d_batch):
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    H = shift_back(meas)
    return H

class NoisePaNet(nn.Module):
    def __init__(self, in_nc=28, out_nc=28, channel=64):
        super(NoisePaNet, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, (3 - 1) // 2, 1, bias=True)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)

        self.fution1 = nn.Conv2d(3, channel, 1, 1, 0, bias=True)

        self.down_sample1 = nn.Conv2d(channel, channel, 3, (3 - 1) // 2, 1, bias=True)
        self.mlp1 = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x_mask, x_rgb):

        x = self.down_sample(self.relu(self.fution(x_mask)))
        x = self.mlp(x)

        x_rgb = self.down_sample1(self.relu1(self.fution1(x_rgb)))
        x_rgb = self.mlp1(x_rgb)

        x = x - x_rgb + 1e-6
        return x

class AMDC(nn.Module):
    def __init__(self, num_iterations=3, n_feat=28):
        super(AMDC, self).__init__()
        self.num_iterations = num_iterations - 1

        # init
        self.fution_init = nn.Conv2d(56, 28, 1, padding=0, bias=True)
        self.fution1 = nn.Conv2d(56, 28, 1, padding=0, bias=True)
        self.fution_rgb = nn.Sequential(
                nn.Conv2d(3, 28, 1, padding=0, bias=True),
                nn.Conv2d(28, 56, 1, padding=0, bias=True),
                nn.Conv2d(56, 28, 1, padding=0, bias=True))
        self.noise_estimator = NoisePaNet()

        # 1stg
        self.conv_in1 = nn.Conv2d(28, 28, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.body1 = CASSI_block(dim=n_feat)
        self.conv_out1 = nn.Conv2d(28, 28, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.spatial1 = SpatialAB(stage=0, dim=28, num_blocks=1)

        self.conv_in2 = nn.Conv2d(28, 28, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.body2 = RGB_block(dim=n_feat)
        self.conv_out2 = nn.Conv2d(28, 28, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.spatial2 = SpatialAB(stage=0, dim=28, num_blocks=1)

        self.fution = nn.Sequential(
                nn.Conv2d(56, 112, 1, padding=0, bias=True),
                nn.Conv2d(112, 112, 1, padding=0, bias=True),
                nn.Conv2d(112, 28, 1, padding=0, bias=True))

        # 2stg
        # self.denoiser_body = nn.ModuleList([])
        # for _ in range(self.num_iterations):
        #     self.denoiser_body.append(
        #         SST_block(dim=28),
        #     )
        self.fution_rgb_body = nn.Conv2d(28+3, 28, 1, padding=0, bias=True)
        self.fution_body = nn.Conv2d(56, 28, 1, padding=0, bias=True)
        self.conv_in_body = nn.Conv2d(28, 28, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.fution_all_body = nn.Conv2d(56, 28, 1, padding=0, bias=True)
        self.body = Spatial_block()
        self.conv_out_body = nn.Conv2d(28, 28, kernel_size=3, padding=(3 - 1) // 2, bias=False)

    def forward(self, y, rgb, input_mask=None):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :param Phi_PhiT: [b,256,310]
        :return: z_crop: [b,28,256,256]
        """

        Phi, Phi_shift = input_mask

        # init
        # start=time.time()
        noise = self.noise_estimator(y, rgb)
        # print('noise: ' + str(time.time()-start))
        # 1stg
        # start = time.time()
        x = y - noise
        x = self.fution_init(torch.cat([x, Phi], dim=1))
        x = self.fution1(torch.cat([x, Phi_shift], dim=1))
        x_rgb = self.fution_rgb(rgb)
        # print('1stg-init: ' + str(time.time() - start))
        # start = time.time()
        x = self.conv_in1(x)
        h = self.body1(x)
        h = self.conv_out1(h)
        h = x + h
        h1 = self.spatial1(h)

        x_rgb = self.conv_in2(x_rgb)
        h_rgb, fea = self.body2(x_rgb)
        h_rgb = self.conv_out2(h_rgb)
        h_rgb = x_rgb + h_rgb
        h_rgb1 = self.spatial2(h_rgb)

        h1 = self.fution(torch.cat([h1, h_rgb1], dim=1))
        h = h1 + h
        # print('1stg: ' + str(time.time() - start))
        # 2stg - 9stg
        for i in range(self.num_iterations):
            # start = time.time()
            input_meas = gen_meas_torch(h, Phi)
            x = y - noise - input_meas

            h_rgb2 = self.fution_rgb_body(torch.cat([x_rgb, rgb], dim=1))
            x = self.fution_body(torch.cat([x, Phi_shift], dim=1))
            x = self.conv_in_body(x)
            h2 = self.fution_all_body(torch.cat([x, h_rgb2], dim=1))
            h2, fea = self.body(h2, fea)
            h2 = self.conv_out_body(h2)
            h2 = h2 + x
            h = h + h2
            # print('2stg: ' + str(time.time() - start))

        return h[:, :, :, 0:256]
