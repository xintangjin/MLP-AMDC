# import os
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
from timm.models.layers import DropPath, to_2tuple

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


class HS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=28,
            heads=8,
            only_local_branch=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.only_local_branch = only_local_branch

        # position embedding
        if only_local_branch:
            seq_l = window_size[0] * window_size[1]
            self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
            trunc_normal_(self.pos_emb)
        else:
            seq_l1 = window_size[0] * window_size[1]
            self.pos_emb1 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l1, seq_l1))
            h,w = 256//self.heads,320//self.heads
            seq_l2 = h*w//seq_l1
            self.pos_emb2 = nn.Parameter(torch.Tensor(1, 1, heads//2, seq_l2, seq_l2))
            trunc_normal_(self.pos_emb1)
            trunc_normal_(self.pos_emb2)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'
        if self.only_local_branch:
            x_inp = rearrange(x, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=w_size[0], b1=w_size[1])
            q = self.to_q(x_inp)
            k, v = self.to_kv(x_inp).chunk(2, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
            q *= self.scale
            sim = einsum('b h i d, b h j d -> b h i j', q, k)
            sim = sim + self.pos_emb
            attn = sim.softmax(dim=-1)
            out = einsum('b h i j, b h j d -> b h i d', attn, v)
            out = rearrange(out, 'b h n d -> b n (h d)')
            out = self.to_out(out)
            out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        else:
            q = self.to_q(x)
            k, v = self.to_kv(x).chunk(2, dim=-1)
            q1, q2 = q[:,:,:,:c//2], q[:,:,:,c//2:]
            k1, k2 = k[:,:,:,:c//2], k[:,:,:,c//2:]
            v1, v2 = v[:,:,:,:c//2], v[:,:,:,c//2:]

            # local branch
            q1, k1, v1 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                              b0=w_size[0], b1=w_size[1]), (q1, k1, v1))
            q1, k1, v1 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q1, k1, v1))
            q1 *= self.scale
            sim1 = einsum('b n h i d, b n h j d -> b n h i j', q1, k1)
            sim1 = sim1 + self.pos_emb1
            attn1 = sim1.softmax(dim=-1)
            out1 = einsum('b n h i j, b n h j d -> b n h i d', attn1, v1)
            out1 = rearrange(out1, 'b n h mm d -> b n mm (h d)')

            # non-local branch
            q2, k2, v2 = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (b0 b1) c',
                                                 b0=w_size[0], b1=w_size[1]), (q2, k2, v2))
            q2, k2, v2 = map(lambda t: t.permute(0, 2, 1, 3), (q2.clone(), k2.clone(), v2.clone()))
            q2, k2, v2 = map(lambda t: rearrange(t, 'b n mm (h d) -> b n h mm d', h=self.heads//2), (q2, k2, v2))
            q2 *= self.scale
            sim2 = einsum('b n h i d, b n h j d -> b n h i j', q2, k2)
            sim2 = sim2 + self.pos_emb2
            attn2 = sim2.softmax(dim=-1)
            out2 = einsum('b n h i j, b n h j d -> b n h i d', attn2, v2)
            out2 = rearrange(out2, 'b n h mm d -> b n mm (h d)')
            out2 = out2.permute(0, 2, 1, 3)

            out = torch.cat([out1,out2],dim=-1).contiguous()
            out = self.to_out(out)
            out = rearrange(out, 'b (h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        return out


class HSAB(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, HS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, only_local_branch=(heads==1))),
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


class HST(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, num_blocks=[1,1,1]):
        super(HST, self).__init__()
        self.dim = dim
        self.scales = len(num_blocks)

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim
        for i in range(self.scales-1):
            self.encoder_layers.append(nn.ModuleList([
                HSAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
            ]))
            dim_scale *= 2

        # Bottleneck
        self.bottleneck = HSAB(dim=dim_scale, dim_head=dim, heads=dim_scale // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_scale, dim_scale // 2, 1, 1, bias=False),
                HSAB(dim=dim_scale // 2, num_blocks=num_blocks[self.scales - 2 - i], dim_head=dim,
                     heads=(dim_scale // 2) // dim),
            ]))
            dim_scale //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # Embedding
        fea = self.embedding(x)
        x = x[:,:28,:,:]

        # Encoder
        fea_encoder = []
        for (HSAB, FeaDownSample) in self.encoder_layers:
            fea = HSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, HSAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.scales-2-i]], dim=1))
            fea = HSAB(fea)

        # Mapping
        out = self.mapping(fea) + x
        return out[:, :, :h_inp, :w_inp]


def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y


def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x


def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs


def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs


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
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
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
                #SwinTransformerBlock(dim=dim, input_resolution=256 // (2 ** stage), num_heads=heads, window_size=256 //(2 ** stage), shift_size=0),
                SwinTransformerBlock(dim=dim, input_resolution=(256 // (2 ** stage), 320 // (2 ** stage)),
                                     num_heads=2 ** stage, window_size=8,
                                     shift_size=0),
                SwinTransformerBlock(dim=dim, input_resolution=(256 // (2 ** stage), 320 // (2 ** stage)),
                                     num_heads=2 ** stage, window_size=8,
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


class SST_block(nn.Module):
    def __init__(self, dim=31):
        super(SST_block, self).__init__()
        self.dim = dim
        # Input projection
        self.embedding = nn.Conv2d(dim+1, dim, 3, 1, 1, bias=False)

        """spatial"""
        # Encoder_spatial
        self.down_0_0 = SpatialAB(stage=0, dim=dim, num_blocks=1)
        self.downTranspose_0_0 = nn.Conv2d(dim, dim*2, 4, 2, 1, bias=False)


        self.down_1_0 = SpatialAB(stage=1, dim=dim*2, num_blocks=1)
        self.downTranspose_1_0 = nn.Conv2d(dim*2, dim*4, 4, 2, 1, bias=False)

        # Bottleneck_spatial
        self.bottleneck_2_0 = SpatialAB(stage=2, dim=dim*4, num_blocks=1)

        # Decoder_spatial
        self.upTranspose_1_1 = nn.ConvTranspose2d(dim*4, dim*2, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upconv_1_1 = nn.Conv2d(dim*4, dim*2, 1, 1, bias=False)
        self.up_1_1 = SpatialAB(stage=1, dim=dim*2, num_blocks=1)

        self.upTranspose_0_1 = nn.ConvTranspose2d(dim*2, dim, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upconv_0_1 = nn.Conv2d(dim*2, dim, 1, 1, bias=False)
        self.up_0_1 = SpatialAB(stage=0, dim=dim, num_blocks=1)

        self.upTranspose_0_2 = nn.ConvTranspose2d(dim*2, dim, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upconv_0_2 = nn.Conv2d(dim*2, dim, 1, 1, bias=False)
        self.up_0_2 = SpatialAB(stage=0, dim=dim, num_blocks=1)
        self.conv_0_2 = nn.Conv2d(dim*2, dim, 1, 1, bias=False)


        """spectral"""
        # Encoder_spectral
        self.down_0_3 = SpectralAB(dim=dim, num_blocks=1, dim_head=dim, heads=1)
        self.downTranspose_0_3 = nn.Conv2d(dim, dim*2, 4, 2, 1, bias=False)

        self.down_1_2 = SpectralAB(dim=dim*2, num_blocks=1, dim_head=dim, heads=2)
        self.downconv_1_2 = nn.Conv2d(dim*4, dim*2, 1, 1, bias=False)
        self.downTranspose_1_2 = nn.Conv2d(dim*2, dim*4, 4, 2, 1, bias=False)
        self.conv_1_2 = nn.Conv2d(dim*4, dim*2, 1, 1, bias=False)


        # Bottleneck_spectral
        self.bottleneck_2_2 = SpectralAB(dim=dim*4, dim_head=dim, heads=4, num_blocks=1)
        self.conv_2_2 = nn.Conv2d(dim*8, dim*4, 1, 1, bias=False)

        # Decoder_spectral
        self.upTranspose_1_3 = nn.ConvTranspose2d(dim*4, dim*2, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upconv_1_3 = nn.Conv2d(dim*4, dim*2, 1, 1, bias=False)
        self.up_1_3 = SpectralAB(dim=dim*2, num_blocks=1, dim_head=dim, heads=2)

        self.upTranspose_0_4 = nn.ConvTranspose2d(dim*2, dim, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upconv_0_4 = nn.Conv2d(dim*2, dim, 1, 1, bias=False)
        self.up_0_4 = SpectralAB(dim=dim, num_blocks=1, dim_head=dim, heads=1)
        self.conv_0_4 = nn.Conv2d(dim*2, dim, 1, 1, bias=False)

        self.upTranspose_0_5 = nn.ConvTranspose2d(dim*2, dim, stride=2, kernel_size=2, padding=0, output_padding=0)
        self.upconv_0_5 = nn.Conv2d(dim*2, dim, 1, 1, bias=False)
        self.up_0_5 = SpectralAB(dim=dim, num_blocks=1, dim_head=dim, heads=1)
        self.conv_0_5 = nn.Conv2d(dim*3, dim, 1, 1, bias=False)


        # Output projection
        self.mapping = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)


        # inter conv
        self.inter_conv1 = nn.Conv2d(dim*2, dim, 1, 1, bias=False)
        self.inter_conv2 = nn.Conv2d(dim*2, dim, 1, 1, bias=False)
        self.bottle1 = nn.Conv2d(dim * 7, dim, 1, 1, bias=False)
        self.bottle2 = nn.Conv2d(dim * 7, dim, 1, 1, bias=False)


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

    def forward(self, x, inter_fea1=None, inter_fea2=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 32, 32
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        fea = self.embedding(x)
        x = x[:,:self.dim,:,:]
        """spatial"""
        # down
        x_0_0 = self.down_0_0(fea)
        fea = self.downTranspose_0_0(x_0_0)

        x_1_0 = self.down_1_0(fea)
        fea = self.downTranspose_1_0(x_1_0)

        #bottleneck
        x_2_0 = self.bottleneck_2_0(fea)

        # up
        x_1_1 = self.upTranspose_1_1(x_2_0)
        x_1_1 = self.upconv_1_1(torch.cat([x_1_0, x_1_1], dim=1))
        x_1_1 = self.up_1_1(x_1_1)

        x_0_1 = self.upTranspose_0_1(x_1_0)
        x_0_1 = self.upconv_0_1(torch.cat([x_0_1, x_0_0], dim=1))
        x_0_1 = self.up_0_1(x_0_1)

        if inter_fea1 is not None:
            x_0_1 = self.inter_conv1(torch.cat([x_0_1, inter_fea1], dim=1))

        x_0_2 = self.upTranspose_0_2(x_1_1)
        x_0_2 = self.upconv_0_2(torch.cat([x_0_2, x_0_1], dim=1))
        x_0_2 = self.up_0_2(x_0_2)
        x_0_2 = self.conv_0_2(torch.cat([x_0_2, x_0_0], dim=1))

        """spectral"""
        # down
        x_0_3 = self.down_0_3(x_0_2)
        fea = self.downTranspose_0_3(x_0_3)

        x_1_2 = self.downconv_1_2(torch.cat([fea, x_1_1], dim=1))
        x_1_2 = self.down_1_2(x_1_2)
        x_1_2 = self.conv_1_2(torch.cat([x_1_2, x_1_0], dim=1))
        fea = self.downTranspose_1_2(x_1_2)

        #bottleneck
        x_2_2 = self.bottleneck_2_2(fea)
        x_2_2 = self.conv_2_2(torch.cat([x_2_2, x_2_0], dim=1))

        # up
        x_1_3 = self.upTranspose_1_3(x_2_2)
        x_1_3 = self.upconv_1_3(torch.cat([x_1_3, x_1_2], dim=1))
        x_1_3 = self.up_1_3(x_1_3)

        x_0_4 = self.upTranspose_0_4(x_1_2)
        x_0_4 = self.upconv_0_4(torch.cat([x_0_4, x_0_3], dim=1))
        x_0_4 = self.up_0_4(x_0_4)
        x_0_4 = self.conv_0_4(torch.cat([x_0_4, x_0_1], dim=1))

        if inter_fea2 is not None:
            x_0_4 = self.inter_conv1(torch.cat([x_0_4, inter_fea2], dim=1))

        x_0_5 = self.upTranspose_0_5(x_1_3)
        x_0_5 = self.upconv_0_5(torch.cat([x_0_5, x_0_4], dim=1))
        x_0_5 = self.up_0_5(x_0_5)
        x_0_5 = self.conv_0_5(torch.cat([x_0_5, x_0_3, x_0_0], dim=1))


        # inter fea
        res21 = F.interpolate(x_1_0, scale_factor=2, mode='bilinear')
        res32 = F.interpolate(x_2_0, scale_factor=2, mode='bilinear')
        res31 = F.interpolate(res32, scale_factor=2, mode='bilinear')
        inter_fea1 = self.bottle1(torch.cat([x_0_1, res21, res31], dim=1))

        res21 = F.interpolate(x_1_2, scale_factor=2, mode='bilinear')
        res32 = F.interpolate(x_2_2, scale_factor=2, mode='bilinear')
        res31 = F.interpolate(res32, scale_factor=2, mode='bilinear')
        inter_fea2 = self.bottle1(torch.cat([x_0_4, res21, res31], dim=1))

        # Mapping
        out = self.mapping(x_0_5) + x

        return out[:, :, :h_inp, :w_inp], inter_fea1, inter_fea2

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# stage = 0
# a = SST_block().to(device)
# b = torch.rand(1, 28, 256, 320).to(device)
# c = a.forward(b)
# print(c.shape)

class HyPaNet(nn.Module):
    def __init__(self, in_nc=31, out_nc=3, channel=64):
        super(HyPaNet, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):
        x = self.down_sample(self.relu(self.fution(x)))
        x = self.avg_pool(x)
        x = self.mlp(x) + 1e-6

        return x[:,0,:,:], x[:,1:2,:,:], x[:,2:,:,:]



class create_alpha_beta(nn.Module):
    def __init__(self):
        super(create_alpha_beta, self).__init__()
        self.nC = 31
        self.step = 2
        self.para_estimator = HyPaNet(in_nc=64)
        self.fution = nn.Conv2d(31*2, 64, 1, 1, 0, bias=True)

    def forward(self,  y, Phi):
        """
        :param y: [b,28, 256,310]
        :param Phi: [b,28,256,310]
        :return: temp: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        y = y / self.nC * 2
        alpha, beta, lamb = self.para_estimator(self.fution(torch.cat([y, Phi], dim=1)))
        return alpha, beta, lamb



class OLU(nn.Module):

    def __init__(self, num_iterations=3, channel=31):
        super(OLU, self).__init__()
        self.para_estimator = HyPaNet(in_nc=channel)
        self.fution = nn.Conv2d(channel*2, channel, 1, padding=0, bias=True)
        self.num_iterations = num_iterations - 2
        self.denoiser_init = SST_block(dim=channel)

        # self.denoiser_body = nn.ModuleList([])
        # for _ in range(self.num_iterations):
        #     self.denoiser_body.append(
        #         SST_block(dim=28),
        #     )

        self.denoiser_body = SST_block(dim=channel)

        self.denoiser_end = SST_block(dim=channel)

        self.create_alpha_beta = nn.ModuleList([])
        for _ in range(self.num_iterations):
            self.create_alpha_beta.append(
                create_alpha_beta(),
            )


    def initial(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: temp: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        _, nC,_,_= Phi.shape
        step = 2
        y = y / nC * 2
        bs,row,col = y.shape
        y_shift = torch.zeros(bs, nC, row, col).cuda().float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fution(torch.cat([y_shift, Phi], dim=1))
        alpha, beta, lamb = self.para_estimator(self.fution(torch.cat([y_shift, Phi], dim=1)))

        return z, alpha, beta

    def forward(self, y, input_mask=None):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :param Phi_PhiT: [b,256,310]
        :return: z_crop: [b,28,256,256]
        """
        #[2,31,256,316] [2,256,316]
        Phi, Phi_shift = input_mask
        # 1stg
        z, alpha, beta = self.initial(y, Phi)
        Phi_z = A(z, Phi)
        x = z + At(torch.div(y - Phi_z, alpha + Phi_shift), Phi)
        z_last = z
        x = shift_back_3d(x)
        beta_repeat = beta.repeat(1, 1, x.shape[2], x.shape[3])
        z, inter_fea1, inter_fea2 = self.denoiser_init(torch.cat([x, beta_repeat], dim=1))
        z = shift_3d(z)

        for i in range(self.num_iterations):
            alpha, beta, lamb = self.create_alpha_beta[i](z, Phi)
            Phi_z = A(z, Phi)
            x = z + At(torch.div(y-Phi_z,alpha+Phi_shift), Phi)
            lamb_repeat = lamb.repeat(1, 1, z.shape[2], z.shape[3])
            x = x + lamb_repeat * (z - z_last)
            z_last = z
            x = shift_back_3d(x)
            beta_repeat = beta.repeat(1,1,x.shape[2], x.shape[3])
            z, inter_fea1, inter_fea2 = self.denoiser_body(torch.cat([x, beta_repeat],dim=1), inter_fea1, inter_fea2)
            z = shift_3d(z)


        alpha, beta, lamb = self.create_alpha_beta[i](z, Phi)
        Phi_z = A(z, Phi)
        x = z + At(torch.div(y-Phi_z,alpha+Phi_shift), Phi)
        lamb_repeat = lamb.repeat(1, 1, z.shape[2], z.shape[3])
        x = x + lamb_repeat * (z - z_last)
        x = shift_back_3d(x)
        beta_repeat = beta.repeat(1,1,x.shape[2], x.shape[3])
        z, inter_fea1, inter_fea2 = self.denoiser_end(torch.cat([x, beta_repeat], dim=1), inter_fea1, inter_fea2)


        return z[:, :, :, 0:256]
