import functools
import math
from operator import mul
from typing import List, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange
from mmdet3d.models.builder import HEADS
from torch import nn
from inspect import isfunction
from .vanilla_attention import Attend
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
# helper functions
def sigmoid_xent_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    return F.binary_cross_entropy_with_logits(inputs, targets, reduction=reduction)


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "mean",
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class BEVGridTransform(nn.Module):
    def __init__(
            self,
            *,
            input_scope: List[Tuple[float, float, float]],
            output_scope: List[Tuple[float, float, float]],
            prescale_factor: float = 1,
    ) -> None:
        super().__init__()
        self.input_scope = input_scope
        self.output_scope = output_scope
        self.prescale_factor = prescale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prescale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.prescale_factor,
                mode="bilinear",
                align_corners=False,
            )

        coords = []
        for (imin, imax, _), (omin, omax, ostep) in zip(
                self.input_scope, self.output_scope
        ):
            v = torch.arange(omin + ostep / 2, omax, ostep)
            v = (v - imin) / (imax - imin) * 2 - 1
            coords.append(v.to(x.device))

        u, v = torch.meshgrid(coords, indexing="ij")
        grid = torch.stack([v, u], dim=-1)
        grid = torch.stack([grid] * x.shape[0], dim=0)

        x = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return x


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def mul_reduce(tup):
    return functools.reduce(mul, tup)


def divisible_by(numer, denom):
    return (numer % denom) == 0


mlist = nn.ModuleList


# for time conditioning

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.theta = theta
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        assert dtype == torch.float, 'input to sinusoidal pos emb must be a float type'

        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1).type(dtype)


# layernorm 3d

class RMSNorm(nn.Module):  # layer normalization 3D
    def __init__(self, chan, dim=1):
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(chan))

    def forward(self, x):
        dim = self.dim
        right_ones = (dim + 1) if dim < 0 else (x.ndim - 1 - dim)
        gamma = self.gamma.reshape(-1, *((1,) * right_ones))
        return F.normalize(x, dim=dim) * (x.shape[dim] ** 0.5) * gamma


# feedforward

def shift_token(t):
    t, t_shift = t.chunk(2, dim=1)
    t_shift = F.pad(t_shift, (0, 0, 0, 0, 1, -1), value=0.)
    return torch.cat((t, t_shift), dim=1)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=1)  # split the tensor into two parts along the second dimension
        return x * F.gelu(gate)  # apply the gelu function to the gate and multiply it with the x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()

        inner_dim = int(dim * mult * 2 / 3)
        self.proj_in = nn.Sequential(
            nn.Conv3d(dim, inner_dim * 2, 1, bias=False),
            GEGLU()
        )

        self.proj_out = nn.Sequential(
            RMSNorm(inner_dim),
            nn.Conv3d(inner_dim, dim, 1, bias=False)
        )
        # equals to nn.Linear(inner_dim, dim) for each position in 3D features

    def forward(self, x, enable_time=True):

        is_video = x.ndim == 5
        enable_time &= is_video  # enable_time = enable_time and is_video

        if not is_video:
            x = rearrange(x, 'b c h w -> b c 1 h w')

        x = self.proj_in(x)

        if enable_time:
            x = shift_token(x)

        out = self.proj_out(x)

        if not is_video:
            out = rearrange(out, 'b c 1 h w -> b c h w')

        return out


# best relative positional encoding

class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
            self,
            *,
            dim,
            heads,
            num_dims=1,
            layers=2
    ):
        super().__init__()
        self.num_dims = num_dims

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), nn.SiLU()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), nn.SiLU()))

        self.net.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, *dimensions):
        device = self.device

        shape = torch.tensor(dimensions, device=device)
        rel_pos_shape = 2 * shape - 1

        # calculate strides

        strides = torch.flip(rel_pos_shape, (0,)).cumprod(dim=-1)
        strides = torch.flip(F.pad(strides, (1, -1), value=1), (0,))

        # get all positions and calculate all the relative distances

        positions = [torch.arange(d, device=device) for d in dimensions]
        grid = torch.stack(torch.meshgrid(*positions, indexing='ij'), dim=-1)
        grid = rearrange(grid, '... c -> (...) c')
        rel_dist = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

        # get all relative positions across all dimensions

        rel_positions = [torch.arange(-d + 1, d, device=device) for d in dimensions]
        rel_pos_grid = torch.stack(torch.meshgrid(*rel_positions, indexing='ij'), dim=-1)
        rel_pos_grid = rearrange(rel_pos_grid, '... c -> (...) c')

        # mlp input

        bias = rel_pos_grid.float()

        for layer in self.net:
            bias = layer(bias)

        # convert relative distances to indices of the bias

        rel_dist += (shape - 1)  # make sure all positive
        rel_dist *= strides
        rel_dist_indices = rel_dist.sum(dim=-1)

        # now select the bias for each unique relative position combination

        bias = bias[rel_dist_indices]
        return rearrange(bias, 'i j h -> h i j')


# helper classes

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            cond_dim=None,
            dim_head=64,
            heads=8,
            flash=False,
            causal=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.attend = Attend(flash=flash, causal=causal)

        self.norm = RMSNorm(dim, dim=-1)
        cond_dim = default(cond_dim, dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(cond_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        nn.init.zeros_(self.to_out.weight.data)  # identity with skip connection

    def forward(
            self,
            x,
            cond=None,
            rel_pos_bias=None
    ):
        x = self.norm(x)  # normalize along the last dimension
        cond = default(cond,x)
        q, k, v = self.to_q(x), *self.to_kv(cond).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        # (batch,sequence,(heads*dim_head)) -> (batch,heads,sequence,dim_head)

        out = self.attend(q, k, v, bias=rel_pos_bias)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# main contribution - pseudo 3d conv

class PseudoConv3d(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            kernel_size=3,
            *,  # * means that the following parameters are keyword-only
            temporal_kernel_size=None,
            **kwargs
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        temporal_kernel_size = default(temporal_kernel_size, kernel_size)
        # if temporal_kernel_size is not specified, it is the same as kernel_size

        self.spatial_conv = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2)
        # padding == kernel_size // 2 is to make sure the output has the same size as the input
        self.temporal_conv = nn.Conv1d(dim_out, dim_out, kernel_size=temporal_kernel_size,
                                       padding=temporal_kernel_size // 2) if kernel_size > 1 else None

        if exists(self.temporal_conv):
            nn.init.dirac_(self.temporal_conv.weight.data)  # initialized to be identity
            nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(
            self,
            x,
            enable_time=True
    ):
        b, c, *_, h, w = x.shape

        is_video = x.ndim == 5
        enable_time &= is_video

        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) c h w')
            # in this case, spatial_conv will be applied to each frame of the video

        x = self.spatial_conv(x)

        if is_video:
            x = rearrange(x, '(b f) c h w -> b c f h w', b=b)

        if not enable_time or not exists(self.temporal_conv) or x.shape[2] == 1:
            # if we only have one frame, or we don't want to set enable_time,then we don't need to apply the temporal conv
            return x

        x = rearrange(x, 'b c f h w -> (b h w) c f')
        # conv1D that is applied to each position of the 3D feature(across the frames)

        x = self.temporal_conv(x)

        x = rearrange(x, '(b h w) c f -> b c f h w', h=h, w=w)

        return x


# factorized spatial temporal attention from Ho et al.


class SpatioTemporalAttention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            dim_head=64,
            heads=8,
            add_feed_forward=True,
            ff_mult=4,
            pos_bias=True,
            flash=False,
            causal_time_attn=False,
            cross_attention=False,
            cond_dim=None,
    ):
        super().__init__()
        assert not (flash and pos_bias), 'learned positional attention bias is not compatible with flash attention'

        self.spatial_attn = Attention(dim=dim, dim_head=dim_head, heads=heads, flash=flash)
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim // 2, heads=heads, num_dims=2) if pos_bias else None

        self.temporal_attn = Attention(dim=dim, dim_head=dim_head, heads=heads, flash=flash, causal=causal_time_attn)
        self.temporal_rel_pos_bias = ContinuousPositionBias(dim=dim // 2, heads=heads, num_dims=1) if pos_bias else None
        self.cross_attention = cross_attention
        if self.cross_attention:
            self.cross_attn = Attention(dim=dim, cond_dim=cond_dim,dim_head=dim_head, heads=heads, flash=flash)
        self.has_feed_forward = add_feed_forward
        if not add_feed_forward:
            return

        self.ff = FeedForward(dim=dim, mult=ff_mult)  # ff_mult defines the inner dimension of the feedforward network

    def forward(
            self,
            x,
            cond=None,
            enable_time=True
    ):
        b, c, *_, h, w = x.shape
        is_video = x.ndim == 5
        enable_time &= is_video

        if is_video:
            x = rearrange(x, 'b c f h w -> (b f) (h w) c')
            # first, for spatial attention, f will be treated as the batch dimension, h*w=num_query/key
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')

        space_rel_pos_bias = self.spatial_rel_pos_bias(h, w) if exists(self.spatial_rel_pos_bias) else None

        x = self.spatial_attn(x=x, cond=None,rel_pos_bias=space_rel_pos_bias) + x

        if is_video:
            x = rearrange(x, '(b f) (h w) c -> b c f h w', b=b, h=h, w=w)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if enable_time:
            x = rearrange(x, 'b c f h w -> (b h w) f c')  # for each position, attention on the frame axis

            time_rel_pos_bias = self.temporal_rel_pos_bias(x.shape[1]) if exists(self.temporal_rel_pos_bias) else None

            x = self.temporal_attn(x=x, cond=None,rel_pos_bias=time_rel_pos_bias) + x

            x = rearrange(x, '(b h w) f c -> b c f h w', w=w, h=h)
        if self.cross_attention:
            assert isinstance(cond, torch.Tensor), 'cross attention requires a condition'
            if is_video:
                x = rearrange(x, 'b c f h w -> (b f) (h w) c')
                # first, for spatial attention, f will be treated as the batch dimension, h*w=num_query/key
            else:
                x = rearrange(x, 'b c h w -> b (h w) c')
            cond_resize = rearrange(cond, 'b c h w -> b (h w) c')
            f = int(x.shape[0]/cond_resize.shape[0])
            # reshape cond to match the shape of x
            cond_resize = repeat(cond_resize, 'b n c -> (b f) n c', f=f)
            x = self.cross_attn(x=x, cond=cond_resize,rel_pos_bias=None) + x
            if is_video:
                x = rearrange(x, '(b f) (h w) c -> b c f h w', b=b, h=h, w=w)
            else:
                x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        if self.has_feed_forward:
            x = self.ff(x, enable_time=enable_time) + x

        return x


# resnet block

class Block(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            kernel_size=3,
            temporal_kernel_size=None,
            groups=8
    ):
        super().__init__()
        self.project = PseudoConv3d(dim, dim_out, 3)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(
            self,
            x,
            scale_shift=None,
            enable_time=False
    ):
        x = self.project(x, enable_time=enable_time)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift  # time embedding for scale and shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            *,
            timestep_cond_dim=None,
            groups=8
    ):
        super().__init__()

        self.timestep_mlp = None

        if exists(timestep_cond_dim):
            self.timestep_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(timestep_cond_dim, dim_out * 2)
            )

        self.block1 = Block(dim, dim_out, groups=groups)  # mlp + group normalization + SiLU
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = PseudoConv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        # if dimension is the same, then the input is added to the output
        # conv 2d in each frame and conv1d between ecah frame

    def forward(
            self,
            x,
            timestep_emb=None,
            enable_time=True
    ):
        assert not (exists(timestep_emb) ^ exists(self.timestep_mlp))

        scale_shift = None

        if exists(self.timestep_mlp) and exists(timestep_emb):
            time_emb = self.timestep_mlp(timestep_emb)  # for each batch, timesteps are the same
            to_einsum_eq = 'b c 1 1 1' if x.ndim == 5 else 'b c 1 1'
            time_emb = rearrange(time_emb, f'b c -> {to_einsum_eq}')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift, enable_time=enable_time)

        h = self.block2(h, enable_time=enable_time)

        return h + self.res_conv(x)


# pixelshuffle upsamples and downsamples
# where time dimension can be configured

class Downsample(nn.Module):
    def __init__(
            self,
            dim,
            downsample_space=True,
            downsample_time=False,
            nonlin=False
    ):
        super().__init__()
        assert downsample_space or downsample_time

        self.down_space = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
            nn.Conv2d(dim * 4, dim, 1, bias=False),
            nn.SiLU() if nonlin else nn.Identity()
        ) if downsample_space else None

        self.down_time = nn.Sequential(
            Rearrange('b c (f p) h w -> b (c p) f h w', p=2),
            nn.Conv3d(dim * 2, dim, 1, bias=False),
            nn.SiLU() if nonlin else nn.Identity()
        ) if downsample_time else None

    def forward(
            self,
            x,
            enable_time=True
    ):
        is_video = x.ndim == 5

        if is_video:
            x = rearrange(x, 'b c f h w -> b f c h w')
            x, ps = pack([x], '* c h w')  # x to bf, c, h, w?

        if exists(self.down_space):
            x = self.down_space(x)

        if is_video:
            x, = unpack(x, ps, '* c h w')
            x = rearrange(x, 'b f c h w -> b c f h w')

        if not is_video or not exists(self.down_time) or not enable_time:
            return x

        x = self.down_time(x)

        return x


class Upsample(nn.Module):
    def __init__(
            self,
            dim,
            upsample_space=True,
            upsample_time=False,
            nonlin=False
    ):
        super().__init__()
        assert upsample_space or upsample_time

        self.up_space = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.SiLU() if nonlin else nn.Identity(),
            Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)', p1=2, p2=2)
        ) if upsample_space else None

        self.up_time = nn.Sequential(
            nn.Conv3d(dim, dim * 2, 1),
            nn.SiLU() if nonlin else nn.Identity(),
            Rearrange('b (c p) f h w -> b c (f p) h w', p=2)
        ) if upsample_time else None

        self.init_()

    def init_(self):
        if exists(self.up_space):
            self.init_conv_(self.up_space[0], 4)

        if exists(self.up_time):
            self.init_conv_(self.up_time[0], 2)

    def init_conv_(self, conv, factor):
        o, *remain_dims = conv.weight.shape
        conv_weight = torch.empty(o // factor, *remain_dims)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r=factor)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(
            self,
            x,
            enable_time=True
    ):
        is_video = x.ndim == 5

        if is_video:
            x = rearrange(x, 'b c f h w -> b f c h w')
            x, ps = pack([x], '* c h w')

        if exists(self.up_space):
            x = self.up_space(x)

        if is_video:
            x, = unpack(x, ps, '* c h w')
            x = rearrange(x, 'b f c h w -> b c f h w')

        if not is_video or not exists(self.up_time) or not enable_time:
            return x

        x = self.up_time(x)

        return x


# space time factorized 3d unet
@HEADS.register_module()
class SpaceTimeUnet(nn.Module):
    def __init__(
            self,
            dim,
            classes,
            loss,
            grid_transform,
            channels=3,
            dim_mult=(1, 2, 4, 8),
            self_attns=(True, True, True, True),
            temporal_compression=(False, True, True, True),
            resnet_block_depths=(2, 2, 2, 2),
            attn_dim_head=64,
            attn_heads=8,
            condition_on_timestep=True,
            attn_pos_bias=True,
            flash_attn=False,
            causal_time_attn=False,
            receptive_field=1,
            future_frames=2,
            cross_attention=True,
            cond_dim=None,
            enable_time=False,
            **kwargs,
    ):
        super().__init__()
        assert len(dim_mult) == len(self_attns) == len(temporal_compression) == len(resnet_block_depths)
        self.seq_length = receptive_field + future_frames
        self.classes = classes
        self.loss = loss
        self.transform = BEVGridTransform(**grid_transform)
        self.enable_time = enable_time

        num_layers = len(dim_mult)  # number of layer in the unet (half)

        dims = [dim, *map(lambda mult: mult * dim, dim_mult)]  # [64, 64, 128, 256, 512]
        dim_in_out = zip(dims[:-1], dims[1:])  # （64, 64）, （64, 128）, （128, 256）, （256, 512）
        # determine the valid multiples of the image size and frames of the video

        self.frame_multiple = 2 ** sum(
            tuple(map(int, temporal_compression)))  # 2 ** 3 = 8， 3 is the number of True in temporal_compression
        self.image_size_multiple = 2 ** num_layers
        # timestep conditioning for DDPM, not to be confused with the time dimension of the video

        self.to_timestep_cond = None
        timestep_cond_dim = (dim * 4) if condition_on_timestep else None

        if condition_on_timestep:
            self.to_timestep_cond = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, timestep_cond_dim),
                nn.SiLU()
            )

        # layers

        self.downs = mlist([])
        self.ups = mlist([])

        attn_kwargs = dict(
            dim_head=attn_dim_head,
            heads=attn_heads,
            pos_bias=attn_pos_bias,
            flash=flash_attn,
            causal_time_attn=causal_time_attn,
            cross_attention=cross_attention,
            cond_dim=cond_dim,
        )

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, timestep_cond_dim=timestep_cond_dim)
        # timestep_cond_dim will be mapped to dim*2 and add to features as a position bias
        # inside ResNetBlock, features will be processed by PseudoConv3D, then group normalization and SiLU
        self.mid_attn = SpatioTemporalAttention(dim=mid_dim, **attn_kwargs)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, timestep_cond_dim=timestep_cond_dim)

        for _, self_attend, (dim_in, dim_out), compress_time, resnet_block_depth in zip(range(num_layers), self_attns,
                                                                                        dim_in_out,
                                                                                        temporal_compression,
                                                                                        resnet_block_depths):
            assert resnet_block_depth >= 1

            self.downs.append(mlist([
                ResnetBlock(dim_in, dim_out, timestep_cond_dim=timestep_cond_dim),
                mlist([ResnetBlock(dim_out, dim_out) for _ in range(resnet_block_depth)]),
                SpatioTemporalAttention(dim=dim_out, **attn_kwargs) if self_attend else None,
                Downsample(dim_out, downsample_time=compress_time)
            ]))

            self.ups.append(mlist([
                ResnetBlock(dim_out * 2, dim_in, timestep_cond_dim=timestep_cond_dim),  # *2 because of skip connection
                mlist(
                    [ResnetBlock(dim_in + (dim_out if ind == 0 else 0), dim_in) for ind in range(resnet_block_depth)]),
                # _ dim_out if ind == 0 else 0 is to add the skip connection only to the first block
                SpatioTemporalAttention(dim=dim_in, **attn_kwargs) if self_attend else None,
                Upsample(dim_out, upsample_time=compress_time)

            ]))

        self.skip_scale = 2 ** -0.5  # paper shows faster convergence

        self.conv_in = PseudoConv3d(dim=channels, dim_out=dim, kernel_size=7, temporal_kernel_size=3)
        # from 512 to 64
        self.conv_out = PseudoConv3d(dim=dim, dim_out=len(classes), kernel_size=3, temporal_kernel_size=3)
        #64 to 7

    def forward(
            self,
            x,
            cond,
            timestep=None,
            embed_timesteps=None,
            target=None,
    ):
        """

        Args:
            x: noise gt with shape (bs, channels, frames, bev_h, bev_w)
            cond: conditional bev features with shape (bs, channels, h, w), h,w is smaller than bev_h, bev_w
            timestep: embedding of the timestep with shape (bs, timestep_dim)
            target: ground truth bev segmentation map with shape (bs,num_classes,frames,bev_h,bev_w)

        Returns:

        """
        # TODO: where to put the condition text?
        # some asserts
        is_video = x.ndim == 5
        # resize_x = []
        # for frame_idx in range(x.size(2)):  # Iterate over the 'frames' dimension
        #     slice = x[:, :, frame_idx, :, :]
        #     processed_slice = self.transform(slice)
        #     resize_x.append(processed_slice)
        # # Combine the processed frames back into a tensor
        # x = torch.stack(resize_x, dim=2)  # (bs, channels, frames, h, w)

        if self.enable_time and is_video:
            frames = x.shape[2]
            assert divisible_by(frames,
                                self.frame_multiple), f'number of frames on the video ({frames}) must be divisible by the frame multiple ({self.frame_multiple})'

        height, width = x.shape[-2:]
        assert divisible_by(height, self.image_size_multiple) and divisible_by(width,
                                                                               self.image_size_multiple), f'height and width of the image or video must be a multiple of {self.image_size_multiple}'

        # main logic
        if isinstance(embed_timesteps, torch.Tensor):
            t = embed_timesteps
        else:
            t = self.to_timestep_cond(rearrange(timestep, '... -> (...)')) if exists(timestep) else None
        exapnd_cond = cond.unsqueeze(2).expand(-1, -1, self.seq_length, -1, -1)
        x = torch.cat((x, exapnd_cond), dim=1)
        assert self.seq_length== x.size(2)
        #before feed in self.conv_in, the shape of x is torch.Size([1, 512, 3, 128, 128])
        x = self.conv_in(x, enable_time=self.enable_time)  # including a 2D in each frame and 1D between each frame
        # after conv_in the shape of x is torch.Size([1, 64, 3, 128, 128])
        # x = rearrange(x, 'b c f h w -> b (c f) h w')
        # x = self.transform(x)
        # x = rearrange(x, 'b (c f) h w -> b c f h w', f=self.seq_length)
        hiddens = []

        for init_block, blocks, maybe_attention, downsample in self.downs:
            x = init_block(x, t, enable_time=self.enable_time)

            hiddens.append(x.clone())  # first resnet block

            for block in blocks:
                x = block(x, enable_time=self.enable_time)  # the rest of the resnet blocks

            if exists(maybe_attention):
                x = maybe_attention(x, cond=cond,enable_time=self.enable_time)  # maybe attention

            hiddens.append(x.clone())

            x = downsample(x, enable_time=self.enable_time)

            # for each layer, we save two tensors for skip-connection

        x = self.mid_block1(x, t, enable_time=self.enable_time)
        x = self.mid_attn(x, cond=cond,enable_time=self.enable_time)
        x = self.mid_block2(x, t, enable_time=self.enable_time)

        for init_block, blocks, maybe_attention, upsample in reversed(self.ups):
            x = upsample(x, enable_time=self.enable_time)

            x = torch.cat((hiddens.pop() * self.skip_scale, x), dim=1)

            x = init_block(x, t, enable_time=self.enable_time)

            x = torch.cat((hiddens.pop() * self.skip_scale, x), dim=1)

            for block in blocks:
                x = block(x, enable_time=self.enable_time)

            if exists(maybe_attention):
                x = maybe_attention(x, cond=cond,enable_time=self.enable_time)

        x = rearrange(x, 'b c f h w -> b (c f) h w')
        x = self.transform(x)
        x = rearrange(x, 'b (c f) h w -> b c f h w', f=self.seq_length)
        x = self.conv_out(x, enable_time=self.enable_time)
        if self.training:
            losses = {}
            assert x.shape == target.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            target = target.permute(0, 2, 1, 3, 4).contiguous()
            # x = x.view(bs, self.seq_length, len(self.classes), bev_h, bev_w)
            # target = target.view(bs, self.seq_length, len(self.classes), bev_h, bev_w)
            for timesteps in range(self.seq_length):
                for index, name in enumerate(self.classes):
                    if self.loss == "xent":
                        loss = sigmoid_xent_loss(x[:, timesteps, index], target[:, timesteps, index])
                    elif self.loss == "focal":
                        loss = sigmoid_focal_loss(x[:, timesteps, index], target[:, timesteps, index])
                    else:
                        raise ValueError(f"unsupported loss: {self.loss}")
                    losses[f"class: {name}/timesteps: {timesteps}/loss: {self.loss}"] = loss
            return losses
        else:
            #x = x.permute(0, 2, 1, 3, 4).contiguous()
            return torch.sigmoid(x)