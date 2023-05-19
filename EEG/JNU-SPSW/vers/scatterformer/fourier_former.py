from itertools import filterfalse
import math
from pickle import FALSE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scatnet_learn import InvariantLayerj1, ScatLayerj1, InvariantLayerj1_dct, InvariantLayerj2
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
from fourier import FFC
from torch.utils.data.dataset import Dataset
import os
import random
from fourier import SpectralTransform

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Channel_Layernorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class Mlp(nn.Module):
    def __init__(self, act_layer, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()

        # depthwise in between, a choice
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FourierPatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel=(3, 3),
                 stride=(2, 2),
                 padding=1, combinations="first_order"):
        super().__init__()
        self.combinations = combinations
        if self.combinations == "second_order":
            self.proj1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 2,
                                                 kernel_size=(3, 3), stride=(2, 2), padding=1),
                                       nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels,
                                                kernel_size=(3, 3), stride=(2, 2), padding=1)
                                      )

        elif self.combinations == "first_order":
            self.proj1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=(3, 3), stride=(2, 2), padding=1)
        elif self.combinations == "inv_second":
            self.proj1 = nn.Sequential(SpectralTransform(in_channels, out_channels // 2, stride=2),
                                       SpectralTransform(out_channels // 2, out_channels, stride=2))
        elif self.combinations == "inv_first":
            self.proj1 = nn.Sequential(SpectralTransform(in_channels, out_channels, stride=2))
        #  self.proj2 =
        elif self.combinations == "inv_second_full":
            self.proj1 = InvariantLayerj1(in_channels, in_channels * 7, alpha='full')
            self.proj2 = InvariantLayerj1(in_channels * 7, out_channels, alpha='full')
        elif self.combinations == "first_full":
            self.proj1 = InvariantLayerj1(in_channels, (in_channels + out_channels) // 2, alpha='full')
            self.proj2 = InvariantLayerj1((in_channels + out_channels) // 2, out_channels, alpha='full', stride=1)
            # nn.Sequential(InvariantLayerj1(C=in_channels, F=out_channels, stride=2))

    def forward(self, x, interms):
        #    if self.combinations == "second_order" or "first_order":
        #       x = self.proj(x)
        #       return x, interms
        #if "inv" in self.combinations:
        x = self.proj1(x)
      #  interms.append(x)
        return x, interms
    #  interms.append(x)
    #  return x, interms

class FourierAttention(nn.Module):
    def __init__(self, emb_size, num_heads,
                 attn="vanilla",
                 use_rpe=False,
                 use_upsampling="bilinear",
                 alpha=None,
                 dropout=0.,
                 is_high=0):
        # alpha = None: 1*1 conv
        # alpha = "full": 3*3 conv
        # alpha = "w/o": no conv
        # alpha = "order2": order2 wavelet scattering transform

        # now using a simpler layer!!
        # use DWconv intead of Conv
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.scale = (emb_size // num_heads) ** (-0.5)
        if alpha is None or alpha == "full":
            #    self.q = nn.Sequential(nn.Conv2d(emb_size, emb_size, (3, 3), (1, 1), 1),
            #                         nn.BatchNorm2d(emb_size))
            self.inv_q = nn.Sequential(SpectralTransform(emb_size // 2, emb_size // 2, stride=2),
                                       nn.BatchNorm2d(emb_size // 2))
            self.v = nn.Sequential(nn.Conv2d(emb_size // 2, emb_size // 2, (3, 3), padding=1),
                                   nn.BatchNorm2d(emb_size // 2))
            self.k = nn.Sequential(nn.Conv2d(emb_size // 2, emb_size // 2, (3, 3), padding=1),
                                   nn.BatchNorm2d(emb_size // 2))
            self.q = nn.Sequential(nn.Conv2d(emb_size // 2, emb_size // 2, (3, 3), padding=1),
                                   nn.BatchNorm2d(emb_size // 2))
            self.k2 = nn.Sequential(nn.Conv2d(emb_size // 2, emb_size // 2, (3, 3), (2, 2), padding=1),
                                    nn.BatchNorm2d(emb_size // 2))
            self.v2 = nn.Sequential(nn.Conv2d(emb_size // 2, emb_size // 2, (3, 3), (2, 2), padding=1),
                                    nn.BatchNorm2d(emb_size // 2))
        #  self.k_gaussian_blurring = InvariantLayerj1(C=emb_size, F=emb_size, alpha=alpha)
        #   self.v_gaussian_blurring = InvariantLayerj1(C=emb_size, F=emb_size, alpha=alpha)

        #  self.H = 0
        #  self.W = 0
        #  self.blurring = GaussianBlur(channels=emb_size)

        self.attn_drop = nn.Dropout(dropout)

        self.attn_type = attn
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        self.scaling = (self.emb_size // num_heads) ** -0.5

        self.use_rpe = use_rpe
        self.rpe = nn.Conv2d(emb_size, emb_size, (3, 3), (1, 1), 1, groups=emb_size)

        self.conv = nn.Conv2d(emb_size, emb_size, (1, 1))

        self.use_upsampling = use_upsampling

        #    if self.use_upsampling == "bilinear":
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        #  self.proj = nn.Linear(emb_size, emb_size)
        self.is_high = is_high

    def forward(self, x, interms):

        # x = rearrange(x, 'b (h w) e ->  b e h w')
        B, E, H, W = x.shape
       # self.H, self.W = H, W
        x1 = x[:, 0:E // 2, :, :]
        x2 = x[:, E // 2:E, :, :]

        q_l = self.q(x1)
        k_l, v_l = self.k(x1), self.v(x1)
        q_h = self.inv_q(x2)
        k_h, v_h = self.k2(x2), self.v2(x2)

        q_h = rearrange(q_h, 'b c h w -> b (h w) c')
        q_l, k_h, v_h, k_l, v_l = rearrange(q_l, 'b c h w -> b (h w) c'), \
                                  rearrange(k_h,
                                            'b c h w -> b (h w) c'), rearrange(
            v_h,
            'b c h w -> b (h w) c'), \
                                  rearrange(k_l, 'b c h w -> b (h w) c'), rearrange(v_l, 'b c h w -> b (h w) c')
        k_l = rearrange(k_l, 'b t (h d)-> b h t d', h=self.num_heads)
        v_l = rearrange(v_l, 'b t (h d)-> b h t d', h=self.num_heads)
        k_h = rearrange(k_h, 'b t (h d)-> b h t d', h=self.num_heads)
        v_h = rearrange(v_h, 'b t (h d)-> b h t d', h=self.num_heads)
        q_l = rearrange(q_l, 'b t (h d) -> b h t d', h=self.num_heads)
        q_h = rearrange(q_h, 'b t (h d) -> b h t d', h=self.num_heads)
        attn_h = self.attn_drop(
            (F.normalize(q_h.transpose(-2, -1), dim=-1) @ F.normalize(k_h, dim=-1)) * self.temperature).softmax(dim=-1)
        attn_h = (attn_h @ v_h.transpose(-2, -1)).permute(0, 3, 1, 2)

        del v_h, k_h, q_h
        attn_l = self.attn_drop(
            (F.normalize(q_l.transpose(-2, -1), dim=-1) @ F.normalize(k_l, dim=-1)) * self.temperature).softmax(dim=-1)
        attn_l = (attn_l @ v_l.transpose(-2, -1)).permute(0, 3, 1, 2)
        del k_l, v_l, q_l
        attn_l = rearrange(attn_l, 'b t h d -> b t (h d)')
        attn_l = rearrange(attn_l, 'b (h w) c ->  b c h w', h=H, w=W)
        attn_h = rearrange(attn_h, 'b t h d -> b t (h d)')
        attn_h = rearrange(attn_h, 'b (h w) c ->  b c h w', h=H // 2, w=W // 2)
        attn_h = self.upsampling(attn_h)
      #  interms.append(attn_l)
      #  interms.append(attn_h)
        interms.append(attn_h)
        interms.append(attn_l)
        attn = torch.cat([attn_h, attn_l], dim=1)
        interms.append(attn_h)

        del attn_h, attn_l

        attn = self.attn_drop(self.conv(attn))

        if self.use_rpe:
            attn = attn + self.rpe(x)
     #       interms.append(rearrange(attn, 'b c h w ->  b c (h w)'))
            return attn, interms  # + self.rpe(x), interms

        else:
            return attn, interms


        # attn_h = self.proj(attn_h)

class FourierEncoder(nn.Module):
    def __init__(self, dim,
                 num_heads=8, mlp_ratio=4,
                 drop=0., attn_drop=0.,
                 drop_path=0., use_rpe=True, use_upsampling="bilinear", alpha=None,
                 act_layer=nn.Mish,
                 norm_layer=Channel_Layernorm, is_high=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FourierAttention(dim, num_heads,
                                        use_rpe=use_rpe, use_upsampling=use_upsampling,
                                        alpha=alpha, dropout=attn_drop, is_high=is_high)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, interms):
        r = x
        x, interms = self.attn(self.norm1(x), interms)
        x = r + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
       # interms.append(rearrange(x, 'b c h w ->  b c (h w)'))
    #    interms.append(x)
        return x, interms


class FourierFormer(nn.Module):
    def __init__(self, mode="train", act=nn.Mish):
        super().__init__()

        self.patch_merg_convs = nn.ModuleList([])
        self.blocks = nn.ModuleList([])
        self.mode = mode
        dims = [96, 192, 384, 512]
        num_conv_encoders = [2, 2, 8, 2]
        heads = [8, 8, 8, 8]
        kernels = [3, 5, 7, 9]
        self.patch_merg_convs.append(FourierPatchEmbedding(in_channels=3, out_channels=dims[0],
                                                              combinations="inv_second"))
        self.patch_merg_convs.append(FourierPatchEmbedding(in_channels=dims[0], out_channels=dims[1],
                                                              combinations="inv_first"))
        self.patch_merg_convs.append(FourierPatchEmbedding(in_channels=dims[1], out_channels=dims[2],
                                                              combinations="inv_first"))
        self.patch_merg_convs.append(FourierPatchEmbedding(in_channels=dims[2], out_channels=dims[3],
                                                              combinations="inv_first"))
        #   i = 0
        #  layers = nn.ModuleList(
        #      [InvariantScatteringEncoder(emb_size=dims[i], num_heads=heads[i], kernel_size=kernels[i]) for _ in
        #       range(num_conv_encoders[i])])
        #  self.blocks.append(layers)
        #  del i
        for i in range(4):
            layers = nn.ModuleList(
                [FourierEncoder(dim=dims[i], act_layer=act, num_heads=heads[i]) for _ in range(num_conv_encoders[i])])
            # layers.append(ScatteringEncoder(dim=dims[i], num_heads=heads[i]))
            self.blocks.append(layers)

        self.proj = nn.Linear(dims[-1], 2)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        interms = []
        idx = 0
        if self.mode == "train":
            for patch_merg_conv, blocks in zip(self.patch_merg_convs, self.blocks):
                idx += 1
                x, interms = patch_merg_conv(x, interms)
                interms.clear()
                for block in blocks:
                    x, interms = block(x, interms)
                interms.clear()

            x = self.softmax(self.proj(x.mean([2, 3])))
            return x  # , interms

        elif self.mode == "inferrence":
            interms = []
            idx = 0
            for patch_merg_conv, blocks in zip(self.patch_merg_convs, self.blocks):
                idx += 1
                x, interms = patch_merg_conv(x, interms)
                for block in blocks:
                    x, interms = block(x, interms)

            x = self.softmax(self.proj(x.mean([2, 3])))
            return x, interms

def get_fourier(mode="train", act=nn.Mish):
    return FourierFormer(mode=mode, act=act)

#https://github.com/albertcheng19/scatterformer/blob/main/fourier_former.py
if __name__ == "__main__":
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn((8, 1, 250)).to(device)
    model = FourierFormer(mode="inferrence").to(device)
    y = model(inputs)
    print(y.shape)