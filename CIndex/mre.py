import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as torchvision_models

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
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

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class DynConv(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 16):
        super(DynConv, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

    def forward(self, x):
        return self.features(x)

#Masked Region Encoder for Medical Image Retrieval
class MRE(nn.Module):
    def __init__(self, img_size =224, patch_size = 8, feat_len=1000, in_channels = 3, out_channels = 16, mask_ratio=0.75):
        super(MRE, self).__init__()

        self.glbconv = partial(torchvision_models.__dict__['resnet50'], zero_init_residual=True)(num_classes=feat_len)
        self.sigmoid = nn.Sigmoid()
        #dynamic region-aware convolution
        #self.regconv = OrderedDict([])
        #for i in range(self.num_patches):
        #    block = DynConv(in_channels, out_channels)
        #    self.regconv.update({'regblock%d' % (i + 1): block})
        """
        assert 0. < mask_ratio < 1., f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'
        self.mask_ratio = mask_ratio
        assert img_size%patch_size == 0, f'img_size must be dividede with no remainder by patch_size, got: {patch_size}'
        self.patch_size = patch_size
        self.num_patches = img_size//patch_size
        dim = patch_size*patch_size*self.num_patches*in_channels
        self.regtrans = Transformer(dim=dim, depth=6, heads=16, dim_head=64, mlp_dim=2048)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes), nn.Sigmoid())
        """
        """
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            #elif isinstance(m, nn.Linear):
            #    nn.init.constant_(m.bias, 0)
        """
    def forward(self, x):

        glb_out =  self.glbconv(x)
        return self.sigmoid(glb_out)
        """
        b, c, h, w = x.shape
        # (b, c=3, h, w)->(b, n_patches, patch_size**2 * c)
        patches = x.view(
            b, c,
            h // self.patch_size, self.patch_size, 
            w // self.patch_size, self.patch_size
        ).permute(0, 2, 4, 3, 5, 1).reshape(b, self.num_patches, -1)
        reg_out = self.regtrans(patches)

        reg_out = reg_out.mean(dim = 1)
        reg_out = self.to_latent(reg_out)
        reg_out = self.mlp_head(reg_out)

        return reg_out
        """

if __name__ == '__main__':
    #x = torch.rand(2, 3,14,14).cuda()
    #dconv = DynConv().cuda()
    #out = dconv(x)
    #print(out.shape)
    x = torch.rand(2, 3, 224 ,224).cuda()
    mre = MRE(feat_len=512).cuda()
    out = mre(x)
    print(out.shape)