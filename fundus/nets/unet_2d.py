# encoding: utf-8
"""
2D UNet.
Author: Jason.Fang
Update time: 17/10/2022
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(

            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#https://github.com/milesial/Pytorch-UNet
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, latent_dim=512, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sigmoid = nn.Sigmoid()

        self.gem_mu = GeM()
        self.gem_var = GeM()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        mu = self.gem_mu(x5).view(x5.size(0), -1)
        var = self.gem_var(x5).view(x5.size(0), -1)
        z = self.reparameterize(mu, var)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        logits = self.sigmoid(logits)
        return [logits, mu, var, z]

    def reparameterize(self, mu: torch.tensor, logvar: torch.tensor) -> torch.tensor:
            """
            Will a single z be enough ti compute the expectation
            for the loss??
            :param mu: (Tensor) Mean of the latent Gaussian
            :param logvar: (Tensor) Standard deviation of the latent Gaussian
            :return:
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class KL_Loss(nn.Module):
    def __init__(self, reduction="batchmean"): 
        super(KL_Loss, self).__init__()

        self.kl_loss = nn.KLDivLoss(reduction=reduction)
 
    def	forward(self, unet_z, vae_z):
        
        return self.kl_loss(F.log_softmax(unet_z,dim=1), F.softmax(vae_z,dim=1))

def Dice_Loss(mask, target):
        #calculate dice loss
        N = target.size(0)
        smooth = 1
    
        mask_flat = mask.view(N, -1)
        target_flat = target.view(N, -1)
    
        intersection = mask_flat * target_flat
    
        dice_loss = 2 * (intersection.sum(1) + smooth) / (mask_flat.sum(1) + target_flat.sum(1) + smooth)
        #dice_loss = 1 - dice_loss.sum() / N
        dice_loss = dice_loss.sum() / N
    
        return dice_loss

if __name__ == "__main__":
    #for debug   
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    img = torch.rand(10, 1, 224, 224).cuda()#.to(torch.device('cuda:%d'%4))
    unet = UNet(n_channels=1, n_classes=1, latent_dim=512).cuda()
    logits, mu, var, z= unet(img)
    print(logits.size())
    print(mu.size())
    print(var.size())
    print(z.size())