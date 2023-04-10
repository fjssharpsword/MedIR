
# encoding: utf-8
"""
Net for WSI
Author: Jason.Fang
Update time: 26/09/2022
"""
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50,densenet121,mobilenet_v2
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LatentEncoding(nn.Module):
    def __init__(self, dim_in: int, dim_cov: int = 64, dim_out: int = 256):
        super().__init__()
        """
        Args:
            dim_in: scalar, input dimension of each patch and each latent covariate
            dim_cov: scalar, number of latent covariates
            dim_out: scalar, output of dimension of each patch and each latent covariate
            tnm_stage: scalar, TNM stages, 1-10
        """
        #each latent covariate is represented by a dim_in-dimension vector.
        position = torch.arange(dim_cov).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_in, 2) * (-math.log(10000.0) / dim_in))
        lce = torch.zeros(dim_cov, dim_in)
        lce[:, 0::2] = torch.sin(position * div_term)
        lce[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('lce', lce)

        #https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder
        encoder_layers = TransformerEncoderLayer(dim_in, nhead=8)
        self.trans_enc = TransformerEncoder(encoder_layers, num_layers=6)#mx1024->mx1024

        self.maxpool = nn.AdaptiveMaxPool2d((dim_cov,dim_out))
        self.avgpool = nn.AdaptiveAvgPool2d((dim_cov,dim_out))
        self.latent_linear = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, tnm: int) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [variable_seq_len, embedding_dim]
        """
        x_in = torch.cat((x,tnm*self.lce), 0)#[variable_seq_len+dim_cov, embedding_dim]
        x_in = self.trans_enc(x_in.unsqueeze(0))
        x_max = self.maxpool(x_in.unsqueeze(0)) 
        x_avg = self.avgpool(x_in.unsqueeze(0))
        x_cov = self.latent_linear(tnm*self.lce)
        x_out = torch.cat((x_max.squeeze(0), x_avg.squeeze(0), x_cov.unsqueeze(0)), 0)
        return x_out

class TMCNet(nn.Module):
    def __init__(self, n_var_vec, n_class):
        super(TMCNet, self).__init__()
        #extract variable features of patches
        self.encoder = resnet50(pretrained=False, num_classes=n_var_vec) #mx3x256x256->mx1024
        #self.encoder = mobilenet_v2(pretrained=False, num_classes=n_var_vec)
        #variable features -> Invariable features
        self.latent_encoder = LatentEncoding(n_var_vec) #mx1024->3x256x256
        #classifier
        self.classifier = densenet121(pretrained=False, num_classes=n_class)#3x256x256->FC

    def forward(self, x, tnm):
        x = self.encoder(x)
        x = self.latent_encoder(x, tnm)
        x = self.classifier(x.unsqueeze(0)) #self.dense_net_121.features(x) 
        return F.softmax(x,dim=1)

if __name__ == "__main__":
    #for debug  
    # a wsi=m*3*256*256, m is the number of patches
    x = torch.rand(64, 3, 256, 256).cuda() 
    model = TMCNet(n_var_vec=1024, n_class=32).cuda()
    #model = nn.DataParallel(model).cuda()
    out = model(x, tnm=2) #tnm =[1,10]
    print(out.size())
    