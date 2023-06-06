import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np
from torchvision.transforms import GaussianBlur

class Conv2d_DoG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_DoG, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.weight = torch.cat((torch.FloatTensor([[1, 2, 1]]), torch.FloatTensor([[2, 4, 2]]), torch.FloatTensor([[1, 2, 1]])), 0) / 16

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        self.weight = torch.ones(C_in, C_in, H_k, W_k) * self.weight
        conv_weight = F.conv2d(self.conv.weight, self.weight.cuda(), stride=1, padding=1)
        out_normal = F.conv2d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            #pdb.set_trace()
            # kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = conv_weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class Conv2d_LoG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_LoG, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.weight1 = torch.cat((torch.FloatTensor([[1, 2, 1]]), torch.FloatTensor([[2, 4, 2]]), torch.FloatTensor([[1, 2, 1]])), 0) / 16
        self.weight2 = torch.cat(
            (torch.FloatTensor([[-1, -1, -1]]), torch.FloatTensor([[-1, 8, -1]]), torch.FloatTensor([[-1, -1, -1]])), 0) / 16

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape
        self.weight1 = torch.ones(C_in, C_in, H_k, W_k) * self.weight1
        self.weight2 = torch.ones(C_in, C_in, H_k, W_k) * self.weight2

        conv_weight1 = F.conv2d(self.conv.weight, self.weight1.cuda(), stride=1, padding=1)
        conv_weight2 = F.conv2d(conv_weight1, self.weight2.cuda(), stride=1, padding=1)

        out_normal = F.conv2d(input=x, weight=conv_weight2, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        return out_normal


class Conv2d_RoG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv2d_RoG, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.DoG = Conv2d_DoG(in_channels, out_channels, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.LoG = Conv2d_LoG(in_channels, out_channels, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        #self.conv_ = nn.Conv2d(out_channels*3, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x):
        [C_out, C_in, H_k, W_k] = self.conv.weight.shape

        x_DoG = self.DoG(x)
        x_LoG = self.LoG(x)
        x_res = self.conv(x)
        # print("x_res", x_res.shape)
        # out = x_DoG + x_LoG + x_res
        out = torch.cat((x_DoG, x_LoG, x_res), 1)
        #out = self.conv_(out)

        return out
