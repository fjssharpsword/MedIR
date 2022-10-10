# encoding: utf-8
"""
Spectral Decay Regularizer.
Author: Jason.Fang
Update time: 09/10/2022
"""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchvision.models import resnet50
#approximated SVD
#https://jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/
def Power_iteration(W, eps=1e-10, Ip=1):
    """
    power iteration for max_singular_value
    """
    v = torch.FloatTensor(W.size(1), 1).normal_(0, 1).cuda()
    W_s = torch.matmul(W.T, W)
    #while True:
    for _ in range(Ip):
        v_t = v
        v = torch.matmul(W_s, v_t)
        v = v/torch.norm(v)
        #if abs(torch.dot(v.squeeze(), v_t.squeeze())) > 1 - eps: #converged
        #    break

    u = torch.matmul(W, v)
    s = torch.norm(u)
    u = u/s
    #return left vector, sigma, right vector
    return u, s, v


def UpdateGrad(model, coef=1e-3, p='sd'): #
    #p=sd: spectral decay
    #p=l2: l2 weight decay
    for name, param in model.named_parameters():
        if 'conv' in name and param.grad is not None:
            if p=='sd': 
                out_channels, in_channels, ks1, ks2 = param.data.shape
                weight_mat = param.data.view(out_channels*ks1,in_channels*ks2)
                u, s, v = Power_iteration(weight_mat)
                spec_weight = s*torch.matmul(u, v.T)
                Wgrad = spec_weight.view(param.data.shape)
                param.grad += coef * Wgrad
            elif p=='l2':
                param.grad += coef*param.data
            else:
                pass
        else:
            pass
            

if __name__ == '__main__':
    x = torch.rand(2, 3, 224 ,224).cuda()
    model = resnet50(pretrained=True, num_classes=1000).cuda()
    out = model(x)
    print(out.shape)

 
