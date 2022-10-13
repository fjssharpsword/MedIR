# encoding: utf-8
"""
Spectral Decay Regularizer.
Author: Jason.Fang
Update time: 09/10/2022
"""
import torch
from torchvision.models import resnet50,mobilenet_v2,inception_v3
from tensorboardX import SummaryWriter

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


def UpdateGrad(model, coef=1.0, p='sn'): #
    for name, param in model.named_parameters():
        if 'conv' in name and param.grad is not None:
            if p=='sn':#spectral norm-based weight decay
                if len(param.data.shape)==4: 
                    out_channels, in_channels, ks1, ks2 = param.data.shape
                    weight_mat = param.data.view(out_channels*ks1,in_channels*ks2)
                    #weight_mat = param.data.view(param.data.shape[0],-1) 
                    u, s, v = Power_iteration(weight_mat)
                    weight_mat_rank_one = s*torch.matmul(u, v.T)
                    #Wgrad = (weight_mat-weight_mat_rank_one).view(param.data.shape)
                    Wgrad = weight_mat_rank_one.view(param.data.shape)
                    param.grad += coef * Wgrad
            elif p=='ln':#L2 norm-based weight decay
                param.grad += coef*param.data
            else:
                pass
        else:
            pass
            

if __name__ == '__main__':

    model = inception_v3(pretrained=True, num_classes=1000).cuda()
    for name, param in model.named_parameters():
        if 'conv' in name:
            print(name +':'+ str(param.data.shape))
    """
    x = torch.rand(2, 3, 224 ,224).cuda()
    log_writer = SummaryWriter('/data/tmpexec/tb_log')

    model = resnet50(pretrained=True, num_classes=1000).cuda()
    model.train()
    out = model(x)

    for name, param in model.named_parameters():
        if "conv" in name:
            log_writer.add_histogram(name + '_data', param.clone().cpu().data.numpy(), 1)
            if param.grad is not None: #leaf node in the graph retain gradient
                log_writer.add_histogram(name + '_grad', param.grad, 1)

    
    UpdateGrad(model, coef=1e-3, p='sd')

    for name, param in model.named_parameters():
        if "conv" in name:
            log_writer.add_histogram(name + '_data', param.clone().cpu().data.numpy(), 2)
            if param.grad is not None: #leaf node in the graph retain gradient
                log_writer.add_histogram(name + '_grad', param.grad, 2)

    log_writer.close() #shut up the tensorboard
    """

 
