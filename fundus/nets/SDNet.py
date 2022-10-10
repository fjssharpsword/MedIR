import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.models as torchvision_models
from torchvision.models import resnet50,densenet121,mobilenet_v2

#Masked Region Encoder for Medical Image Retrieval
class SDNet(nn.Module):
    def __init__(self, num_vectors=1000):
        super(SDNet, self).__init__()

        self.encoder = resnet50(pretrained=True, num_classes=num_vectors) #b*3*224*224->b*num_vectors
        #self.encoder = densenet121(pretrained=True, num_classes=num_vectors)
        #self.encoder = mobilenet_v2(pretrained=True, num_classes=num_vectors)
        self.bn = nn.BatchNorm1d(num_vectors) #b*num_vectors->b*num_vectors
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.bn(x)

        return x

if __name__ == '__main__':
    #x = torch.rand(2, 3,14,14).cuda()
    #dconv = DynConv().cuda()
    #out = dconv(x)
    #print(out.shape)
    x = torch.rand(2, 3, 224 ,224).cuda()
    model = SDNet(num_vectors=1000).cuda()
    out = model(x)
    print(out.shape)