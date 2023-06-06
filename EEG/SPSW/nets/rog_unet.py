import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):#smooth=1e-5
        super(DiceLoss, self).__init__()

        self.smooth = smooth
            
    def	forward(self, input, target):
        N = target.size(0)
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
    
        intersection = input_flat * target_flat
    
        loss = 2 * (intersection.sum(1) + self.smooth) / (input_flat.sum(1) + target_flat.sum(1) + self.smooth)
        loss = 1 - loss.sum() / N
        #loss = loss.sum() / N
        return loss

class Conv1d_DoG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv1d_DoG, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.weight = torch.FloatTensor([1, 2, 1]) / 16

    def forward(self, x):
        device = x.get_device()
        [C_out, C_in, K] = self.conv.weight.shape
        self.weight = torch.ones(C_in, C_in, K) * self.weight
        conv_weight = F.conv1d(self.conv.weight, self.weight.to(device), stride=1, padding=1)
        out_normal = F.conv1d(input=x, weight=conv_weight, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            kernel_diff = conv_weight.sum(2)
            kernel_diff = kernel_diff[:, :, None]
            out_diff = F.conv1d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class Conv1d_LoG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv1d_LoG, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.weight1 = torch.FloatTensor([1, 2, 1]) / 16
        self.weight2 = torch.FloatTensor([-1, 8, -1]) / 16

    def forward(self, x):
        device = x.get_device()
        [C_out, C_in, K] = self.conv.weight.shape
        self.weight1 = torch.ones(C_in, C_in, K) * self.weight1
        self.weight2 = torch.ones(C_in, C_in, K) * self.weight2

        conv_weight1 = F.conv1d(self.conv.weight, self.weight1.to(device), stride=1, padding=1)
        conv_weight2 = F.conv1d(conv_weight1, self.weight2.to(device), stride=1, padding=1)

        out_normal = F.conv1d(input=x, weight=conv_weight2, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)

        return out_normal

class Conv1d_RoG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(Conv1d_RoG, self).__init__()
        
        self.theta = theta
        self.DoG = Conv1d_DoG(in_channels, out_channels, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.LoG = Conv1d_LoG(in_channels, out_channels, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.conv = nn.Conv1d(out_channels*2, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        #x_DoG = self.DoG(x)
        x_LoG = self.LoG(x)
        #out = torch.cat((x_DoG, x_LoG), 1)
        #out = self.conv(out)
        out = x_LoG
        return out
        
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = Conv1d_RoG(in_c, out_c, kernel_size=3, padding=1) #nn.Conv1d
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = Conv1d_RoG(out_c, out_c, kernel_size=3, padding=1) #nn.Conv1d
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool1d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)

        # input is CHW
        diff = skip.size()[2] - x.size()[2]

        x = F.pad(x, [diff // 2, diff - diff // 2])
    
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class build_unet(nn.Module):
    def __init__(self, in_ch =1, n_classes=1):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(in_ch, 16)
        self.e2 = encoder_block(16, 32)
        self.e3 = encoder_block(32, 64)
        self.e4 = encoder_block(64, 128)
        """ Bottleneck """
        self.b = conv_block(128, 256)
        """ Decoder """
        self.d1 = decoder_block(256, 128)
        self.d2 = decoder_block(128, 64)
        self.d3 = decoder_block(64, 32)
        self.d4 = decoder_block(32, 16)
        """ Classifier """
        self.outputs = nn.Conv1d(16, n_classes, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        #self.dropout = nn.Dropout(p=0.1) 

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        #p1 = self.dropout(p1)
        s2, p2 = self.e2(p1)
        #p2 = self.dropout(p2)
        s3, p3 = self.e3(p2)
        #p3 = self.dropout(p3)
        s4, p4 = self.e4(p3)
        #p4 = self.dropout(p4)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        outputs = self.sigmoid(outputs)
        return outputs
    
#https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
if __name__ == "__main__":
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn((8, 1, 250)).to(device)
    model = build_unet(n_classes=1).to(device)
    y = model(inputs)
    print(y.shape)