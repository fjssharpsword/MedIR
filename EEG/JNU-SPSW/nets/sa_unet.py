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

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1)
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

class SelfAttention_layer(nn.Module): 
    def __init__(self, in_ch=1, k=1):
        super(SelfAttention_layer, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        self.f = nn.Sequential(
            nn.Conv1d(self.in_ch, self.mid_ch, 1, 1),
            nn.BatchNorm1d(self.mid_ch),
            nn.ReLU())
        self.g = nn.Sequential(
            nn.Conv1d(self.in_ch, self.mid_ch, 1, 1),
            nn.BatchNorm1d(self.mid_ch),
            nn.ReLU())
        self.h = nn.Conv1d(self.in_ch, self.mid_ch, 1, 1)
        self.v = nn.Conv1d(self.mid_ch, self.out_ch, 1, 1)

        self.softmax = nn.Softmax(dim=-1)

        for conv in [self.f, self.g, self.h]: 
            conv.apply(weights_init)
        self.v.apply(constant_init)

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def forward(self, x):
        B, C, D = x.shape

        f_x = self.f(x).view(B, self.mid_ch, D)  # B * mid_ch * D
        g_x = self.g(x).view(B, self.mid_ch, D)  # B * mid_ch * D
        h_x = self.h(x).view(B, self.mid_ch, D)  # B * mid_ch * D

        z = torch.bmm(f_x.permute(0, 2, 1), g_x)  # B * D * D
        attn = self.softmax((self.mid_ch ** -.50) * z)

        z = torch.bmm(attn, h_x.permute(0, 2, 1))  # B * D * mid_ch
        z = z.permute(0, 2, 1).view(B, self.mid_ch, D)  # B * mid_ch * D

        z = self.v(z)
        x = torch.add(z, x) # z + x
        return x

## Kaiming weight initialisation
def weights_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass
def constant_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass

class Spatial_layer(nn.Module):#spatial attention layer
    def __init__(self):
        super(Spatial_layer, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        identity = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)*identity

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
        self.global_sa = SelfAttention_layer()
        self.local_sa = Spatial_layer()

    def forward(self, inputs):
        inputs = self.global_sa(inputs)
        #inputs = inputs - inputs.mean(dim=2).unsqueeze(1)
        
        """ Encoder """
        s1, p1 = self.e1(inputs)
        #p1 = self.local_sa(p1)
        #p1 = self.dropout(p1)

        s2, p2 = self.e2(p1)
        #p2 = self.local_sa(p2)
        #p2 = self.dropout(p2)

        s3, p3 = self.e3(p2)
        #p3 = self.local_sa(p3)
        #p3 = self.dropout(p3)

        s4, p4 = self.e4(p3)
        #p4 = self.local_sa(p4)
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