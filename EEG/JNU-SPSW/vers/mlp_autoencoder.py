import torch.nn as nn
import torch

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
        
class build_ae(nn.Module):
    def __init__(self, in_dim=250, hidden_dim=[128, 32, 8]):
        super(build_ae, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim[2], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
            """
            :param [b, 1, 250]:
            :return [b, 1, 250]:
            """
            b, c, d = x.size(0), x.size(1), x.size(2)
            # flatten
            x = x.view(b, -1)
            # encode
            x = self.encoder(x)
            # decode
            x = self.decoder(x)
            # reshape
            x = x.view(b, c, d)
    
            return x
    
if __name__ == "__main__":
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn((2, 1, 250)).to(device)
    model = build_ae(in_dim=250).to(device)
    y = model(inputs)
    print(y.shape)