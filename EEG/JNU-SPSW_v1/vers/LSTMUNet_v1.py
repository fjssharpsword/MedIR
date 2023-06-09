import torch
import torch.nn as nn
import torch.nn.functional as F
from .ConvUNet import build_unet
#https://github.com/torcheeg/torcheeg/blob/main/torcheeg/models/rnn/lstm.py
class LSTMUNet(nn.Module):
    def __init__(self,
                 num_electrodes: int = 32,
                 hid_channels: int = 64,
                 num_classes: int = 2):
        super(LSTMUNet, self).__init__()

        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_classes = num_classes

        self.enc = nn.LSTM(input_size=num_electrodes,
                                 hidden_size=hid_channels,
                                 num_layers=2,
                                 bias=True,
                                 batch_first=True)
        #self.dec = nn.LSTM(input_size=hid_channels+1, hidden_size=num_classes,num_layers=2,bias=True,batch_first=True)
        #self.dense = nn.Conv1d(hid_channels, num_classes, kernel_size=1, padding=0)
        #
        self.seg = build_unet(in_ch=hid_channels, n_classes=num_classes)
        """ Decoder """
        #self.d1 = nn.ConvTranspose1d(hid_channels, 32, kernel_size=1, padding=0)
        #self.d2 = nn.ConvTranspose1d(32, 8, kernel_size=1, padding=0)
        #self.d3 = nn.ConvTranspose1d(8, num_classes, kernel_size=1, padding=0)

        #self.out = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 32, 128]`. 
            Here, :obj:`n` corresponds to the batch size, 
                  :obj:`32` corresponds to :obj:`num_electrodes`
                  :obj:`128` corresponds to the number of data points included in the input EEG chunk.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = x.permute(0, 2, 1)
        x, (_, _) = self.enc(x, None)
        #x = x[:, -1, :]
        x = x.permute(0, 2, 1)
        x = self.seg(x)
        return x
    
if __name__ == "__main__":
    """
    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = rnn(input, (h0, c0))
    print(output)
    """
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    x = torch.rand(2, 1, 250).to(device)
    model = LSTMUNet(num_electrodes = 1, hid_channels=8, num_classes=1).to(device)
    out = model(x)
    print(out.shape)