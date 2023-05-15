import torch
import torch.nn as nn
import torch.nn.functional as F

#https://github.com/torcheeg/torcheeg/blob/main/torcheeg/models/rnn/lstm.py
class LSTMSeg(nn.Module):
    r'''
    A simple but effective long-short term memory (LSTM) network structure from the book of Zhang et al. For more details, please refer to the following information.

    - Book: Zhang X, Yao L. Deep Learning for EEG-Based Brain-Computer Interfaces: Representations, Algorithms and Applications[M]. 2021.
    - URL: https://www.worldscientific.com/worldscibooks/10.1142/q0282#t=aboutBook
    - Related Project: https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/pythonscripts/4-1-1_LSTM.py

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        dataset = DEAPDataset(io_path=f'./deap',
                    root_path='./data_preprocessed_python',
                    online_transform=transforms.ToTensor(),
                    label_transform=transforms.Compose([
                        transforms.Select('valence'),
                        transforms.Binary(5.0),
                    ]))
        model = GRU(num_electrodes=32, hid_channels=64, num_classes=2)

    Args:
        num_electrodes (int): The number of electrodes, i.e., :math:`C` in the paper. (defualt: :obj:`32`)
        hid_channels (int): The number of hidden nodes in the GRU layers and the fully connected layer. (defualt: :obj:`64`)
        num_classes (int): The number of classes to predict. (defualt: :obj:`2`)
    '''
    def __init__(self,
                 num_electrodes: int = 32,
                 hid_channels: int = 64,
                 num_classes: int = 2):
        super(LSTMSeg, self).__init__()

        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_classes = num_classes

        self.enc = nn.LSTM(input_size=num_electrodes,
                                 hidden_size=hid_channels,
                                 num_layers=2,
                                 bias=True,
                                 batch_first=True)
        
        """ Decoder """
        self.d1 = nn.ConvTranspose1d(hid_channels, 32, kernel_size=3, stride=1, padding=1)
        self.d2 = nn.ConvTranspose1d(32, 8, kernel_size=3, stride=1, padding=1)
        self.d3 = nn.ConvTranspose1d(8, 1, kernel_size=3, stride=1, padding=1)

        self.out = nn.Sigmoid()

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

        x = x.permute(0, 2, 1)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        x = self.out(x)
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
    x = torch.rand(10, 1, 250).to(device)
    model = LSTMSeg(num_electrodes = 1, hid_channels=64, num_classes=1).to(device)
    out = model(x)
    print(out.shape)