import torch.nn as nn
from torch.nn import functional as F
import torch

class EEG1DConvNet(nn.Module):
    def __init__(self, in_ch = 22, num_classes=2):
        # We optimize dropout rate in a convolutional neural network.
        super(EEG1DConvNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=32, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size = 3)

        self.dropout = nn.Dropout(p=0.2) 

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size = 3)

        self.pool3 = nn.AdaptiveAvgPool2d((32,32))
        self.fc1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.pool3(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
class EEG2DConvNet(nn.Module):
    def __init__(self, in_ch = 22, num_classes=2):
        # We optimize dropout rate in a convolutional neural network.
        super(EEG2DConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size = 3)

        self.dropout = nn.Dropout(p=0.2) 

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size = 3)

        self.pool3 = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.pool3(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
if __name__ == "__main__":

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    x = torch.rand(10, 1, 250).to(device)
    model = EEG1DConvNet(in_ch = 1, num_classes=2).to(device)
    out = model(x)
    print(out.shape)