import torch
import torch.nn as nn
import torch.nn.functional as F
from .ConvUNet import build_unet

#https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv1d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, eeg_size):
        return (torch.zeros(batch_size, self.hidden_dim, eeg_size, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, eeg_size, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    def __init__(self, input_dim, n_classes, hidden_dim=[2, 4, 8], kernel_size=3, num_layers=3, bias=True):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size,
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.seg = build_unet(in_ch=hidden_dim[-1], n_classes=n_classes)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            3-D Tensor either of shape (b, c, d)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        b, _, d = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send eeg size here
            hidden_state = self._init_hidden(batch_size=b, eeg_size=d)

        layer_output_list = []
        #last_state_list = []
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            h, c = self.cell_list[layer_idx](cur_layer_input, cur_state=[h, c])
            cur_layer_input = h

            layer_output_list.append(h)
            #last_state_list.append([h, c])
        out = layer_output_list[-1]
        out = self.seg(out)
        return out

    def _init_hidden(self, batch_size, eeg_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, eeg_size))
        return init_states
    
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
    x = torch.rand(8, 1, 250).to(device) #(b, c, d)
    model = ConvLSTM(input_dim=1, n_classes=1).to(device)
    out = model(x)
    print(out.shape)